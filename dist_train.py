import argparse
import json
import shutil
import torch
import random
import os
from tqdm import tqdm
from utils.distributed import distributed_init, is_master, synchronize, distributed_cleanup
import torch.distributed as dist
import time

from easydict import EasyDict as edict

from utils.utils import (
    clear_logger,
    format_time,
    get_config,
    get_logger_and_tb_writer,
    load_model,
)

import utils.layer_wise_lr_decay as lrd
import utils.lr_sched as lr_sched
from utils.layer_wise_lr_decay import create_optimizer

from dataset.dataset import DemonstratorProficiencyDataset

from dataset.dataset_video import DemonProfVideoDataset

from models.loss import build_criterion
from models.ego_video_resnet import EgoVideoResnet3D
from models.ego_fusion import EgoFusion

import val
import test
import inference_for_visual

from utils.utils import AverageMeter

# def custom_collate_fn(batch):
#     shapes = [item["inputs"].shape for item in batch]  # 获取每个 item 的形状
#     unique_shapes = set(shapes)  # 获取所有不同的形状

#     if len(unique_shapes) > 1:
#         print("警告：批次中存在形状不一致的样本！")
#         for idx, shape in enumerate(shapes):
#             print(f"样本 {idx} 的形状: {shape}")
#         1

#     return torch.utils.data.dataloader.default_collate(batch)

def get_train_loader(config, logger, is_distributed=False):
    dataset_name = config.DATASET.get("NAME", "DemonstratorProficiencyDataset")
    split = config["TRAIN"].get("SPLIT", "train")
    dataset = eval(dataset_name)(config, split=split, logger=logger)

    if is_distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=config["TRAIN"]["SHUFFLE"])
        shuffle = False
    else:
        sampler = None
        shuffle = config["TRAIN"]["SHUFFLE"]

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["TRAIN"]["BATCH_SIZE"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config["WORKERS_DATALOADER"],
        drop_last=True,
        pin_memory=True,
        # collate_fn=custom_collate_fn,
    )
    return train_loader

def train(
    config,
    device,
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    logger
):
    model.train()
    # loss_total = 0

    time_total = 0
    time_forward = 0
    time_backward = 0
    time_dataloader = 0

    layer_decay = config.TRAIN.get("LAYER_DECAY", 1.0)

    # Initialize tqdm
    progress_bar = tqdm(train_loader, ncols=80, position=0)
    progress_bar.set_description("Training")

    loss_total = AverageMeter()
    data_time = AverageMeter()
    model_time = AverageMeter()
    train_time = AverageMeter()

    preds = []
    labels = []

    t0 = time.time()

    for index, data_dir in enumerate(progress_bar):

        t1 = time.time()
        data_time.update(t1 - t0)
        
        # Adjust learning rate
        if layer_decay < 1.0: 
            # we use a per iteration (instead of per epoch) lr scheduler for VideoMAEv2
            lr_args = edict()
            lr_args.lr = config.TRAIN.LR
            lr_args.warmup_epochs = config.TRAIN.WARMUP_EPOCH # 5
            lr_args.min_lr = config.TRAIN.LR_MIN
            lr_args.epochs = config.TRAIN.END_EPOCH
            lr = lr_sched.adjust_learning_rate(optimizer, index / len(train_loader) + epoch, lr_args)
            # print("learning rate =", lr)
        else:
            lr = optimizer.param_groups[0]['lr']

        # parse data from dataloader
        inputs_image = data_dir["inputs_image"].to(device)
        inputs_pose = data_dir["inputs_pose"].to(device)    
        label = data_dir["label"].to(device)

        # forward
        pred = model(inputs_image, inputs_pose)

        loss = criterion(pred, label)
        # loss_total += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total.update(loss.item())
        
        preds.append(pred)
        labels.append(label)

        # update tqdm info
        if index == len(train_loader) - 1:
            loss_name = config["TRAIN"]["LOSS_CRITERION"]
            progress_bar.set_postfix({})
        else:
            loss_name = config["TRAIN"]["LOSS_CRITERION"]
            progress_bar.set_postfix({f"{loss_name}": loss.item()})

        t2 = time.time()
        model_time.update(t2-t1)

        t3 = time.time()
        train_time.update(t3-t0)

        t0 = time.time()
        
        if is_master():

            if index % 50 == 1:
                msg = (
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Lr  {Lr:.6f}\t"
                    "CE Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "data: {data_time:.4f}\t"
                    "model: {model_time:.4f}\t"
                    "train: {train_time:.4f}".format(
                        epoch, 
                        index + 1, 
                        len(train_loader), 
                        loss=loss_total, 
                        Lr=lr, 
                        data_time=data_time.avg,
                        model_time=model_time.avg, 
                        train_time=train_time.avg
                    )
                )
                logger.info(msg)

    train_loss = loss_total.avg
    
    # compute the accuracy
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = torch.argmax(preds, dim=1)
    acc = torch.sum(preds == labels).item() / len(labels)

    return train_loss, acc


def main(args):

    config = get_config(args)
    logger, tb_writer = get_logger_and_tb_writer(config, split="train")

    if is_master():

        shutil.copyfile(args.config_path, os.path.join(config["OUTPUT_DIR_TRAIN"], os.path.basename(args.config_path)))
        logger.info(f"config: {json.dumps(config, indent=4)}")
        logger.info(f"args: {json.dumps(vars(args), indent=4)}")
        tb_writer.add_text("config", str(config))
        tb_writer.add_text("args", str(args))

    ################## dataloader ##################
    train_loader = get_train_loader(config, logger, is_distributed=args.dist)
    val_loader = val.get_val_loader(config, logger, is_distributed=args.dist)

    ################## device ##################
    # conside multi-gpu training and validation
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")


    
    ################## model ##################

    model = eval(config.SELECT_MODEL)(config)

    if is_master():
        logger.info(f"device: {device}")
        logger.info(f"Use model: {config['SELECT_MODEL']}")
        logger.info(model)

    if config["TRAIN"]["PRETRAINED_MODEL"] is not None:
        model_path = config["TRAIN"]["PRETRAINED_MODEL"]
        if is_master():
            logger.info(f"Load pretrained model: {model_path}")
        try:
            model.load_state_dict(load_model(model_path), strict=False)
        except:
            if is_master():
                logger.info(f"Could not load pretrained model: {model_path}")   
    
    model_without_ddp = model    
    model = model.to(device)

    if args.dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.rank],
            output_device=args.rank,
            find_unused_parameters=False #True
        )

    ################## criterion  ##################

    criterion = build_criterion(config)
    criterion = criterion.to(device)

    ################## optimizer ##################
    layer_decay = config.TRAIN.get("LAYER_DECAY", 1.0)
    optimizer, scheduler = create_optimizer(config, model_without_ddp)


    ################## Train model & validation & save best ##################
    flag_have_pretrained_model = False
    if flag_have_pretrained_model:
    # evaluate on validation dataset
        val_loss, val_acc, _ = val.val(
            config,
            device,
            val_loader,
            model,
            criterion,
        )
        if is_master():
            logger.info(f"Epoch: init: val_loss: {val_loss}")
        
    best_val_loss = 1e1000
    best_val_acc = 0.0

    model_path_best_acc = ""
    model_path_best_loss = ""
    model_path_final_epoch = ""

    synchronize()
    
    for epoch in range(config["TRAIN"]["BEGIN_EPOCH"], config["TRAIN"]["END_EPOCH"]):
        # train for one epoch
        train_loss, train_acc = train(
            config,
            device,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            logger
        )
        
        # 记录当前学习率到 TensorBoard
        current_lr = optimizer.param_groups[0]["lr"]

        if is_master():
            tb_writer.add_scalar("train/lr", current_lr, epoch + 1)

        # 更新学习率
        # if config["TRAIN"]["OPTIMIZER"] == "AdamW":
        if layer_decay == 1.0: 
            scheduler.step()

        # evaluate on validation set

        if is_master():
            val_loss, val_acc, val_results = val.val(
                config,
                device,
                val_loader,
                model,
                criterion,
            )
        else: 
            val_loss = 9999999
            val_acc = 0

        # # 收集所有进程的结果
        # all_results = [None for _ in range(args.world_size)]
        # dist.all_gather_object(all_results, val_results)

        dist.barrier()  # 所有进程在此等待，直到全部到达      
        if is_master():
            # Save best model weight by val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                # Save model weight
                if os.path.isfile(model_path_best_loss):
                    os.remove(model_path_best_loss)
                model_path_best_loss = os.path.join(
                    config["OUTPUT_DIR_TRAIN"],
                    f"{config['CONFIG_NAME'].split('_')[0]}-best-e{epoch+1}-train-{train_loss:.2f}-val-{val_loss:.2f}.pt",
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    model_path_best_loss,
                )

            # Save best model weight by val acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc

                # Save model weight
                if os.path.isfile(model_path_best_acc):
                    os.remove(model_path_best_acc)
                model_path_best_acc = os.path.join(
                    config["OUTPUT_DIR_TRAIN"],
                    f"{config['CONFIG_NAME'].split('_')[0]}_best-e{epoch+1}_val-acc-{val_acc:.2f}.pt",
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                    },
                    model_path_best_acc,
                )
            

            logger.info(
                f"Epoch: {epoch+1}, train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}, "
                f"val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f} "
                f"best_val_loss: {best_val_loss:.3f}"
            )
            logger.info(f"Epoch: {epoch+1}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, best_val_loss: {best_val_loss:.4f}")
            tb_writer.add_scalar("train/loss", train_loss, epoch + 1)
            tb_writer.add_scalar("val/loss", val_loss, epoch + 1)

            # Save model weight every 10 epoch
            save_interval = config["TRAIN"].get("SAVE_INTERVAL", 10)
            if (epoch + 1) % save_interval == 0 or epoch == config["TRAIN"]["END_EPOCH"] - 1:
                if os.path.isfile(model_path_final_epoch):
                    os.remove(model_path_final_epoch)
                model_path_final_epoch = os.path.join(
                    config["OUTPUT_DIR_TRAIN"],
                    f"{config['CONFIG_NAME'].split('_')[0]}-final-e{epoch+1}-train-{train_loss:.2f}-val-{val_loss:.2f}.pt",
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    model_path_final_epoch,
                )
                tqdm.write(f"Save model: {model_path_final_epoch}")
    
        dist.barrier()  # 所有进程在此等待，直到全部到达     
    
    ################## Test ##################
    if is_master():
        # test on best model weight
        if os.path.isfile(model_path_best_loss):
            tmp_args = argparse.Namespace(
                config_path=args.config_path,
                data_dir=args.data_dir,
                model_path=model_path_best_loss,
            )
            test.main(tmp_args, logger)

        if 0:
            # test on final model weight
            if os.path.isfile(model_path_final_epoch):
                tmp_args = argparse.Namespace(
                    config_path=args.config_path,
                    data_dir=args.data_dir,
                    model_path=model_path_final_epoch,
                )
                test.main(tmp_args, logger)
        
        if 0:
            ################## Visual ##################
            # visual on best model weight
            if os.path.isfile(model_path_best_loss):
                tmp_args = argparse.Namespace(
                    config_path=args.config_path,
                    data_dir=args.data_dir,
                    model_path=model_path_best_loss,
                    take_num_train=20,
                    take_num_val=20,
                    take_num_test=20,
                )
                inference_for_visual.main(tmp_args, logger)
            
            # test on final model weight
            if os.path.isfile(model_path_final_epoch):
                tmp_args = argparse.Namespace(
                    config_path=args.config_path,
                    data_dir=args.data_dir,
                    model_path=model_path_final_epoch,
                    take_num_train=20,
                    take_num_val=20,
                    take_num_test=20,
                )
                inference_for_visual.main(tmp_args, logger)
    
    ################### end ###############
    if is_master():
        tb_writer.close()
        clear_logger(logger)
        
        # rename the folder add the best loss
        os.rename(
            config["OUTPUT_DIR_TRAIN"],
            config["OUTPUT_DIR_TRAIN"] + f"-{best_val_loss:.4f}",
        )


def distributed_main(device_id, args):
    args.rank = args.start_rank + device_id
    args.device_id = device_id
    print(f"Start device_id = {device_id}, rank = {args.rank}")
    
    torch.cuda.set_device(args.device_id)
    # torch.cuda.init()
    
    distributed_init(args)
    torch.cuda.empty_cache()

    print("MASTER_ADDR =", os.environ['MASTER_ADDR'])
    print("MASTER_PORT =", os.environ['MASTER_PORT'])
    
    # 设置 CuDNN 为确定性模式
    torch.backends.cudnn.deterministic = True
    # 关闭 CuDNN 自动寻找最优算法的功能
    torch.backends.cudnn.benchmark = False

    main(args)

    distributed_cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default=r"config/dev.yaml",
        help="Config file path of egoexo4D-demonstrator-proficiency",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Data root dir of egoexo4D-demonstrator-proficiency",
    )
    parser.add_argument('--dist', action='store_true', help='Launch distributed training')
    parser.add_argument('--world_size', type=int, default=1, help='Distributed world size')
    parser.add_argument('--init_method', type=str, help='Distributed init method')
    parser.add_argument('--backend', type=str, default='nccl', help='Distributed backend')
    parser.add_argument('--rank', type=int, default=0, help='Rank id')
    parser.add_argument('--start_rank', type=int, default=0, help='Start rank')
    parser.add_argument('--device_id', type=int, default=0, help='Device id')
    args = parser.parse_args()
    # dist training
    args.dist = True
    if args.dist:
        args.world_size = max(1, torch.cuda.device_count())
        print("World size =", args.world_size)

        if args.world_size > 1:
            port = random.randint(10000, 30000)
            args.init_method = f"tcp://localhost:{port}"
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args,),
                nprocs=args.world_size,
            )
        else:
            print("Distributed training is not enabled - not enough GPUs!")  
    else:
        main(args)
