import torch.nn as nn
import torch
import einops
import torch.nn.functional as F

# from models.output_head_transformer import OutputHeadTransformer
from models.backbone.build_backbone import build_backbone

# class CustomTransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=True):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.norm1 = nn.BatchNorm1d(d_model)  # 使用 BatchNorm 替代 LayerNorm
#         self.norm2 = nn.BatchNorm1d(d_model)  # 使用 BatchNorm 替代 LayerNorm
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.activation = nn.ReLU()
        
#     def forward(self, src, src_mask=None, src_key_padding_mask=None):
#         src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
#         src = src + self.dropout1(src2)
#         src = self.norm1(src.permute(0, 2, 1)).permute(0, 2, 1)  # 注意 BatchNorm 的输入维度
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src.permute(0, 2, 1)).permute(0, 2, 1)  # 注意 BatchNorm 的输入维度
#         return src


class EgoVideoResnet3D(nn.Module):
    def __init__(self, config):
        super().__init__()

        print('Building EgoVideoR3D...')    

        self.temporal_fusion = config.MODEL.get("TEMPORAL_FUSION", "mean")

        self.backbone_cfg = config["MODEL"]["BACKBONE"] 
        self.backbone = build_backbone(config=self.backbone_cfg)
        self.feat_dim = config.MODEL.BACKBONE.EMBED_DIM
        # if self.temporal_fusion == 'transformer_encoder':
        #     encoder_layer = nn.TransformerEncoderLayer(self.img_feat_dim, nhead=nhead, batch_first=True, dropout=dropout_rate)
        #     self.img_temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer) 
        pooling = config.MODEL.get("POOLING", "avg")
        if pooling == "avg":
            self.pooling = nn.AdaptiveAvgPool3d(1)
        elif pooling == "max":
            self.pooling = nn.AdaptiveMaxPool3d(1)
        else:
            self.pooling = nn.Identity()

        # Pose encoder
        self.use_pose = config.DATASET.get("USE_POSE_INPUT", False)
        if self.use_pose:
            pose_cfg = config.MODEL.POSE_BACKBONE
            input_dim = pose_cfg["INPUT_DIM"]
            embed_dim = pose_cfg["EMBED_DIM"]
            nhead = pose_cfg["NHEAD"]
            num_layer = pose_cfg["NUM_LAYER"]
            dropout_rate = pose_cfg.get("DROPOUT", 0)
            self.pose_proj = nn.Linear(input_dim, embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead, batch_first=True, dropout=dropout_rate)
            self.pose_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer) 
        else:
            self.pose_proj = None
            self.pose_encoder = None
            
        # Head
        output_head_cfg = config['MODEL']["OUTPUT_HEAD"]
        self.head_type = output_head_cfg["TYPE"]
        if output_head_cfg["TYPE"] == "mlp":
            feature_dim = output_head_cfg["FEATURE_DIM"]
            hidden_dim = output_head_cfg["HIDDEN_DIM"]
            dropout_rate = output_head_cfg["DROPOUT"]
            num_classes = output_head_cfg["NUM_CLASSES"]
            # embed_dim = embed_dim + img_feat_dim
            self.stabilizer = nn.Sequential(
                        nn.Linear(feature_dim, hidden_dim),
                        nn.ReLU(),
                        # nn.Dropout(dropout_rate),
                        nn.Linear(hidden_dim, num_classes)
            )            
        # elif output_head_cfg["TYPE"] == "transfromer_decoder":
        #     feature_dim = output_head_cfg["FEATURE_DIM"]
        #     decoder_dim = output_head_cfg["DECODER_DIM"]
        #     decoder_depth = output_head_cfg["DECODER_DEPTH"]
            
        #     self.stabilizer = OutputHeadTransformer(
        #         feature_dim=feature_dim,
        #         decoder_dim=decoder_dim,
        #         decoder_depth=decoder_depth,
        #         num_feature_pos_enc=None,
        #         feature_mapping_mlp=False,
        #         queries="per_joint",
        #         joints_num=17,
        #     )

        self._initialize()
                    
        ### Class token
        # self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, 
        input_image=None,
        input_pose=None
        # input_depth=None
        ):

        batch_size = input_image.shape[0]

        x = input_image.permute(0, 2, 1, 3, 4)   # N T C H W -> N C T H W
        x = self.backbone(x)
        if x.dim() ==2 :
            img_feat = x
        else:
            img_feat = self.pooling(x).squeeze()     

        # Pose encoder / feat
        if self.use_pose:
            x = self.pose_proj(input_pose)
            x = self.pose_encoder(x)
            pose_feat = x[:, -1, :]  # Select the last time step
        else:
            pose_feat = torch.empty(batch_size, 0).to(input_image.device)

        ### fusion layer - imu + image + depth
        # TODO: attention based fusion
        fused_feat = torch.cat((img_feat, pose_feat), dim=1)

        # Output head
        output = self.stabilizer(fused_feat)
        
        return output
        
