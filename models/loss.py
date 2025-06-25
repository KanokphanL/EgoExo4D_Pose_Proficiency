import torch
import torch.nn as nn
import torch.nn.functional as F

def build_criterion(config):
    loss_type = config["TRAIN"]["LOSS_CRITERION"]
    weight = config.TRAIN.get("CLASS_WEIGHTS", [1.0, 1.0, 1.0, 1.0])
    if loss_type == 'CrossEntropyLoss':
        weight_tensor = torch.tensor(weight, dtype=torch.float)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    elif loss_type == 'FocalLoss':
        criterion = FocalLoss()

    else:
        raise NotImplementedError(f"{loss_type} is not implemented")

    return criterion
    

class LabelSmoothingCrossEntropy(nn.Module):
    """
    带标签平滑的交叉熵损失函数
    
    参数:
    eps (float): 平滑参数，控制标签平滑的程度，通常取0.0-0.2之间
    reduction (str): 损失聚合方式，可选 'mean', 'sum', 'none'
    ignore_index (int): 忽略的标签索引，用于屏蔽某些样本
    """
    def __init__(self, eps: float = 0.1, reduction: str = 'mean', ignore_index: int = -100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算带标签平滑的交叉熵损失
        
        参数:
        inputs (torch.Tensor): 模型输出的logits，形状 [batch_size, num_classes]
        targets (torch.Tensor): 真实标签，形状 [batch_size]
        
        返回:
        torch.Tensor: 计算得到的损失
        """
        # 确保输入维度正确
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # [N,C,H,W] -> [N,C,H*W]
            inputs = inputs.transpose(1, 2)    # [N,C,H*W] -> [N,H*W,C]
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # [N,H*W,C] -> [N*H*W,C]
        
        # 获取类别数
        num_classes = inputs.size(-1)
        
        # 创建平滑后的标签分布
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # 屏蔽ignore_index的样本
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            targets = targets.masked_fill(mask == 0, 0)
            log_probs = log_probs.masked_fill(mask.unsqueeze(1) == 0, 0.0)
        
        # 计算平滑后的标签分布
        # 真实类别概率: 1 - eps + eps/num_classes
        # 其他类别概率: eps/num_classes
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.eps / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.eps)
        
        # 计算损失
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        
        # 聚合损失
        if self.reduction == 'mean':
            if self.ignore_index >= 0:
                return loss.sum() / mask.sum()
            else:
                return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class FocalLoss(nn.Module):
    """
    Focal Loss - 解决类别不平衡问题的损失函数
    论文: https://arxiv.org/abs/1708.02002
    
    参数:
    alpha (float or list): 类别权重，用于平衡正负样本或类别频率
    gamma (float): 聚焦参数，调节易分类样本的权重
    reduction (str): 损失聚合方式，可选 'mean', 'sum', 'none'
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        
        参数:
        inputs (torch.Tensor): 模型输出的logits，形状 [batch_size, num_classes]
        targets (torch.Tensor): 真实标签，形状 [batch_size]
        
        返回:
        torch.Tensor: 计算得到的Focal Loss
        """
        # 确保输入维度正确
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # [N,C,H,W] -> [N,C,H*W]
            inputs = inputs.transpose(1, 2)    # [N,C,H*W] -> [N,H*W,C]
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # [N,H*W,C] -> [N*H*W,C]
        
        # 计算交叉熵损失
        logpt = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-logpt)
        
        # 计算alpha权重
        if isinstance(self.alpha, float):
            # 二分类情况，alpha为正样本权重
            at = torch.full_like(logpt, self.alpha)
            at[targets == 0] = 1 - self.alpha
        else:
            # 多分类情况，alpha为每个类别的权重列表
            at = self.alpha.gather(0, targets)
        
        # 计算focal loss
        loss = -at * ((1 - pt) ** self.gamma) * logpt
        
        # 聚合损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    带标签平滑的交叉熵损失函数
    
    参数:
    eps (float): 平滑参数，控制标签平滑的程度，通常取0.0-0.2之间
    reduction (str): 损失聚合方式，可选 'mean', 'sum', 'none'
    ignore_index (int): 忽略的标签索引，用于屏蔽某些样本
    """
    def __init__(self, eps: float = 0.1, reduction: str = 'mean', ignore_index: int = -100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算带标签平滑的交叉熵损失
        
        参数:
        inputs (torch.Tensor): 模型输出的logits，形状 [batch_size, num_classes]
        targets (torch.Tensor): 真实标签，形状 [batch_size]
        
        返回:
        torch.Tensor: 计算得到的损失
        """
        # 确保输入维度正确
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # [N,C,H,W] -> [N,C,H*W]
            inputs = inputs.transpose(1, 2)    # [N,C,H*W] -> [N,H*W,C]
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # [N,H*W,C] -> [N*H*W,C]
        
        # 获取类别数
        num_classes = inputs.size(-1)
        
        # 创建平滑后的标签分布
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # 屏蔽ignore_index的样本
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            targets = targets.masked_fill(mask == 0, 0)
            log_probs = log_probs.masked_fill(mask.unsqueeze(1) == 0, 0.0)
        
        # 计算平滑后的标签分布
        # 真实类别概率: 1 - eps + eps/num_classes
        # 其他类别概率: eps/num_classes
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.eps / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.eps)
        
        # 计算损失
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        
        # 聚合损失
        if self.reduction == 'mean':
            if self.ignore_index >= 0:
                return loss.sum() / mask.sum()
            else:
                return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class LossAnnotation2D(nn.Module):
    def __init__(self):
        super(LossAnnotation2D, self).__init__()

    def forward(self, pred, target, mask):
        """
        pred: (batch_size, 17, 2)
        target: (batch_size, 17, 2)
        mask: (batch_size, 17, 1)
        """
        loss = torch.nn.functional.mse_loss(pred * mask, target * mask, reduction='sum')
        
        loss = loss / mask.sum()
        
        return loss
    
class LossAnnotation3D(nn.Module):
    def __init__(self):
        super(LossAnnotation3D, self).__init__()

    def forward(self, pred, target, mask):
        """
        pred: (batch_size, 17, 3)
        target: (batch_size, 17, 3)
        mask: (batch_size, 17, 1)
        """
        loss = torch.nn.functional.mse_loss(pred * mask, target * mask, reduction='sum')
        
        loss = loss / mask.sum()
        
        return loss
    
class MPJPELoss(nn.Module):
    def __init__(self):
        super(MPJPELoss, self).__init__()

    def forward(self, pred, target):
        """
        pred: (batch_size, num_joints, 3)
        target: (batch_size, num_joints, 3)
        """
        # 检查 target 中的 NaN 值，并创建掩码
        valid_mask = ~torch.isnan(target).any(dim=-1)  # 如果一个关节的任何一个坐标是 NaN，则该关节无效
        target[~valid_mask] = 0
        # 计算每个关节的欧几里得距离
        distance = torch.norm(pred - target, dim=-1) * valid_mask

        # 计算平均值
        mpjpe = torch.sum(distance) / (valid_mask.sum() + 1e-8)  # 添加一个小的常数避免除以零
        mpjpe = mpjpe * 100 # 将距离转换为厘米
        return mpjpe

# 使用示例
if __name__ == "__main__":
    # 初始化Focal Loss
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # 模拟模型输出和标签
    logits = torch.randn(32, 10)  # 32个样本，10个类别
    targets = torch.randint(0, 10, (32,))  # 随机标签
    
    # 计算损失
    loss = criterion(logits, targets)
    print(f"Focal Loss: {loss.item()}")