import torch.nn as nn   # 导入 torch 的神经网络模块
from torchvision.models import resnet18, resnet34, resnet50, convnext_base, convnext_tiny, convnext_small, convnext_large          # 导入 resnet50 和 ResNet18 模型
import torch.nn.functional as F
import torch

# stage one ,unsupervised learning
class ConZWNetwork_Stage1(nn.Module):
    def __init__(self, feature_dim=128):
        super(ConZWNetwork_Stage1, self).__init__()

        # encoder
        small = convnext_small(weights=None)

        # 去掉最后的分类头（全连接层），保留全局池化
        self.robust_feature_extractor = nn.Sequential(*list(small.children())[:-1])

        # projection head
        self.projection_head = nn.Sequential(
                                                nn.Linear(768, 256, bias=False),
                                                nn.BatchNorm1d(256),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(256, feature_dim, bias=True)
                                            )

    def forward(self, x):
        feature = self.robust_feature_extractor(x)  # [batchsize, 768, 1, 1]
        feature = torch.flatten(feature, start_dim = 1) # [batchsize, 768]
        out = self.projection_head(feature)
        return feature, F.normalize(out, dim=-1)


# stage two ,supervised learning
class ConZWNetwork_Stage2(torch.nn.Module):
    def __init__(self, 
                 copyright_image_num_class, 
                 robust_feature_extractor,
                 host_image_feature_dims = 768,
                 copyright_image_feature_dims = 768,
                 zeroWatermark_feature_dims = 1536
                 ):
        super(ConZWNetwork_Stage2, self).__init__()
        # encoder
        self.robust_feature_extractor = robust_feature_extractor

        # fusion
        self.zeroWatermark_feature_fusion = nn.Sequential(
                                                            nn.Linear(host_image_feature_dims+copyright_image_feature_dims, 1024),
                                                            nn.ReLU(),
                                                            nn.Linear(1024, zeroWatermark_feature_dims)
                                                        )
        
        self.shortcut = nn.Identity()
        
        # 版权图分类
        self.copyright_discrimination = nn.Linear(zeroWatermark_feature_dims, copyright_image_num_class)

        

    def forward(self, host_image, copyright_image):
        # 利用 鲁棒特征提取器 提取 宿主图特征
        host_feature = torch.flatten(self.robust_feature_extractor(host_image), start_dim = 1)

        # 利用 鲁棒特征提取器 提取 版权图特征
        copyright_feature = torch.flatten(self.robust_feature_extractor(copyright_image), start_dim = 1)
        
        # 特征拼接
        cat_featrues = torch.cat((host_feature, copyright_feature), dim=1)

        # 融合 拼接 特征
        zero_watermark_featruemap = 0.1 * self.zeroWatermark_feature_fusion(cat_featrues) + 0.9 * self.shortcut(cat_featrues)

        # 应用分类任务
        pred_copyright = self.copyright_discrimination(zero_watermark_featruemap)

        return  zero_watermark_featruemap, pred_copyright
    

class Loss_SimCLR(torch.nn.Module):
    def __init__(self):
        super(Loss_SimCLR,self).__init__()

    def forward(self, out_1, out_2, temperature):
        # 分母 ：X.X.T，再去掉对角线值，分析结果一行，可以看成它与除了这行外的其他行都进行了点积运算（包括out_1和out_2）,
        # 而每一行为一个batch的一个取值，即一个输入图像的特征表示，
        # 因此，X.X.T，再去掉对角线值表示，每个输入图像的特征与其所有输出特征（包括out_1和out_2）的点积，用点积来衡量相似性
        # 加上exp操作，该操作实际计算了分母
        batch_size = out_1.shape[0]
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # 分子： *为对应位置相乘，也是点积
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        
        return (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()