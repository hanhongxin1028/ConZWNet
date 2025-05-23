import torch
import os
from data_loader import Dataset_Phase2, test_Phase2    # 加载数据集
from train import train_phase2, valid_phase2
from torch.utils.data import DataLoader
from ConZWNetwork import ConZWNetwork_Stage1, ConZWNetwork_Stage2
from utils import LogUtils      # 导入记录日志工具

version = 'baseline_model'
log_utils = LogUtils(log_file = f'log/{version}.log')
model_dir = f"models/{version}"
os.makedirs(model_dir, exist_ok=True)

state_dict = torch.load('models/weakStrong_convnext_small/phase1.pth', weights_only=True)
# 去掉 'module.' 前缀
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '')  
    new_state_dict[new_key] = v

'''
    一、定义超参数 和 模型
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100        # 100轮迭代次数
batch_size = 256
learning_rate = 1e-4
model_phase1 = ConZWNetwork_Stage1()
model_phase1.load_state_dict(new_state_dict)
# 冻结参数
for param in model_phase1.robust_feature_extractor.parameters():
    param.requires_grad = False
model_phase1.eval()
model_phase2 = ConZWNetwork_Stage2(copyright_image_num_class = 200, robust_feature_extractor = model_phase1.robust_feature_extractor).to(device)
optimizer=torch.optim.Adamax(model_phase2.parameters(), lr=learning_rate)
scaler = torch.GradScaler(enabled=True)



'''
    二、定义数据集
'''
# 数据集文件路径
hostImage_dir = './data/images'  # miniImageNet数据集
train_csv = './data/train.csv'  # 用于划分训练集
val_csv = './data/val.csv'  # 用于划分验证集
test_csv = './data/test.csv'  # 用于划分测试集

copyright_dir = 'data/copyright'
test_copyright_dir = 'data/nufe.jpg'


train_dataset_phase2 = Dataset_Phase2(hostImage_dir, train_csv, copyright_dir)
val_dataset_phase2 = Dataset_Phase2(hostImage_dir, val_csv, copyright_dir)
test_dataset_phase2 = test_Phase2(hostImage_dir, test_csv, test_copyright_dir)

# 创建数据加载器
train_loader_phase2 = DataLoader(train_dataset_phase2, batch_size=batch_size, shuffle=True, num_workers=30, pin_memory = True)
val_loader_phase2 = DataLoader(val_dataset_phase2, batch_size=batch_size, shuffle=True, num_workers=30, pin_memory = True)
test_loader_phase2 = DataLoader(test_dataset_phase2, batch_size=32, shuffle=True, num_workers=30, pin_memory = True)



'''
    三、训练
'''
# 接着训练 zero-watermark 版权辨别任务
for epoch in range(1,epochs+1):
    model_phase2.train()
    train_phase2(model_phase2, train_loader_phase2, epoch, optimizer, log_utils, device, scaler)

    model_phase2.eval()
    avg_accuracy_watermark_classifier, avg_collision_nc = valid_phase2(model_phase2, val_loader_phase2, epoch, log_utils, device)

    torch.save(model_phase2.state_dict(), f"{model_dir}/phase2.pth")

    if(avg_collision_nc < 0.8 and avg_accuracy_watermark_classifier == 1 ):    # 能辨别copyright就停止训练
            break 