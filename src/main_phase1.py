import torch
import os
from data_loader import Dataset_Phase1, test_Phase1    # 加载数据集
from train import train_phase1, valid_phase1, test_phase1
from torch.utils.data import DataLoader
from ConZWNetwork import ConZWNetwork_Stage1, Loss_SimCLR
from utils import LogUtils      # 导入记录日志工具

version = 'weakStrong_convnext_small'    # 版本B : 强弱数据增强 + ResNet34
log_utils = LogUtils(log_file = f'log/{version}.log')
model_dir = f"models/{version}"
os.makedirs(model_dir, exist_ok=True)

'''
    一、定义超参数 和 模型
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 200        # 100轮迭代次数
batch_size = 256
learning_rate = 1e-3
temperature = 0.5
model_phase1 = torch.nn.DataParallel(ConZWNetwork_Stage1()).to(device)
SimCLR_loss = Loss_SimCLR().to(device)
optimizer=torch.optim.Adam(model_phase1.parameters(), lr=learning_rate, weight_decay=1e-6)
scaler = torch.GradScaler(enabled=True)


'''
    二、定义数据集
'''
# 数据集文件路径
hostImage_dir = './data/images'  # miniImageNet数据集
train_csv = './data/train.csv'  # 用于划分训练集
val_csv = './data/val.csv'  # 用于划分验证集
test_csv = './data/test.csv'  # 用于划分测试集

# 数据增强 
train_dataset_phase1 = Dataset_Phase1(hostImage_dir, train_csv)
val_dataset_phase1 = Dataset_Phase1(hostImage_dir, val_csv)
test_dataset_phase1 = test_Phase1(hostImage_dir, test_csv)

# 创建数据加载器
train_loader_phase1 = DataLoader(train_dataset_phase1, batch_size=batch_size, shuffle=True, num_workers=30, pin_memory = True)
val_loader_phase1 = DataLoader(val_dataset_phase1, batch_size=batch_size, shuffle=True, num_workers=30, pin_memory = True)
test_loader_phase1 = DataLoader(test_dataset_phase1, batch_size=32, shuffle=True, num_workers=30, pin_memory = True)



'''
    三、训练
'''
# 100轮训练 鲁棒特征提取器
for epoch in range(1,epochs+1):
    model_phase1.train()
    train_phase1(model_phase1, train_loader_phase1, epoch, optimizer, log_utils, device, scaler, SimCLR_loss, temperature)

    if epoch % 10 == 0:
        model_phase1.eval() 
        avg_robust_nc, avg_collision_nc = valid_phase1(model_phase1, val_loader_phase1, epoch, log_utils, device, SimCLR_loss, temperature)
        torch.save(model_phase1.state_dict(), f"{model_dir}/phase1.pth")


# 测试
model_phase1.eval()
test_phase1(model_phase1, test_loader_phase1, log_utils, device)
    