from train import test_phase2
from data_loader import test_Phase2, copyright_collision_dateset    # 加载数据集
from utils import calculate_collision_nc, LogUtils
from torch.utils.data import DataLoader
import torch
from ConZWNetwork import ConZWNetwork_Stage1, ConZWNetwork_Stage2

version = 'baseline_model'
log_utils = LogUtils(log_file = f'log/{version}.log')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载自定义权重
state_dict = torch.load('models/weakStrong_convnext_small/phase1.pth', weights_only=True)
# 去掉 'module.' 前缀
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '')  
    new_state_dict[new_key] = v

model_phase1 = ConZWNetwork_Stage1()
model_phase1.load_state_dict(new_state_dict)
# 冻结参数
for param in model_phase1.robust_feature_extractor.parameters():
    param.requires_grad = False
model_phase1.eval()
model_phase2 = ConZWNetwork_Stage2(copyright_image_num_class = 200, robust_feature_extractor = model_phase1.robust_feature_extractor)

state_dict = torch.load('models/baseline_model/phase2.pth', weights_only=True)
# 去掉 'module.' 前缀
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '')  
    new_state_dict[new_key] = v

model_phase2.load_state_dict(new_state_dict)
model_phase2 = model_phase2.to(device)
model_phase2.eval()


# 数据路径和host_img路径如下
host_img_path = 'data/len_std.jpg'  # 替换为固定的host图像路径
copyright_dir = 'data/copyright'  # 替换为版权图像文件夹路径

# 创建数据集对象
dataset = copyright_collision_dateset(host_img_path=host_img_path, copyright_dir=copyright_dir )

zero_watermark_featruemap_arrays = []
total_correct = 0
total_samples = 0

with torch.no_grad():  # 在评估时不计算梯度
    for host_img, copyright_img, copyright_label in dataset:
        # 将数据移到GPU（如果有）
        host_img = host_img.unsqueeze(0).to(device, non_blocking=True)  # 增加batch维度并转移到GPU
        copyright_img = copyright_img.unsqueeze(0).to(device, non_blocking=True) # 增加batch维度并转移到GPU
        copyright_label = torch.tensor(copyright_label).to(device, non_blocking=True)

        # 前向传播
        zero_watermark_featruemap, pred_copyright = model_phase2(host_img, copyright_img)

        # 获取预测的标签
        copyright_predictor_class = torch.argmax(pred_copyright, dim=1)

        # 计算正确预测的数量
        copyright_correct_predictions = copyright_predictor_class == copyright_label
        total_correct += copyright_correct_predictions.sum().item()
        total_samples += 1 

        zero_watermark_featruemap_arrays.append(zero_watermark_featruemap.squeeze(0))
        

# 计算总体准确率
copyright_accuracy = total_correct / total_samples

# 计算总体的碰撞性
zero_watermark_featruemap_arrays = torch.stack(zero_watermark_featruemap_arrays).to(device)
copyright_collision_nc = calculate_collision_nc(zero_watermark_featruemap_arrays)

log_utils.info(f"copyright_accuracy: {copyright_accuracy}, copyright_collision_nc: {copyright_collision_nc}")



'''
    二、宿主图碰撞性测试
'''
# 数据集文件路径
hostImage_dir = './data/images'  # miniImageNet数据集
test_csv = './data/test.csv'  # 用于划分测试集
test_copyright_dir = 'data/nufe.jpg'


test_dataset_phase2 = test_Phase2(hostImage_dir, test_csv, test_copyright_dir)
test_loader_phase2 = DataLoader(test_dataset_phase2, batch_size=32, shuffle=True, num_workers=30, pin_memory = True)
test_phase2(model_phase2, test_loader_phase2, log_utils, device)