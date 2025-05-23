import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from data_augment import cornerCrop, salt_pepper_noise, gaussian_noise, median_filtering, gaussian_filtering, jpeg_compress
import random


'''
    作用 : 创建数据集，对  “弱+强”  与 水印 拼接、对 “弱” 与 水印 拼接
    Args : 
        strong_dir : 弱+强 图片 所在的文件路径
        weak_dir : 弱 图片 所在的文件路径
        watermark_dir : 水印 图片 所在的文件路径
    返回值 : weak_img, strong_img,  watermark_img, ground_truth_copyright_label
'''

class Dataset_Phase1(Dataset):
    def __init__(self,  host_dir , csv_file):
        """
        Args:
            strong_dir (str): strong images 的目录路径
            weak_dir (str): weak images 的目录路径
        """
        self.data = pd.read_csv(csv_file, header=0, names=['filename', 'label'])     # header=0
        self.hostimage_dir = host_dir
        
        # 强通道
        self.strong_images_transform = transforms.Compose([
                    transforms.RandomChoice([
                        transforms.RandomResizedCrop(size=(224, 224), scale=(0.05, 0.8)),   # 裁剪
                        transforms.RandomRotation(degrees=random.choice([(120,120), (150,150), (180,180)])),  # 旋转
                        transforms.Lambda(lambda img: cornerCrop(img, 
                                                                 position=random.choice(['left_top', 'right_top', 'left_bottom', 'right_bottom']), # 随机选择位置
                                                                 blackout_size=random.uniform(1/3, 2/3))    # 随机选择大小
                                                                 ),    # 四角黑块
                        transforms.Lambda(lambda img : salt_pepper_noise(img, prob = 0.5)),     # 椒盐噪声
                        transforms.Lambda(lambda img : gaussian_noise(img, var = 0.5)),     # 高斯噪声
                        transforms.Lambda(lambda img : median_filtering(img, ksize= (13,13))),     # 均值滤波
                        transforms.Lambda(lambda img : gaussian_filtering(img, ksize= (13,13))),     # 高斯滤波
                        transforms.Lambda(lambda img : jpeg_compress(img, quality = 20 )),     # JEPG压缩
                    ])
        ])

        # 弱通道
        self.weak_images_transform = transforms.Compose([
                    transforms.RandomChoice([
                        transforms.RandomResizedCrop(size=(224, 224), scale=(0.75, 0.9)),   # 裁剪
                        transforms.RandomRotation(degrees=random.choice([(10,10), (30,30), (60,60)])),  # 旋转
                        transforms.Lambda(lambda img: cornerCrop(img, 
                                                                 position=random.choice(['left_top', 'right_top', 'left_bottom', 'right_bottom']), # 随机选择位置
                                                                 blackout_size=random.uniform(1/10, 1/6))    # 随机选择大小
                                                                 ),    # 四角黑块
                        transforms.Lambda(lambda img : salt_pepper_noise(img, prob = 0.01)),     # 椒盐噪声
                        transforms.Lambda(lambda img : gaussian_noise(img, var = 0.01)),     # 高斯噪声
                        transforms.Lambda(lambda img : median_filtering(img, ksize= (3,3))),     # 均值滤波
                        transforms.Lambda(lambda img : gaussian_filtering(img, ksize= (3,3))),     # 高斯滤波
                        transforms.Lambda(lambda img : jpeg_compress(img, quality = 80 )),     # JEPG压缩
                    ])
        ])


        self.hostImage_normal_standard = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 转换为Tensor，归一化到[0, 1]范围
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
        ])

        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 打开文件
        img_name = os.path.join(self.hostimage_dir, self.data.iloc[idx, 0])  # 获取图像路径
        host_img = Image.open(img_name).convert('RGB')  # 打开图像文件

        # 数据增强
        host_img_copy = host_img.copy() 
        weak_img = self.weak_images_transform(host_img_copy)    # 弱攻击
        weak_img_copy = weak_img.copy()
        strong_img = self.strong_images_transform(weak_img_copy)  # 强攻击

        # 转换为 PyTorch 张量(归一化和标准化)
        host_img = self.hostImage_normal_standard(host_img)
        weak_img = self.hostImage_normal_standard(weak_img)
        strong_img = self.hostImage_normal_standard(strong_img)

        return host_img, weak_img, strong_img 
    

class test_Phase1(Dataset):
    def __init__(self,  host_dir, csv_file):
        self.data = pd.read_csv(csv_file, header=0, names=['filename', 'label'])     # header=0
        self.hostimage_dir = host_dir

        self.normal_standard = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 转换为Tensor，归一化到[0, 1]范围
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
        ])


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 打开文件
        img_name = os.path.join(self.hostimage_dir, self.data.iloc[idx, 0])  # 获取图像路径
        original_img = Image.open(img_name).convert('RGB')  # 打开图像文件
        

        # 数据增强
        original_img_copy = original_img.copy()

        # 旋转角度
        rotation_10 = self.normal_standard(transforms.RandomRotation(degrees=(10,10))(original_img_copy))
        rotation_30 = self.normal_standard(transforms.RandomRotation(degrees=(30,30))(original_img_copy))
        rotation_60 = self.normal_standard(transforms.RandomRotation(degrees=(60,60))(original_img_copy))
        rotation_120 = self.normal_standard(transforms.RandomRotation(degrees=(120,120))(original_img_copy))
        rotation_150 = self.normal_standard(transforms.RandomRotation(degrees=(150,150))(original_img_copy))
        rotation_180 = self.normal_standard(transforms.RandomRotation(degrees=(180,180))(original_img_copy))

        # 椒盐噪声
        sp_noise_1 = self.normal_standard(salt_pepper_noise(original_img_copy, 0.01))
        sp_noise_2 = self.normal_standard(salt_pepper_noise(original_img_copy, 0.05))
        sp_noise_3 = self.normal_standard(salt_pepper_noise(original_img_copy, 0.1))

        # 高斯噪声
        gaussian_noise_1 = self.normal_standard(gaussian_noise(original_img_copy, 0.01))
        gaussian_noise_2 = self.normal_standard(gaussian_noise(original_img_copy, 0.05))
        gaussian_noise_3 = self.normal_standard(gaussian_noise(original_img_copy, 0.1))

        # 随机裁剪
        resize_1 = self.normal_standard(transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 0.8))(original_img_copy))
        resize_2 = self.normal_standard(transforms.RandomResizedCrop(size=(224, 224), scale=(0.55, 0.55))(original_img_copy))
        resize_3 = self.normal_standard(transforms.RandomResizedCrop(size=(224, 224), scale=(0.4, 0.4))(original_img_copy))
        resize_4 = self.normal_standard(transforms.RandomResizedCrop(size=(224, 224), scale=(0.25, 0.25))(original_img_copy))

        # 四角裁剪
        cornerCrop_left_top = self.normal_standard(cornerCrop(original_img_copy, 'left_top', 0.125))
        cornerCrop_right_top = self.normal_standard(cornerCrop(original_img_copy, 'right_top', 0.125))
        cornerCrop_left_bottom = self.normal_standard(cornerCrop(original_img_copy, 'left_bottom', 0.125))
        cornerCrop_right_bottom = self.normal_standard(cornerCrop(original_img_copy, 'right_bottom', 0.125))

        # 高斯滤波
        gaussian_filter_3 = self.normal_standard(gaussian_filtering(original_img_copy, (3, 3)))
        gaussian_filter_5 = self.normal_standard(gaussian_filtering(original_img_copy, (5, 5)))
        gaussian_filter_9 = self.normal_standard(gaussian_filtering(original_img_copy, (9, 9)))
        gaussian_filter_11 = self.normal_standard(gaussian_filtering(original_img_copy, (11, 11)))
        
        # 均值滤波
        median_filter_3 = self.normal_standard(median_filtering(original_img_copy, (3, 3)))
        median_filter_5 = self.normal_standard(median_filtering(original_img_copy, (5, 5)))
        median_filter_9 = self.normal_standard(median_filtering(original_img_copy, (9, 9)))
        median_filter_11 =self.normal_standard( median_filtering(original_img_copy, (11, 11)))

        # JEPG压缩
        jpeg_compress_20 = self.normal_standard(jpeg_compress(original_img_copy, 20))
        jpeg_compress_40 = self.normal_standard(jpeg_compress(original_img_copy, 40))
        jpeg_compress_60 = self.normal_standard(jpeg_compress(original_img_copy, 60))
        jpeg_compress_80 = self.normal_standard(jpeg_compress(original_img_copy, 80))



        # 标准化归一化
        original_img = self.normal_standard(original_img)                                                                                                                                                                                                                           

        return original_img, rotation_10, rotation_30, rotation_60, rotation_120, rotation_150, rotation_180, sp_noise_1, sp_noise_2, sp_noise_3, gaussian_noise_1, gaussian_noise_2, gaussian_noise_3, resize_1, resize_2, resize_3, resize_4, cornerCrop_left_top, cornerCrop_right_top, cornerCrop_left_bottom, cornerCrop_right_bottom, gaussian_filter_3, gaussian_filter_5,  gaussian_filter_9, gaussian_filter_11, median_filter_3, median_filter_5, median_filter_9, median_filter_11, jpeg_compress_20, jpeg_compress_40, jpeg_compress_60, jpeg_compress_80
 



    
class Dataset_Phase2(Dataset):
    def __init__(self,  host_dir, csv_file, copyright_dir ):
        """
        Args:
            strong_dir (str): strong images 的目录路径
            weak_dir (str): weak images 的目录路径
            watermark_dir (str): 水印图像的目录路径
            transform (callable, optional): 图像变换。
        """
        self.data = pd.read_csv(csv_file, header=0, names=['filename', 'label'])     # header=0
        self.hostimage_dir = host_dir
        self.copyright_images = [os.path.join(copyright_dir, img) for img in os.listdir(copyright_dir) if img.endswith('.jpg')]
        self.num_copyrights = len(self.copyright_images)

        self.hostImage_normal_standard = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 转换为Tensor，归一化到[0, 1]范围
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
        ])

        self.copyright_normal_standard = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 转换为Tensor，归一化到[0, 1]范围
            transforms.Normalize([0.747573, 0.773855, 0.794295], [0.354062, 0.311262, 0.291780])  # 标准化
        ])


        # 预加载所有 watermark 图像
        self.copyrights = []
        for i in range(1, self.num_copyrights + 1):
            copyright_path = os.path.join(copyright_dir, f"{i}.jpg")
            copyright = Image.open(copyright_path).convert('RGB')  # 确保是 RGB 格式
            copyright = self.copyright_normal_standard(copyright)
            self.copyrights.append(copyright)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 打开文件
        img_name = os.path.join(self.hostimage_dir, self.data.iloc[idx, 0])  # 获取图像路径
        host_img = Image.open(img_name).convert('RGB')  # 打开图像文件

        # 转换为 PyTorch 张量(归一化和标准化)
        host_img = self.hostImage_normal_standard(host_img)

        watermark_index = random.randint(0, self.num_copyrights-1)
        watermark_img = self.copyrights[watermark_index]  # 水印图像

        return host_img, watermark_img, watermark_index 
    

class test_Phase2(Dataset):
    def __init__(self,  host_dir, csv_file, copyright_dir):
        self.data = pd.read_csv(csv_file, header=0, names=['filename', 'label'])     # header=0
        self.hostimage_dir = host_dir

        self.normal_standard = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 转换为Tensor，归一化到[0, 1]范围
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
        ])

        self.copyright_normal_standard = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 转换为Tensor，归一化到[0, 1]范围
            transforms.Normalize([0.747573, 0.773855, 0.794295], [0.354062, 0.311262, 0.291780])  # 标准化
        ])

        # 固定一张 copyright
        copyright = Image.open(copyright_dir).convert('RGB')
        self.copyright_img = self.copyright_normal_standard(copyright)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 打开文件
        img_name = os.path.join(self.hostimage_dir, self.data.iloc[idx, 0])  # 获取图像路径
        original_img = Image.open(img_name).convert('RGB')  # 打开图像文件
        copyright_img = self.copyright_img  # 获取同一张copyright_img

        # 数据增强
        original_img_copy = original_img.copy()

        # 旋转角度
        rotation_10 = self.normal_standard(transforms.RandomRotation(degrees=(10,10))(original_img_copy))
        rotation_30 = self.normal_standard(transforms.RandomRotation(degrees=(30,30))(original_img_copy))
        rotation_60 = self.normal_standard(transforms.RandomRotation(degrees=(60,60))(original_img_copy))
        rotation_120 = self.normal_standard(transforms.RandomRotation(degrees=(120,120))(original_img_copy))
        rotation_150 = self.normal_standard(transforms.RandomRotation(degrees=(150,150))(original_img_copy))
        rotation_180 = self.normal_standard(transforms.RandomRotation(degrees=(180,180))(original_img_copy))

        # 椒盐噪声
        sp_noise_1 = self.normal_standard(salt_pepper_noise(original_img_copy, 0.01))
        sp_noise_2 = self.normal_standard(salt_pepper_noise(original_img_copy, 0.05))
        sp_noise_3 = self.normal_standard(salt_pepper_noise(original_img_copy, 0.1))

        # 高斯噪声
        gaussian_noise_1 = self.normal_standard(gaussian_noise(original_img_copy, 0.01))
        gaussian_noise_2 = self.normal_standard(gaussian_noise(original_img_copy, 0.05))
        gaussian_noise_3 = self.normal_standard(gaussian_noise(original_img_copy, 0.1))

        # 随机裁剪
        resize_1 = self.normal_standard(transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 0.8))(original_img_copy))
        resize_2 = self.normal_standard(transforms.RandomResizedCrop(size=(224, 224), scale=(0.55, 0.55))(original_img_copy))
        resize_3 = self.normal_standard(transforms.RandomResizedCrop(size=(224, 224), scale=(0.4, 0.4))(original_img_copy))
        resize_4 = self.normal_standard(transforms.RandomResizedCrop(size=(224, 224), scale=(0.25, 0.25))(original_img_copy))

        # 四角裁剪
        cornerCrop_left_top = self.normal_standard(cornerCrop(original_img_copy, 'left_top', 0.125))
        cornerCrop_right_top = self.normal_standard(cornerCrop(original_img_copy, 'right_top', 0.125))
        cornerCrop_left_bottom = self.normal_standard(cornerCrop(original_img_copy, 'left_bottom', 0.125))
        cornerCrop_right_bottom = self.normal_standard(cornerCrop(original_img_copy, 'right_bottom', 0.125))

        # 高斯滤波
        gaussian_filter_3 = self.normal_standard(gaussian_filtering(original_img_copy, (3, 3)))
        gaussian_filter_5 = self.normal_standard(gaussian_filtering(original_img_copy, (5, 5)))
        gaussian_filter_9 = self.normal_standard(gaussian_filtering(original_img_copy, (9, 9)))
        gaussian_filter_11 = self.normal_standard(gaussian_filtering(original_img_copy, (11, 11)))
        
        # 均值滤波
        median_filter_3 = self.normal_standard(median_filtering(original_img_copy, (3, 3)))
        median_filter_5 = self.normal_standard(median_filtering(original_img_copy, (5, 5)))
        median_filter_9 = self.normal_standard(median_filtering(original_img_copy, (9, 9)))
        median_filter_11 =self.normal_standard( median_filtering(original_img_copy, (11, 11)))

        # JEPG压缩
        jpeg_compress_20 = self.normal_standard(jpeg_compress(original_img_copy, 20))
        jpeg_compress_40 = self.normal_standard(jpeg_compress(original_img_copy, 40))
        jpeg_compress_60 = self.normal_standard(jpeg_compress(original_img_copy, 60))
        jpeg_compress_80 = self.normal_standard(jpeg_compress(original_img_copy, 80))



        # 标准化归一化
        original_img = self.normal_standard(original_img)                                                                                                                                                                                                                           

        return original_img, rotation_10, rotation_30, rotation_60, rotation_120, rotation_150, rotation_180, sp_noise_1, sp_noise_2, sp_noise_3, gaussian_noise_1, gaussian_noise_2, gaussian_noise_3, resize_1, resize_2, resize_3, resize_4, cornerCrop_left_top, cornerCrop_right_top, cornerCrop_left_bottom, cornerCrop_right_bottom, gaussian_filter_3, gaussian_filter_5,  gaussian_filter_9, gaussian_filter_11, median_filter_3, median_filter_5, median_filter_9, median_filter_11, jpeg_compress_20, jpeg_compress_40, jpeg_compress_60, jpeg_compress_80, copyright_img


class copyright_collision_dateset(Dataset):
    def __init__(self, host_img_path, copyright_dir):
        """
        初始化数据集类
        :param copyright_dir: 版权图所在文件夹路径
        :param host_img_path: 固定的host图像路径
        """
        self.host_img_path = host_img_path  # 存储host图像路径
        self.copyright_dir = copyright_dir
        self.copyright_names = sorted(os.listdir(copyright_dir))  # 获取文件夹中的所有图片文件，排序
        

        # 定义图像转换
        self.hostImage_normal_standard = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 转换为Tensor，归一化到[0, 1]范围
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
        ])

        self.copyright_normal_standard = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 转换为Tensor，归一化到[0, 1]范围
            transforms.Normalize([0.747573, 0.773855, 0.794295], [0.354062, 0.311262, 0.291780])  # 标准化
        ])

        # 加载host图像并预处理
        self.host_img = Image.open(self.host_img_path).convert("RGB")
        self.host_img = self.hostImage_normal_standard(self.host_img)

        # 一次性加载所有版权图像并预处理
        self.all_data = []
        for copyright_name in self.copyright_names:
            copyright_path = os.path.join(self.copyright_dir, copyright_name)
            copyright_img = Image.open(copyright_path).convert("RGB")
            copyright_img = self.copyright_normal_standard(copyright_img)
            copyright_label = int(copyright_name.split('.')[0]) - 1  # 提取标签
            self.all_data.append((self.host_img, copyright_img, copyright_label))

    def __len__(self):
        """返回数据集的大小"""
        return len(self.all_data)

    def __getitem__(self, idx):
        """返回一个样本（图片和标签）"""
        return self.all_data[idx]