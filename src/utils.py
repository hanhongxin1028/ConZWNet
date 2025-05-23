import time
import io
import numpy as np
import random
import torch
import logging
import random
import numpy as np
from PIL import Image
import io


class Timer:   
    """
        名称：计时器工具
        作用：可评估一段代码的执行时间
        使用方法 : 
                1.先声明对象 timer = Timer()
                2.启动计时器 timer.start()
                3.停止计时器 timer.stop()
    """
    def __init__(self):
        """初始化函数"""
        self.times = []

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()
    

class LogUtils:
    """
        名称：日志记录工具
        作用：记录程序运行时产生的数据
        使用方法 : 
                1.先声明对象 log_utils = LogUtils()
                2.记录日志 log_utils.info(f"Epoch: {epoch}, Batch: {batch_id}, Loss: {avg_loss.item()}") 
    """
    def __init__(self, log_file='log/training.log', level=logging.INFO):
        """
        作用：初始化日志记录器
        Args : 
            log_file: 日志文件名
            level: 日志级别
        """
        # 配置日志
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),  # 输出到文件
                logging.StreamHandler()          # 同时输出到控制台
            ]
        )
        self.logger = logging.getLogger()  # 获取日志记录器

    def info(self, message):
        """记录信息级别的日志"""
        self.logger.info(message)

    def warning(self, message):
        """记录警告级别的日志"""
        self.logger.warning(message)

    def error(self, message):
        """记录错误级别的日志"""
        self.logger.error(message)

    def critical(self, message):
        """记录严重级别的日志"""
        self.logger.critical(message)

    def debug(self, message):
        """记录调试级别的日志"""
        self.logger.debug(message)

def sp_noise(image,prob):
    """
    对图片加 椒盐噪声
    Args：
        image ： 待处理的图片
        prob ： 椒盐噪声的概率，值越大，噪声越多，范围应在 0 到 1 之间
    """
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gasuss_noise(image, mean=0, var=0.005):
    """
    对图片加 高斯噪声
    Args：
        image ： 待处理的图片
        mean ： 均值
        var ： 方差
    """
    image = image.astype('float32') / 255.0
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out


def jpeg_compress(image, quality):
    """
    对输入图像进行 JPEG 压缩并返回 numpy 数组格式。
    Args:
        image: 输入的 numpy 数组图像（RGB）。
        quality: JPEG 压缩质量，值越低压缩越强（1-100）。
    return: 
        压缩后的图像，格式为 numpy 数组。
    """
    pil_image = Image.fromarray(image)  # 将 numpy 转换为 PIL 图像
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)  # 以指定质量保存为 JPEG
    buffer.seek(0)
    
    compressed_image = np.array(Image.open(buffer))  # 读取并转换为 numpy 格式  # 读取压缩后的图像  shape:(224, 224, 3)
    return compressed_image      # 返回 numpy 格式的图像


def calculate_robust_nc(a, b):
    """
    作用 : 适用于计算(batchsize,feature_dim)数组 , 计算a b之间的nc值
    公式 :  NC =  sum( (a - mean(a))  *  (b - mean(b)) ) / sqrt(sum((a - mean(a))^2) * sum((b - mean(b))^2))
    """
    # 计算均值，用于中心化
    a_mean = torch.mean(a, dim=1, keepdim=True)
    b_mean = torch.mean(b, dim=1, keepdim=True)

    # 中心化
    a_centered = a - a_mean
    b_centered = b - b_mean
    # 计算 NC
    numerator = torch.sum(a_centered * b_centered, dim=1)
    denominator = torch.sqrt(torch.sum(a_centered**2, dim=1) * torch.sum(b_centered**2, dim=1))

    # 避免分母为零
    if (denominator == 0).any():
        return 0.0
    
    nc_values = numerator / denominator

    # 返回所有样本的均值 NC 值
    nc_mean = torch.mean(nc_values)
    
    return nc_mean


def calculate_collision_nc(features):
    """
    作用 : 计算同一批次中每对特征向量之间的 NC 值（优化版）
    Args:
        - features: 特征矩阵 (batch_size, feature_dim)
    Returns:
        - nc_matrix: NC 值矩阵 (batch_size, batch_size)
    """

    batch_size, feature_dim = features.shape
    # 中心化
    features_mean = torch.mean(features, dim=1, keepdim=True)
    features_centered = features - features_mean

    # 计算分子：所有特征向量对的点积
    numerator = torch.mm(features_centered, features_centered.t())

    # 计算分母：每个特征向量的 L2 范数
    norm = torch.norm(features_centered, p=2, dim=1)
    denominator = torch.outer(norm, norm)  # 计算外积，得到分母矩阵

    # 避免分母为零
    denominator[denominator == 0] = 1e-8

    # 计算 NC 矩阵
    nc_matrix = numerator / denominator

    # 手动创建上三角部分的索引（不包括对角线）
    triu_indices = torch.triu_indices(batch_size, batch_size, 1)  # 1表示不包括对角线

    # 提取上三角的所有NC值
    upper_triangular_values = nc_matrix[triu_indices[0], triu_indices[1]]

    # 计算这些值的平均数
    collision_nc = upper_triangular_values.mean()


    return collision_nc


def NC(a, b):
    """
    计算两个向量或矩阵之间的 Normalized Correlation (NC)。
    
    Args:
        - a: np.ndarray , 向量或矩阵。
        - b: np.ndarray , 向量或矩阵 , 与 a 的形状相同。
    
    返回:
    - nc: float , a 和 b 之间的 NC 值。
    """
    # 检查形状一致性
    if a.shape != b.shape:
        raise ValueError("Inputs 'a' and 'b' must have the same shape.")
    
    # 中心化操作：减去均值
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    a_centered = a - a_mean
    b_centered = b - b_mean
    
    # 计算分子和分母
    numerator = np.sum(a_centered * b_centered)
    denominator = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
    
    # 避免分母为零
    if denominator == 0:
        return 0.0
    
    # 计算 NC
    return numerator / denominator


def arnold_transform(image, iterations):
    '''
        作用 : 对 image 图片进行 iterations 次Arnold置乱
        Args :
            1. image : 待置乱的图片 , 要求类型为numpy 且 size为(n,n)
            2. iterations : 置乱次数 , 要求为int型
        return : 
            image : 置乱后的图片 , 其类型为numpy , 其size为(n,n)
    '''
    N = image.shape[0]  # 这里假设 image 是 N×N 的二维矩阵 
    for _ in range(iterations):
        new_image = np.zeros_like(image)
        for i in range(N):
            for j in range(N):
                x = (i + j) % N
                y = (i + 2 * j) % N
                new_image[x, y] = image[i, j]
        image = new_image
    return image


def SimCLR_loss(strong_attacked_host_features, weak_attacked_host_features, temperature):
    '''
        作用: 计算两个通道之间的SimCLR_loss
        Args:
            - strong_attacked_host_features: 强通道
            - weak_attacked_host_features: 弱通道
            - temperature: 温度参数  较小的 temperature 会放大相似度的差异，使得正样本对的权重更大，负样本对的权重更小; 会导致梯度较大，模型更新更快，但可能会引入不稳定性
                                    较大的 temperature 会缩小相似度的差异，使得正样本对和负样本对的权重更加均衡; 较大的 temperature 会导致梯度较小，模型更新更慢，但训练过程更稳定
    '''
    # 获取batch_size
    batch_size = strong_attacked_host_features.shape[0]

    # 1. 归一化两通道的特征
    strong_attacked_normalize = torch.nn.functional.normalize(strong_attacked_host_features, dim=-1)
    weak_attacked_normalize = torch.nn.functional.normalize(weak_attacked_host_features, dim=-1)

    # 2. 按行拼接
    all_features = torch.cat([strong_attacked_normalize, weak_attacked_normalize], dim=0)   # [2*batch_size, dims]

    # 3. 计算相似度矩阵
    sim_matrix = torch.exp(torch.mm(all_features, all_features.t().contiguous()) / temperature)     # [2*batch_size, 2*batch_size]
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)    # 去除样本与自身的相似度 [2*batch_size, 2*batch_size-1]

    # 4. compute loss
    pos_sim = torch.exp(torch.sum(strong_attacked_normalize * weak_attacked_normalize, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # [2*batch_size]
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss