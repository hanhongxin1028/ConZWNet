from PIL import Image
import numpy as np
import random
import io
import cv2
from torchvision import transforms

"""     四角裁剪        """
def cornerCrop(img, position, blackout_size):
    """
    在图像的指定位置添加黑色块。
    position: 'left_top', 'right_top', 'left_bottom', 'right_bottom'
    blackout_size: 黑块的大小比例(0 到 1 之间)
    """
    img = np.array(img)
    height, width, _ = img.shape  # 获取图像的高度和宽度

    # 计算黑块的尺寸
    blackout_width = int(blackout_size * width)
    blackout_height = int(blackout_size * height)
    
    # 确定黑块位置
    if position == 'left_bottom':
        img[height - blackout_height:height, 0:blackout_width] = 0
    elif position == 'right_bottom':
        img[height - blackout_height:height, width - blackout_width:width] = 0
    elif position == 'left_top':
        img[0:blackout_height, 0:blackout_width] = 0
    elif position == 'right_top':
        img[0:blackout_height, width - blackout_width:width] = 0

    img = Image.fromarray(img.astype(np.uint8))
    return img

""""    椒盐噪声        """
def salt_pepper_noise(img, prob):
    """
    添加椒盐噪声。
    prob: 噪声的概率，决定多少像素被修改为黑色或白色。
    """
    # 将PIL Image转换为NumPy数组
    img = np.array(img)
    
    # 生成噪声的掩码
    total_pixels = img.size
    num_salt = int(total_pixels * prob / 2)  # 计算白色噪声的数量
    num_pepper = int(total_pixels * prob / 2)  # 计算黑色噪声的数量

    # 添加盐（白色噪声）
    salt_coords = [(random.randint(0, img.shape[0]-1), random.randint(0, img.shape[1]-1)) for _ in range(num_salt)]  # 生成坐标
    for coord in salt_coords:
        img[coord] = 255  # 设置为白色

    # 添加胡椒（黑色噪声）
    pepper_coords = [(random.randint(0, img.shape[0]-1), random.randint(0, img.shape[1]-1)) for _ in range(num_pepper)]  # 生成坐标
    for coord in pepper_coords:
        img[coord] = 0  # 设置为黑色

    # 将NumPy数组转换回PIL Image
    img = Image.fromarray(img)
    
    return img


"""     高斯噪声    """
def gaussian_noise(img, var):
    """
    添加高斯噪声。
    var: 噪声的方差，控制噪声的强度。
    """
    # 将PIL Image转换为NumPy数组
    img = np.array(img)
    
    # 获取图像的形状
    mean = 0
    row, col, ch = img.shape
    
    # 生成符合高斯分布的噪声
    gauss = np.random.normal(mean, var ** 0.5, (row, col, ch))  # 生成正态分布噪声
    
    # 将噪声加到原图像上
    noisy = img + gauss  # 加上噪声
    noisy = np.clip(noisy, 0, 255)  # 保证图像值在[0, 255]之间
    
    # 将NumPy数组转换回PIL Image
    noisy = Image.fromarray(np.uint8(noisy))
    
    return noisy

"""     均值滤波    """
def median_filtering(img, ksize):
    """
    均值滤波。
    ksize: 滤波器的大小
    """
    # 将PIL Image转换为NumPy数组
    img = np.array(img)
    
    # 使用Scipy的uniform_filter进行均值滤波
    img_filtered = cv2.blur(img, ksize)  # 对图像进行均值滤波
    
    # 将NumPy数组转换回PIL Image
    img_filtered = Image.fromarray(img_filtered.astype(np.uint8))
    
    return img_filtered


"""     高斯滤波    """
def gaussian_filtering(img, ksize):
    # 将PIL图像转换为numpy数组
    img = np.array(img)
    
    # 对图像应用高斯滤波
    img_filtered = cv2.GaussianBlur(img, ksize, 0)

    # 将处理后的图像转换回PIL图像
    img_filtered = Image.fromarray(img_filtered)
    return img_filtered

"""     JEPG压缩    """
def jpeg_compress(image, quality):
    """
    对输入图像进行 JPEG 压缩并返回 PIL 图像格式。
    Args:
        image: 输入的 PIL 图像对象。
        quality: JPEG 压缩质量，值越低压缩越强（1-100）。
    return: 
        压缩后的图像，格式为 PIL.Image。
    """
    # 使用 BytesIO 创建一个 buffer
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)  # 以指定质量保存为 JPEG
    buffer.seek(0)  # 回到 buffer 的开始位置
    
    # 读取压缩后的图像并转换为 PIL 图像
    compressed_image = Image.open(buffer)
    
    return compressed_image  # 返回压缩后的 PIL 图像对象
