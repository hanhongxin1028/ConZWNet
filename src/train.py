import torch
from tqdm import tqdm
from utils import calculate_robust_nc,calculate_collision_nc

'''
    阶段一: 鲁棒特征提取器
'''
def train_phase1(model, train_loader, epoch, optimizer, log_utils, device, scaler, SimCLR_loss, temperature):
    """
        作用 ：训练模型
        Args : 
             model : 定义的模型
             train_loader : 训练数据集
             epoch : 迭代数
             optimizer : 优化器
             log_utils : 记录日志
    """
    # 1. 初始化变量
    epoch_loss = 0.0  # 存储每个epoch的总损失
    num_batches = len(train_loader)  # 记录总批次数
    pbar = tqdm(train_loader, total=num_batches, desc=f"Epoch {epoch}", dynamic_ncols=True)  # 进度条
    
    
    # 2. 遍历数据集，进行训练
    for batch_id, (original_img, weak_img, strong_img) in enumerate(train_loader):

        weak_img = weak_img.to(device, non_blocking=True)
        strong_img = strong_img.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled= True):
            # 2.1 前向传播
            _, out_weak  = model(weak_img)
            _, out_strong  = model(strong_img)

            # 2.2 compute loss
            loss = SimCLR_loss(out_weak, out_strong, temperature)
        

        # 反向传播 (混合精度)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        # 2.5 累加整轮的损失和准确率
        epoch_loss += loss.item()

        # 2.6 更新 tqdm 进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}'
        })
        pbar.update(1)  # 更新进度条

    pbar.close()    
    # 3. 每个epoch结束后，计算平均损失和准确率
    avg_loss = epoch_loss / num_batches     # 每一轮的平均loss值
    log_utils.info(f'Train --> Epoch : {epoch}  avg_loss: {avg_loss:.4f} ')
            
    torch.cuda.empty_cache() # 每一个epoch清理一次 GPU 缓存，释放未使用的内存

def valid_phase1(model, val_loader, epoch, log_utils, device, SimCLR_loss, temperature):
    """
        作用 ：验证模型
        Args : 
             model : 定义的模型
             valid_loader : 验证数据集
             epoch : 迭代数
             log_utils : 记录日志
             device : 设备
    """
    # 1. 初始化变量

    epoch_loss = 0.0  # 存储每个epoch的总损失
    epoch_robust_nc = 0.0     # 用于衡量模型能不能抵抗攻击
    epoch_collision_nc = 0.0
    num_batches = len(val_loader)  # 记录总批次数
    pbar = tqdm(val_loader, total=num_batches, desc=f"Valid Epoch {epoch}", dynamic_ncols=True)  # 进度条

    # 2. 遍历数据集 , 进行验证
    with torch.no_grad():
        for batch_id, (original_img, weak_img, strong_img) in enumerate(val_loader):
            # 2.1 数据移动GPU设备
            original_img = original_img.to(device, non_blocking=True)
            weak_img = weak_img.to(device, non_blocking=True)
            strong_img = strong_img.to(device, non_blocking=True)

            # 2.2 前向传播
            original_img_features, _  = model(original_img)
            weak_img_features, out_weak  = model(weak_img)
            strong_img_features, out_strong  = model(strong_img)

            # compute loss
            loss = SimCLR_loss(out_weak, out_strong, temperature)

            # 2.6 累加整轮的损失、准确率、nc值
                # 2.6.1 各项 损失 累加
            # 2.5 计算鲁棒性(用于衡量模型能不能抵抗攻击)
            robust_min_nc = min(calculate_robust_nc(original_img_features , weak_img_features), calculate_robust_nc(original_img_features , strong_img_features))     
            
            # 计算碰撞性(用于衡量模型有没有较好的碰撞性)
            collision_max_nc = calculate_collision_nc(original_img_features)

            # 2.6 累加整轮的损失、准确率、nc值
                # 2.6.1 各项 损失 累加
            epoch_loss += loss.item()
                # 2.6.2 水印辨别的准确率 累加
                # 2.6.3 nc值 累加
            epoch_robust_nc += robust_min_nc
            epoch_collision_nc += collision_max_nc

            # 2.7 进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'robust_nc': f'{robust_min_nc:.4f}',
                'collision_nc': f'{collision_max_nc:.4f}'
            })
            
            pbar.update(1)  # 更新进度条

    pbar.close()

    # 3. 每个epoch结束后，计算平均损失和准确率
        # 3.1 各 损失项 均值
    avg_loss = epoch_loss / num_batches     # 每一轮的平均loss值
    avg_robust_nc = epoch_robust_nc / num_batches        # 每一轮的平均 nc值
    avg_collision_nc = epoch_collision_nc / num_batches

    log_utils.info(f'Valid --> Epoch : {epoch}   avg_loss: {avg_loss:.4f} , avg_robust_nc:{avg_robust_nc:.4f},  avg_collision_nc: {avg_collision_nc:.4f}')
    
    torch.cuda.empty_cache() # 每一个epoch清理一次 GPU 缓存，释放未使用的内存
    
    return avg_robust_nc, avg_collision_nc

def test_phase1(model, test_loader, log_utils, device):
    num_batches = len(test_loader)  # 记录总批次数
    pbar = tqdm(test_loader, total=num_batches,  dynamic_ncols=True)  # 进度条

    # 初始化
    # region epoch_nc
    epoch_collision_nc = 0.0
    epoch_rotation_10_nc = 0.0
    epoch_rotation_30_nc = 0.0
    epoch_rotation_60_nc = 0.0
    epoch_rotation_120_nc = 0.0
    epoch_rotation_150_nc = 0.0
    epoch_rotation_180_nc = 0.0
    epoch_sp_noise_1_nc = 0.0
    epoch_sp_noise_2_nc = 0.0
    epoch_sp_noise_3_nc = 0.0
    epoch_gaussian_noise_1_nc = 0.0
    epoch_gaussian_noise_2_nc = 0.0
    epoch_gaussian_noise_3_nc = 0.0
    epoch_resize_1_nc = 0.0
    epoch_resize_2_nc = 0.0
    epoch_resize_3_nc = 0.0
    epoch_resize_4_nc = 0.0
    epoch_cornerCrop_left_top_nc = 0.0
    epoch_cornerCrop_right_top_nc = 0.0
    epoch_cornerCrop_left_bottom_nc = 0.0
    epoch_cornerCrop_right_bottom_nc = 0.0
    epoch_gaussian_filter_3_nc = 0.0
    epoch_gaussian_filter_5_nc = 0.0
    epoch_gaussian_filter_9_nc = 0.0
    epoch_gaussian_filter_11_nc = 0.0
    epoch_median_filter_3_nc = 0.0
    epoch_median_filter_5_nc = 0.0
    epoch_median_filter_9_nc = 0.0
    epoch_median_filter_11_nc = 0.0
    epoch_jpeg_compress_20_nc = 0.0
    epoch_jpeg_compress_40_nc = 0.0
    epoch_jpeg_compress_60_nc = 0.0
    epoch_jpeg_compress_80_nc = 0.0
    # endregion

    # 2. 遍历数据集 , 进行验证
    with torch.no_grad():
        for batch_id, (
                        original_img, rotation_10, rotation_30, rotation_60, rotation_120, rotation_150, rotation_180,
                        sp_noise_1, sp_noise_2, sp_noise_3, 
                        gaussian_noise_1, gaussian_noise_2, gaussian_noise_3,
                        resize_1, resize_2, resize_3, resize_4, 
                        cornerCrop_left_top, cornerCrop_right_top,cornerCrop_left_bottom, cornerCrop_right_bottom, 
                        gaussian_filter_3, gaussian_filter_5, gaussian_filter_9, gaussian_filter_11, 
                        median_filter_3, median_filter_5, median_filter_9,median_filter_11, 
                        jpeg_compress_20, jpeg_compress_40, jpeg_compress_60, jpeg_compress_80
                       ) in enumerate(test_loader):

            # 移到GPU
            # region toGPU
            original_img = original_img.to(device, non_blocking=True)
            rotation_10 = rotation_10.to(device, non_blocking=True)
            rotation_30 = rotation_30.to(device, non_blocking=True)
            rotation_60 = rotation_60.to(device, non_blocking=True)
            rotation_120 = rotation_120.to(device, non_blocking=True)
            rotation_150 = rotation_150.to(device, non_blocking=True)
            rotation_180 = rotation_180.to(device, non_blocking=True)
            sp_noise_1 = sp_noise_1.to(device, non_blocking=True)
            sp_noise_2 = sp_noise_2.to(device, non_blocking=True)
            sp_noise_3 = sp_noise_3.to(device, non_blocking=True)
            gaussian_noise_1 = gaussian_noise_1.to(device, non_blocking=True)
            gaussian_noise_2 = gaussian_noise_2.to(device, non_blocking=True)
            gaussian_noise_3 = gaussian_noise_3.to(device, non_blocking=True)
            resize_1 = resize_1.to(device, non_blocking=True)
            resize_2 = resize_2.to(device, non_blocking=True)
            resize_3 = resize_3.to(device, non_blocking=True)
            resize_4 = resize_4.to(device, non_blocking=True)
            cornerCrop_left_top = cornerCrop_left_top.to(device, non_blocking=True)
            cornerCrop_right_top = cornerCrop_right_top.to(device, non_blocking=True)
            cornerCrop_left_bottom = cornerCrop_left_bottom.to(device, non_blocking=True)
            cornerCrop_right_bottom = cornerCrop_right_bottom.to(device, non_blocking=True)
            gaussian_filter_3 = gaussian_filter_3.to(device, non_blocking=True)
            gaussian_filter_5 = gaussian_filter_5.to(device, non_blocking=True)
            gaussian_filter_9 = gaussian_filter_9.to(device, non_blocking=True)
            gaussian_filter_11 = gaussian_filter_11.to(device, non_blocking=True)
            median_filter_3 = median_filter_3.to(device, non_blocking=True)
            median_filter_5 = median_filter_5.to(device, non_blocking=True)
            median_filter_9 = median_filter_9.to(device, non_blocking=True)
            median_filter_11 = median_filter_11.to(device, non_blocking=True)
            jpeg_compress_20 = jpeg_compress_20.to(device, non_blocking=True)
            jpeg_compress_40 = jpeg_compress_40.to(device, non_blocking=True)
            jpeg_compress_60 = jpeg_compress_60.to(device, non_blocking=True)
            jpeg_compress_80 = jpeg_compress_80.to(device, non_blocking=True)
            # endregion

            # 前向传播
            # region forward
            original_features, _ = model(original_img)
            rotation_10_features, _ = model(rotation_10)
            rotation_30_features, _ = model(rotation_30)
            rotation_60_features, _ = model(rotation_60)
            rotation_120_features, _ = model(rotation_120)
            rotation_150_features, _ = model(rotation_150)
            rotation_180_features, _ = model(rotation_180)
            sp_noise_1_features, _ = model(sp_noise_1)
            sp_noise_2_features, _ = model(sp_noise_2)
            sp_noise_3_features, _ = model(sp_noise_3)
            gaussian_noise_1_features, _ = model(gaussian_noise_1)
            gaussian_noise_2_features, _ = model(gaussian_noise_2)
            gaussian_noise_3_features, _ = model(gaussian_noise_3)
            resize_1_features, _ = model(resize_1)
            resize_2_features, _ = model(resize_2)
            resize_3_features, _ = model(resize_3)
            resize_4_features, _ = model(resize_4)
            cornerCrop_left_top_features, _ = model(cornerCrop_left_top)
            cornerCrop_right_top_features, _ = model(cornerCrop_right_top)
            cornerCrop_left_bottom_features, _ = model(cornerCrop_left_bottom)
            cornerCrop_right_bottom_features, _ = model(cornerCrop_right_bottom)
            gaussian_filter_3_features, _ = model(gaussian_filter_3)
            gaussian_filter_5_features, _ = model(gaussian_filter_5)
            gaussian_filter_9_features, _ = model(gaussian_filter_9)
            gaussian_filter_11_features, _ = model(gaussian_filter_11)
            median_filter_3_features, _ = model(median_filter_3)
            median_filter_5_features, _ = model(median_filter_5)
            median_filter_9_features, _ = model(median_filter_9)
            median_filter_11_features, _ = model(median_filter_11)
            jpeg_compress_20_features, _ = model(jpeg_compress_20)
            jpeg_compress_40_features, _ = model(jpeg_compress_40)
            jpeg_compress_60_features, _ = model(jpeg_compress_60)
            jpeg_compress_80_features, _ = model(jpeg_compress_80)
            # endregion

            # 计算NC
            # region cal_nc
            collision_nc = calculate_collision_nc(original_features)
            rotation_10_nc =calculate_robust_nc(original_features, rotation_10_features)
            rotation_30_nc = calculate_robust_nc(original_features, rotation_30_features)
            rotation_60_nc = calculate_robust_nc(original_features, rotation_60_features)
            rotation_120_nc = calculate_robust_nc(original_features, rotation_120_features)
            rotation_150_nc = calculate_robust_nc(original_features, rotation_150_features)
            rotation_180_nc = calculate_robust_nc(original_features, rotation_180_features)
            sp_noise_1_nc = calculate_robust_nc(original_features, sp_noise_1_features)
            sp_noise_2_nc = calculate_robust_nc(original_features, sp_noise_2_features)
            sp_noise_3_nc = calculate_robust_nc(original_features, sp_noise_3_features)
            gaussian_noise_1_nc = calculate_robust_nc(original_features, gaussian_noise_1_features)
            gaussian_noise_2_nc = calculate_robust_nc(original_features, gaussian_noise_2_features)
            gaussian_noise_3_nc = calculate_robust_nc(original_features, gaussian_noise_3_features)
            resize_1_nc = calculate_robust_nc(original_features, resize_1_features)
            resize_2_nc = calculate_robust_nc(original_features, resize_2_features)
            resize_3_nc = calculate_robust_nc(original_features, resize_3_features)
            resize_4_nc = calculate_robust_nc(original_features, resize_4_features)
            cornerCrop_left_top_nc = calculate_robust_nc(original_features, cornerCrop_left_top_features)
            cornerCrop_right_top_nc = calculate_robust_nc(original_features, cornerCrop_right_top_features)
            cornerCrop_left_bottom_nc = calculate_robust_nc(original_features, cornerCrop_left_bottom_features)
            cornerCrop_right_bottom_nc = calculate_robust_nc(original_features, cornerCrop_right_bottom_features)
            gaussian_filter_3_nc = calculate_robust_nc(original_features, gaussian_filter_3_features)
            gaussian_filter_5_nc = calculate_robust_nc(original_features, gaussian_filter_5_features)
            gaussian_filter_9_nc = calculate_robust_nc(original_features, gaussian_filter_9_features)
            gaussian_filter_11_nc = calculate_robust_nc(original_features, gaussian_filter_11_features)
            median_filter_3_nc = calculate_robust_nc(original_features, median_filter_3_features)
            median_filter_5_nc = calculate_robust_nc(original_features, median_filter_5_features)
            median_filter_9_nc = calculate_robust_nc(original_features, median_filter_9_features)
            median_filter_11_nc = calculate_robust_nc(original_features, median_filter_11_features)
            jpeg_compress_20_nc = calculate_robust_nc(original_features, jpeg_compress_20_features)
            jpeg_compress_40_nc = calculate_robust_nc(original_features, jpeg_compress_40_features)
            jpeg_compress_60_nc = calculate_robust_nc(original_features, jpeg_compress_60_features)
            jpeg_compress_80_nc = calculate_robust_nc(original_features, jpeg_compress_80_features)
            # endregion

            # 累加
            # region epoch_nc++
            epoch_collision_nc += collision_nc
            epoch_rotation_10_nc += rotation_10_nc
            epoch_rotation_30_nc += rotation_30_nc
            epoch_rotation_60_nc += rotation_60_nc
            epoch_rotation_120_nc += rotation_120_nc
            epoch_rotation_150_nc += rotation_150_nc
            epoch_rotation_180_nc += rotation_180_nc
            epoch_sp_noise_1_nc += sp_noise_1_nc
            epoch_sp_noise_2_nc += sp_noise_2_nc
            epoch_sp_noise_3_nc += sp_noise_3_nc
            epoch_gaussian_noise_1_nc += gaussian_noise_1_nc
            epoch_gaussian_noise_2_nc += gaussian_noise_2_nc
            epoch_gaussian_noise_3_nc += gaussian_noise_3_nc
            epoch_resize_1_nc += resize_1_nc
            epoch_resize_2_nc += resize_2_nc
            epoch_resize_3_nc += resize_3_nc
            epoch_resize_4_nc += resize_4_nc
            epoch_cornerCrop_left_top_nc += cornerCrop_left_top_nc
            epoch_cornerCrop_right_top_nc += cornerCrop_right_top_nc
            epoch_cornerCrop_left_bottom_nc += cornerCrop_left_bottom_nc
            epoch_cornerCrop_right_bottom_nc += cornerCrop_right_bottom_nc
            epoch_gaussian_filter_3_nc += gaussian_filter_3_nc
            epoch_gaussian_filter_5_nc += gaussian_filter_5_nc
            epoch_gaussian_filter_9_nc += gaussian_filter_9_nc
            epoch_gaussian_filter_11_nc += gaussian_filter_11_nc
            epoch_median_filter_3_nc += median_filter_3_nc
            epoch_median_filter_5_nc += median_filter_5_nc
            epoch_median_filter_9_nc += median_filter_9_nc
            epoch_median_filter_11_nc += median_filter_11_nc
            epoch_jpeg_compress_20_nc += jpeg_compress_20_nc
            epoch_jpeg_compress_40_nc += jpeg_compress_40_nc
            epoch_jpeg_compress_60_nc += jpeg_compress_60_nc
            epoch_jpeg_compress_80_nc += jpeg_compress_80_nc
            # endregion

            # 2.7 进度条
            pbar.set_postfix({
                'collision_nc': f'{collision_nc:.4f}',
                'rotation_180_nc': f'{rotation_180_nc:.4f}',
                'sp_noise_3_nc': f'{sp_noise_3_nc:.4f}',
                'gaussian_noise_3_nc': f'{gaussian_noise_3_nc:.4f}',
                'resize_1_nc': f'{resize_1_nc:.4f}',
                'cornerCrop_left_top_nc': f'{cornerCrop_left_top_nc:.4f}',
                'gaussian_filter_11_nc': f'{gaussian_filter_11_nc:.4f}',
                'median_filter_11_nc': f'{median_filter_11_nc:.4f}',
                'jpeg_compress_20_nc': f'{jpeg_compress_20_nc:.4f}'
            })

            pbar.update(1)  # 更新进度条

    pbar.close()

    # 计算均值
    # region avg_nc
    avg_collision_nc = epoch_collision_nc / num_batches
    avg_rotation_10_nc = epoch_rotation_10_nc / num_batches
    avg_rotation_30_nc = epoch_rotation_30_nc / num_batches
    avg_rotation_60_nc = epoch_rotation_60_nc / num_batches
    avg_rotation_120_nc = epoch_rotation_120_nc / num_batches
    avg_rotation_150_nc = epoch_rotation_150_nc / num_batches
    avg_rotation_180_nc = epoch_rotation_180_nc / num_batches
    avg_sp_noise_1_nc = epoch_sp_noise_1_nc / num_batches
    avg_sp_noise_2_nc = epoch_sp_noise_2_nc / num_batches
    avg_sp_noise_3_nc = epoch_sp_noise_3_nc / num_batches
    avg_gaussian_noise_1_nc = epoch_gaussian_noise_1_nc / num_batches
    avg_gaussian_noise_2_nc = epoch_gaussian_noise_2_nc / num_batches
    avg_gaussian_noise_3_nc = epoch_gaussian_noise_3_nc / num_batches
    avg_resize_1_nc = epoch_resize_1_nc / num_batches
    avg_resize_2_nc = epoch_resize_2_nc / num_batches
    avg_resize_3_nc = epoch_resize_3_nc / num_batches
    avg_resize_4_nc = epoch_resize_4_nc / num_batches
    avg_cornerCrop_left_top_nc = epoch_cornerCrop_left_top_nc / num_batches
    avg_cornerCrop_right_top_nc = epoch_cornerCrop_right_top_nc / num_batches
    avg_cornerCrop_left_bottom_nc = epoch_cornerCrop_left_bottom_nc / num_batches
    avg_cornerCrop_right_bottom_nc = epoch_cornerCrop_right_bottom_nc / num_batches
    avg_gaussian_filter_3_nc = epoch_gaussian_filter_3_nc / num_batches
    avg_gaussian_filter_5_nc = epoch_gaussian_filter_5_nc / num_batches
    avg_gaussian_filter_9_nc = epoch_gaussian_filter_9_nc / num_batches
    avg_gaussian_filter_11_nc = epoch_gaussian_filter_11_nc / num_batches
    avg_median_filter_3_nc = epoch_median_filter_3_nc / num_batches
    avg_median_filter_5_nc = epoch_median_filter_5_nc / num_batches
    avg_median_filter_9_nc = epoch_median_filter_9_nc / num_batches
    avg_median_filter_11_nc = epoch_median_filter_11_nc / num_batches
    avg_jpeg_compress_20_nc = epoch_jpeg_compress_20_nc / num_batches
    avg_jpeg_compress_40_nc = epoch_jpeg_compress_40_nc / num_batches
    avg_jpeg_compress_60_nc = epoch_jpeg_compress_60_nc / num_batches
    avg_jpeg_compress_80_nc = epoch_jpeg_compress_80_nc / num_batches
    # endregion

    log_utils.info(f"avg_collision_nc: {avg_collision_nc},\navg_rotation_10_nc: {avg_rotation_10_nc},\n avg_rotation_30_nc: {avg_rotation_30_nc},\n avg_rotation_60_nc: {avg_rotation_60_nc},\n avg_rotation_120_nc: {avg_rotation_120_nc},\n avg_rotation_150_nc: {avg_rotation_150_nc},\n avg_rotation_180_nc: {avg_rotation_180_nc},\n avg_sp_noise_1_nc: {avg_sp_noise_1_nc},\n avg_sp_noise_2_nc: {avg_sp_noise_2_nc},\n avg_sp_noise_3_nc: {avg_sp_noise_3_nc},\n avg_gaussian_noise_1_nc: {avg_gaussian_noise_1_nc},\n avg_gaussian_noise_2_nc: {avg_gaussian_noise_2_nc},\n avg_gaussian_noise_3_nc: {avg_gaussian_noise_3_nc},\n avg_resize_1_nc: {avg_resize_1_nc},\n avg_resize_2_nc: {avg_resize_2_nc},\n avg_resize_3_nc: {avg_resize_3_nc},\n avg_resize_4_nc: {avg_resize_4_nc},\n avg_cornerCrop_left_top_nc: {avg_cornerCrop_left_top_nc},\n avg_cornerCrop_right_top_nc: {avg_cornerCrop_right_top_nc},\n avg_cornerCrop_left_bottom_nc: {avg_cornerCrop_left_bottom_nc},\n avg_cornerCrop_right_bottom_nc: {avg_cornerCrop_right_bottom_nc},\n avg_gaussian_filter_3_nc: {avg_gaussian_filter_3_nc},\n avg_gaussian_filter_5_nc: {avg_gaussian_filter_5_nc},\n avg_gaussian_filter_9_nc: {avg_gaussian_filter_9_nc},\n avg_gaussian_filter_11_nc: {avg_gaussian_filter_11_nc},\n avg_median_filter_3_nc: {avg_median_filter_3_nc},\n avg_median_filter_5_nc: {avg_median_filter_5_nc},\n avg_median_filter_9_nc: {avg_median_filter_9_nc},\n avg_median_filter_11_nc: {avg_median_filter_11_nc},\n avg_jpeg_compress_20_nc: {avg_jpeg_compress_20_nc},\n avg_jpeg_compress_40_nc: {avg_jpeg_compress_40_nc},\n avg_jpeg_compress_60_nc: {avg_jpeg_compress_60_nc},\n avg_jpeg_compress_80_nc: {avg_jpeg_compress_80_nc},\n")



'''
    阶段二: zero-watermark 辨别版权任务
'''
def train_phase2(model, train_loader, epoch, optimizer, log_utils, device, scaler):
    """
        作用 ：训练模型
        Args : 
             model : 定义的模型
             train_loader : 训练数据集
             epoch : 迭代数
             optimizer : 优化器
             log_utils : 记录日志
    """
    # 1. 初始化变量
    epoch_loss = 0.0  # 存储每个epoch的总损失
    epoch_accuracy_watermark_classifier = 0.0  # 存储每个epoch的总准确率
    num_batches = len(train_loader)  # 记录总批次数
    pbar = tqdm(train_loader, total=num_batches, desc=f"Epoch {epoch}", dynamic_ncols=True)  # 进度条

    ce_loss_fn = torch.nn.CrossEntropyLoss()
    # 2. 遍历数据集，进行训练
    for batch_id, (host_img, copyright_img, copyright_label) in enumerate(train_loader):

        host_img = host_img.to(device, non_blocking=True)
        copyright_img = copyright_img.to(device, non_blocking=True)
        copyright_label = copyright_label.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled= True):
            # 2.1 前向传播  
            zeroWatermark_featuremap, copyright_img_pred  = model(host_img, copyright_img)

            # 2.2 compute loss
            loss = ce_loss_fn(copyright_img_pred, copyright_label)

        # 反向传播 (混合精度)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        watermark_predictor_class = torch.argmax(copyright_img_pred, dim=1)
        watermark_correct_predictions = watermark_predictor_class == copyright_label
        watermark_classifier_accuracy = watermark_correct_predictions.sum().item() / copyright_label.size(0)  # 计算准确率

        # 2.5 累加整轮的损失和准确率
        epoch_loss += loss.item()
        epoch_accuracy_watermark_classifier += watermark_classifier_accuracy

        # 2.6 更新 tqdm 进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'accuracy': f'{watermark_classifier_accuracy:.4f}'
        })
        pbar.update(1)  # 更新进度条

    pbar.close()    

    # 3. 每个epoch结束后，计算平均损失和准确率
    avg_loss = epoch_loss / num_batches     # 每一轮的平均loss值
    avg_accuracy_watermark_classifier = epoch_accuracy_watermark_classifier / num_batches

    log_utils.info(f'Train --> Epoch : {epoch}  avg_loss: {avg_loss:.4f},  avg_accuracy: {avg_accuracy_watermark_classifier:.4f}')
            
    torch.cuda.empty_cache() # 每一个epoch清理一次 GPU 缓存，释放未使用的内存

def valid_phase2(model, val_loader, epoch, log_utils, device):
    """
        作用 ：验证模型
        Args : 
             model : 定义的模型
             valid_loader : 验证数据集
             epoch : 迭代数
             log_utils : 记录日志
             device : 设备
    """
    # 1. 初始化变量

    epoch_loss = 0.0  # 存储每个epoch的总损失
    epoch_accuracy_watermark_classifier = 0.0  # 存储每个epoch的总准确率
    epoch_collision_nc = 0.0
    num_batches = len(val_loader)  # 记录总批次数
    pbar = tqdm(val_loader, total=num_batches, desc=f"Valid Epoch {epoch}", dynamic_ncols=True)  # 进度条

    ce_loss_fn = torch.nn.CrossEntropyLoss()

    # 2. 遍历数据集 , 进行验证
    with torch.no_grad():
        for batch_id, (host_img, copyright_img, copyright_label) in enumerate(val_loader):
            # 数据移动GPU设备
            host_img = host_img.to(device, non_blocking=True)
            copyright_img = copyright_img.to(device, non_blocking=True)
            copyright_label = copyright_label.to(device, non_blocking=True)

            # 2.1 前向传播  
            zeroWatermark_featuremap, copyright_img_pred  = model(host_img, copyright_img)

            # 2.2 compute loss
            loss = ce_loss_fn(copyright_img_pred, copyright_label)

            watermark_predictor_class = torch.argmax(copyright_img_pred, dim=1)
            watermark_correct_predictions = watermark_predictor_class == copyright_label
            watermark_classifier_accuracy = watermark_correct_predictions.sum().item() / copyright_label.size(0)  # 计算准确率

            # 计算碰撞性(用于衡量模型有没有较好的碰撞性)
            collision_nc = calculate_collision_nc(zeroWatermark_featuremap)

            # 累加整轮的损失、准确率
            epoch_loss += loss.item()
            epoch_accuracy_watermark_classifier += watermark_classifier_accuracy
            epoch_collision_nc += collision_nc

            
            # 2.7 进度条
            pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'accuracy': f'{watermark_classifier_accuracy:.4f}',
            'collision_nc': f'{collision_nc:.4f}'
            })
            
            pbar.update(1)  # 更新进度条

    pbar.close()

    # 3. 每个epoch结束后，计算平均损失和准确率
        # 3.1 各 损失项 均值
    avg_loss = epoch_loss / num_batches     # 每一轮的平均loss值
    avg_accuracy_watermark_classifier = epoch_accuracy_watermark_classifier / num_batches
    avg_collision_nc = epoch_collision_nc / num_batches

    log_utils.info(f'Valid --> Epoch : {epoch}  avg_loss: {avg_loss:.4f},  avg_accuracy: {avg_accuracy_watermark_classifier:.4f},  avg_collision_nc: {avg_collision_nc:.4f}')

    return avg_accuracy_watermark_classifier, avg_collision_nc

def test_phase2(model, test_loader, log_utils, device):
    num_batches = len(test_loader)  # 记录总批次数
    pbar = tqdm(test_loader, total=num_batches,  dynamic_ncols=True)  # 进度条

    # 初始化
    # region epoch_nc
    epoch_host_collision_nc = 0.0
    epoch_rotation_10_nc = 0.0
    epoch_rotation_30_nc = 0.0
    epoch_rotation_60_nc = 0.0
    epoch_rotation_120_nc = 0.0
    epoch_rotation_150_nc = 0.0
    epoch_rotation_180_nc = 0.0
    epoch_sp_noise_1_nc = 0.0
    epoch_sp_noise_2_nc = 0.0
    epoch_sp_noise_3_nc = 0.0
    epoch_gaussian_noise_1_nc = 0.0
    epoch_gaussian_noise_2_nc = 0.0
    epoch_gaussian_noise_3_nc = 0.0
    epoch_resize_1_nc = 0.0
    epoch_resize_2_nc = 0.0
    epoch_resize_3_nc = 0.0
    epoch_resize_4_nc = 0.0
    epoch_cornerCrop_left_top_nc = 0.0
    epoch_cornerCrop_right_top_nc = 0.0
    epoch_cornerCrop_left_bottom_nc = 0.0
    epoch_cornerCrop_right_bottom_nc = 0.0
    epoch_gaussian_filter_3_nc = 0.0
    epoch_gaussian_filter_5_nc = 0.0
    epoch_gaussian_filter_9_nc = 0.0
    epoch_gaussian_filter_11_nc = 0.0
    epoch_median_filter_3_nc = 0.0
    epoch_median_filter_5_nc = 0.0
    epoch_median_filter_9_nc = 0.0
    epoch_median_filter_11_nc = 0.0
    epoch_jpeg_compress_20_nc = 0.0
    epoch_jpeg_compress_40_nc = 0.0
    epoch_jpeg_compress_60_nc = 0.0
    epoch_jpeg_compress_80_nc = 0.0
    # endregion

    # 2. 遍历数据集 , 进行验证
    with torch.no_grad():
        for batch_id, (
                        original_img, 
                        rotation_10, rotation_30, rotation_60, rotation_120, rotation_150, rotation_180,
                        sp_noise_1, sp_noise_2, sp_noise_3, 
                        gaussian_noise_1, gaussian_noise_2, gaussian_noise_3,
                        resize_1, resize_2, resize_3, resize_4, 
                        cornerCrop_left_top, cornerCrop_right_top,cornerCrop_left_bottom, cornerCrop_right_bottom, 
                        gaussian_filter_3, gaussian_filter_5, gaussian_filter_9, gaussian_filter_11, 
                        median_filter_3, median_filter_5, median_filter_9,median_filter_11, 
                        jpeg_compress_20, jpeg_compress_40, jpeg_compress_60, jpeg_compress_80,
                        copyright_img
                       ) in enumerate(test_loader):

            # 移到GPU
            # region toGPU
            original_img = original_img.to(device, non_blocking=True)
            rotation_10 = rotation_10.to(device, non_blocking=True)
            rotation_30 = rotation_30.to(device, non_blocking=True)
            rotation_60 = rotation_60.to(device, non_blocking=True)
            rotation_120 = rotation_120.to(device, non_blocking=True)
            rotation_150 = rotation_150.to(device, non_blocking=True)
            rotation_180 = rotation_180.to(device, non_blocking=True)
            sp_noise_1 = sp_noise_1.to(device, non_blocking=True)
            sp_noise_2 = sp_noise_2.to(device, non_blocking=True)
            sp_noise_3 = sp_noise_3.to(device, non_blocking=True)
            gaussian_noise_1 = gaussian_noise_1.to(device, non_blocking=True)
            gaussian_noise_2 = gaussian_noise_2.to(device, non_blocking=True)
            gaussian_noise_3 = gaussian_noise_3.to(device, non_blocking=True)
            resize_1 = resize_1.to(device, non_blocking=True)
            resize_2 = resize_2.to(device, non_blocking=True)
            resize_3 = resize_3.to(device, non_blocking=True)
            resize_4 = resize_4.to(device, non_blocking=True)
            cornerCrop_left_top = cornerCrop_left_top.to(device, non_blocking=True)
            cornerCrop_right_top = cornerCrop_right_top.to(device, non_blocking=True)
            cornerCrop_left_bottom = cornerCrop_left_bottom.to(device, non_blocking=True)
            cornerCrop_right_bottom = cornerCrop_right_bottom.to(device, non_blocking=True)
            gaussian_filter_3 = gaussian_filter_3.to(device, non_blocking=True)
            gaussian_filter_5 = gaussian_filter_5.to(device, non_blocking=True)
            gaussian_filter_9 = gaussian_filter_9.to(device, non_blocking=True)
            gaussian_filter_11 = gaussian_filter_11.to(device, non_blocking=True)
            median_filter_3 = median_filter_3.to(device, non_blocking=True)
            median_filter_5 = median_filter_5.to(device, non_blocking=True)
            median_filter_9 = median_filter_9.to(device, non_blocking=True)
            median_filter_11 = median_filter_11.to(device, non_blocking=True)
            jpeg_compress_20 = jpeg_compress_20.to(device, non_blocking=True)
            jpeg_compress_40 = jpeg_compress_40.to(device, non_blocking=True)
            jpeg_compress_60 = jpeg_compress_60.to(device, non_blocking=True)
            jpeg_compress_80 = jpeg_compress_80.to(device, non_blocking=True)
            copyright_img = copyright_img.to(device, non_blocking=True)
            # endregion

            # 前向传播
            # region forward
            original_features, _ = model(original_img, copyright_img)
            rotation_10_features, _ = model(rotation_10, copyright_img)
            rotation_30_features, _ = model(rotation_30, copyright_img)
            rotation_60_features, _ = model(rotation_60, copyright_img)
            rotation_120_features, _ = model(rotation_120, copyright_img)
            rotation_150_features, _ = model(rotation_150, copyright_img)
            rotation_180_features, _ = model(rotation_180, copyright_img)
            sp_noise_1_features, _ = model(sp_noise_1, copyright_img)
            sp_noise_2_features, _ = model(sp_noise_2, copyright_img)
            sp_noise_3_features, _ = model(sp_noise_3, copyright_img)
            gaussian_noise_1_features, _ = model(gaussian_noise_1, copyright_img)
            gaussian_noise_2_features, _ = model(gaussian_noise_2, copyright_img)
            gaussian_noise_3_features, _ = model(gaussian_noise_3, copyright_img)
            resize_1_features, _ = model(resize_1, copyright_img)
            resize_2_features, _ = model(resize_2, copyright_img)
            resize_3_features, _ = model(resize_3, copyright_img)
            resize_4_features, _ = model(resize_4, copyright_img)
            cornerCrop_left_top_features, _ = model(cornerCrop_left_top, copyright_img)
            cornerCrop_right_top_features, _ = model(cornerCrop_right_top, copyright_img)
            cornerCrop_left_bottom_features, _ = model(cornerCrop_left_bottom, copyright_img)
            cornerCrop_right_bottom_features, _ = model(cornerCrop_right_bottom, copyright_img)
            gaussian_filter_3_features, _ = model(gaussian_filter_3, copyright_img)
            gaussian_filter_5_features, _ = model(gaussian_filter_5, copyright_img)
            gaussian_filter_9_features, _ = model(gaussian_filter_9, copyright_img)
            gaussian_filter_11_features, _ = model(gaussian_filter_11, copyright_img)
            median_filter_3_features, _ = model(median_filter_3, copyright_img)
            median_filter_5_features, _ = model(median_filter_5, copyright_img)
            median_filter_9_features, _ = model(median_filter_9, copyright_img)
            median_filter_11_features, _ = model(median_filter_11, copyright_img)
            jpeg_compress_20_features, _ = model(jpeg_compress_20, copyright_img)
            jpeg_compress_40_features, _ = model(jpeg_compress_40, copyright_img)
            jpeg_compress_60_features, _ = model(jpeg_compress_60, copyright_img)
            jpeg_compress_80_features, _ = model(jpeg_compress_80, copyright_img)
            # endregion

            # 计算NC
            # region cal_nc
            collision_nc = calculate_collision_nc(original_features)
            rotation_10_nc =calculate_robust_nc(original_features, rotation_10_features)
            rotation_30_nc = calculate_robust_nc(original_features, rotation_30_features)
            rotation_60_nc = calculate_robust_nc(original_features, rotation_60_features)
            rotation_120_nc = calculate_robust_nc(original_features, rotation_120_features)
            rotation_150_nc = calculate_robust_nc(original_features, rotation_150_features)
            rotation_180_nc = calculate_robust_nc(original_features, rotation_180_features)
            sp_noise_1_nc = calculate_robust_nc(original_features, sp_noise_1_features)
            sp_noise_2_nc = calculate_robust_nc(original_features, sp_noise_2_features)
            sp_noise_3_nc = calculate_robust_nc(original_features, sp_noise_3_features)
            gaussian_noise_1_nc = calculate_robust_nc(original_features, gaussian_noise_1_features)
            gaussian_noise_2_nc = calculate_robust_nc(original_features, gaussian_noise_2_features)
            gaussian_noise_3_nc = calculate_robust_nc(original_features, gaussian_noise_3_features)
            resize_1_nc = calculate_robust_nc(original_features, resize_1_features)
            resize_2_nc = calculate_robust_nc(original_features, resize_2_features)
            resize_3_nc = calculate_robust_nc(original_features, resize_3_features)
            resize_4_nc = calculate_robust_nc(original_features, resize_4_features)
            cornerCrop_left_top_nc = calculate_robust_nc(original_features, cornerCrop_left_top_features)
            cornerCrop_right_top_nc = calculate_robust_nc(original_features, cornerCrop_right_top_features)
            cornerCrop_left_bottom_nc = calculate_robust_nc(original_features, cornerCrop_left_bottom_features)
            cornerCrop_right_bottom_nc = calculate_robust_nc(original_features, cornerCrop_right_bottom_features)
            gaussian_filter_3_nc = calculate_robust_nc(original_features, gaussian_filter_3_features)
            gaussian_filter_5_nc = calculate_robust_nc(original_features, gaussian_filter_5_features)
            gaussian_filter_9_nc = calculate_robust_nc(original_features, gaussian_filter_9_features)
            gaussian_filter_11_nc = calculate_robust_nc(original_features, gaussian_filter_11_features)
            median_filter_3_nc = calculate_robust_nc(original_features, median_filter_3_features)
            median_filter_5_nc = calculate_robust_nc(original_features, median_filter_5_features)
            median_filter_9_nc = calculate_robust_nc(original_features, median_filter_9_features)
            median_filter_11_nc = calculate_robust_nc(original_features, median_filter_11_features)
            jpeg_compress_20_nc = calculate_robust_nc(original_features, jpeg_compress_20_features)
            jpeg_compress_40_nc = calculate_robust_nc(original_features, jpeg_compress_40_features)
            jpeg_compress_60_nc = calculate_robust_nc(original_features, jpeg_compress_60_features)
            jpeg_compress_80_nc = calculate_robust_nc(original_features, jpeg_compress_80_features)
            # endregion

            # 累加
            # region epoch_nc++
            epoch_host_collision_nc += collision_nc
            epoch_rotation_10_nc += rotation_10_nc
            epoch_rotation_30_nc += rotation_30_nc
            epoch_rotation_60_nc += rotation_60_nc
            epoch_rotation_120_nc += rotation_120_nc
            epoch_rotation_150_nc += rotation_150_nc
            epoch_rotation_180_nc += rotation_180_nc
            epoch_sp_noise_1_nc += sp_noise_1_nc
            epoch_sp_noise_2_nc += sp_noise_2_nc
            epoch_sp_noise_3_nc += sp_noise_3_nc
            epoch_gaussian_noise_1_nc += gaussian_noise_1_nc
            epoch_gaussian_noise_2_nc += gaussian_noise_2_nc
            epoch_gaussian_noise_3_nc += gaussian_noise_3_nc
            epoch_resize_1_nc += resize_1_nc
            epoch_resize_2_nc += resize_2_nc
            epoch_resize_3_nc += resize_3_nc
            epoch_resize_4_nc += resize_4_nc
            epoch_cornerCrop_left_top_nc += cornerCrop_left_top_nc
            epoch_cornerCrop_right_top_nc += cornerCrop_right_top_nc
            epoch_cornerCrop_left_bottom_nc += cornerCrop_left_bottom_nc
            epoch_cornerCrop_right_bottom_nc += cornerCrop_right_bottom_nc
            epoch_gaussian_filter_3_nc += gaussian_filter_3_nc
            epoch_gaussian_filter_5_nc += gaussian_filter_5_nc
            epoch_gaussian_filter_9_nc += gaussian_filter_9_nc
            epoch_gaussian_filter_11_nc += gaussian_filter_11_nc
            epoch_median_filter_3_nc += median_filter_3_nc
            epoch_median_filter_5_nc += median_filter_5_nc
            epoch_median_filter_9_nc += median_filter_9_nc
            epoch_median_filter_11_nc += median_filter_11_nc
            epoch_jpeg_compress_20_nc += jpeg_compress_20_nc
            epoch_jpeg_compress_40_nc += jpeg_compress_40_nc
            epoch_jpeg_compress_60_nc += jpeg_compress_60_nc
            epoch_jpeg_compress_80_nc += jpeg_compress_80_nc
            # endregion

            # 2.7 进度条
            pbar.set_postfix({
                'collision_nc': f'{collision_nc:.4f}',
                'rotation_180_nc': f'{rotation_180_nc:.4f}',
                'sp_noise_3_nc': f'{sp_noise_3_nc:.4f}',
                'gaussian_noise_3_nc': f'{gaussian_noise_3_nc:.4f}',
                'resize_4_nc': f'{resize_4_nc:.4f}',
                'cornerCrop_left_top_nc': f'{cornerCrop_left_top_nc:.4f}',
                'gaussian_filter_11_nc': f'{gaussian_filter_11_nc:.4f}',
                'median_filter_11_nc': f'{median_filter_11_nc:.4f}',
                'jpeg_compress_20_nc': f'{jpeg_compress_20_nc:.4f}'
            })

            pbar.update(1)  # 更新进度条

    pbar.close()

    # 计算均值
    # region avg_nc
    avg_host_collision_nc = epoch_host_collision_nc / num_batches
    avg_rotation_10_nc = epoch_rotation_10_nc / num_batches
    avg_rotation_30_nc = epoch_rotation_30_nc / num_batches
    avg_rotation_60_nc = epoch_rotation_60_nc / num_batches
    avg_rotation_120_nc = epoch_rotation_120_nc / num_batches
    avg_rotation_150_nc = epoch_rotation_150_nc / num_batches
    avg_rotation_180_nc = epoch_rotation_180_nc / num_batches
    avg_sp_noise_1_nc = epoch_sp_noise_1_nc / num_batches
    avg_sp_noise_2_nc = epoch_sp_noise_2_nc / num_batches
    avg_sp_noise_3_nc = epoch_sp_noise_3_nc / num_batches
    avg_gaussian_noise_1_nc = epoch_gaussian_noise_1_nc / num_batches
    avg_gaussian_noise_2_nc = epoch_gaussian_noise_2_nc / num_batches
    avg_gaussian_noise_3_nc = epoch_gaussian_noise_3_nc / num_batches
    avg_resize_1_nc = epoch_resize_1_nc / num_batches
    avg_resize_2_nc = epoch_resize_2_nc / num_batches
    avg_resize_3_nc = epoch_resize_3_nc / num_batches
    avg_resize_4_nc = epoch_resize_4_nc / num_batches
    avg_cornerCrop_left_top_nc = epoch_cornerCrop_left_top_nc / num_batches
    avg_cornerCrop_right_top_nc = epoch_cornerCrop_right_top_nc / num_batches
    avg_cornerCrop_left_bottom_nc = epoch_cornerCrop_left_bottom_nc / num_batches
    avg_cornerCrop_right_bottom_nc = epoch_cornerCrop_right_bottom_nc / num_batches
    avg_gaussian_filter_3_nc = epoch_gaussian_filter_3_nc / num_batches
    avg_gaussian_filter_5_nc = epoch_gaussian_filter_5_nc / num_batches
    avg_gaussian_filter_9_nc = epoch_gaussian_filter_9_nc / num_batches
    avg_gaussian_filter_11_nc = epoch_gaussian_filter_11_nc / num_batches
    avg_median_filter_3_nc = epoch_median_filter_3_nc / num_batches
    avg_median_filter_5_nc = epoch_median_filter_5_nc / num_batches
    avg_median_filter_9_nc = epoch_median_filter_9_nc / num_batches
    avg_median_filter_11_nc = epoch_median_filter_11_nc / num_batches
    avg_jpeg_compress_20_nc = epoch_jpeg_compress_20_nc / num_batches
    avg_jpeg_compress_40_nc = epoch_jpeg_compress_40_nc / num_batches
    avg_jpeg_compress_60_nc = epoch_jpeg_compress_60_nc / num_batches
    avg_jpeg_compress_80_nc = epoch_jpeg_compress_80_nc / num_batches
    # endregion

    log_utils.info(f"host_collision_nc: {avg_host_collision_nc},\navg_rotation_10_nc: {avg_rotation_10_nc},\n avg_rotation_30_nc: {avg_rotation_30_nc},\n avg_rotation_60_nc: {avg_rotation_60_nc},\n avg_rotation_120_nc: {avg_rotation_120_nc},\n avg_rotation_150_nc: {avg_rotation_150_nc},\n avg_rotation_180_nc: {avg_rotation_180_nc},\n avg_sp_noise_1_nc: {avg_sp_noise_1_nc},\n avg_sp_noise_2_nc: {avg_sp_noise_2_nc},\n avg_sp_noise_3_nc: {avg_sp_noise_3_nc},\n avg_gaussian_noise_1_nc: {avg_gaussian_noise_1_nc},\n avg_gaussian_noise_2_nc: {avg_gaussian_noise_2_nc},\n avg_gaussian_noise_3_nc: {avg_gaussian_noise_3_nc},\n avg_resize_1_nc: {avg_resize_1_nc},\n avg_resize_2_nc: {avg_resize_2_nc},\n avg_resize_3_nc: {avg_resize_3_nc},\n avg_resize_4_nc: {avg_resize_4_nc},\n avg_cornerCrop_left_top_nc: {avg_cornerCrop_left_top_nc},\n avg_cornerCrop_right_top_nc: {avg_cornerCrop_right_top_nc},\n avg_cornerCrop_left_bottom_nc: {avg_cornerCrop_left_bottom_nc},\n avg_cornerCrop_right_bottom_nc: {avg_cornerCrop_right_bottom_nc},\n avg_gaussian_filter_3_nc: {avg_gaussian_filter_3_nc},\n avg_gaussian_filter_5_nc: {avg_gaussian_filter_5_nc},\n avg_gaussian_filter_9_nc: {avg_gaussian_filter_9_nc},\n avg_gaussian_filter_11_nc: {avg_gaussian_filter_11_nc},\n avg_median_filter_3_nc: {avg_median_filter_3_nc},\n avg_median_filter_5_nc: {avg_median_filter_5_nc},\n avg_median_filter_9_nc: {avg_median_filter_9_nc},\n avg_median_filter_11_nc: {avg_median_filter_11_nc},\n avg_jpeg_compress_20_nc: {avg_jpeg_compress_20_nc},\n avg_jpeg_compress_40_nc: {avg_jpeg_compress_40_nc},\n avg_jpeg_compress_60_nc: {avg_jpeg_compress_60_nc},\n avg_jpeg_compress_80_nc: {avg_jpeg_compress_80_nc},\n")
