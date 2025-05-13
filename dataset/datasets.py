# pylint: disable=line-too-long,too-many-lines,missing-docstring
import os
import warnings
import cv2
from collections import deque
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

from . import video_transforms, volume_transforms
from .loader import get_image_loader, get_video_loader
from .random_erasing import RandomErasing


class VideoClsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self,
                 anno_path,
                 data_root='',
                 mode='train',
                 clip_len=8,
                 frame_sample_rate=2,
                 crop_size=224,
                 short_side_size=256,
                 new_height=256,
                 new_width=340,
                 keep_aspect_ratio=True,
                 num_segment=1,
                 num_crop=1,
                 test_num_segment=10,
                 test_num_crop=3,
                 sparse_sample=False,
                 args=None):
        self.anno_path = anno_path
        self.data_root = data_root
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.sparse_sample = sparse_sample
        self.args = args
        self.aug = False
        self.rand_erase = False
        self.feature_cache = {}  # 用于缓存特征
        self.feature_extractor = None  # 延迟加载特征提取器
        self.device = "cpu"  # 默认使用CPU以避免CUDA问题
        self.keyframe_time = 0.0  # 使用浮点型而非整型
        self.keyframe_extraction_count = 0  # 新增：统计调用次数
        self.total_videos_processed = 0  # 新增：统计处理的视频数

        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        self.video_loader = get_video_loader()

        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        self.dataset_samples = list(
            cleaned[0].apply(lambda row: os.path.join(self.data_root, row)))
        self.label_array = list(cleaned.values[:, 1])

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(
                    self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(
                    size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(
                    size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    @staticmethod
    def _get_feature_extractor():
        """使用最新API加载特征提取器"""
        # 使用新API加载预训练模型
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        # 只保留特征提取部分
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        feature_extractor.eval()
        return feature_extractor

    def _ensure_extractor_loaded(self):
        """确保特征提取器已加载（延迟加载）"""
        if self.feature_extractor is None:
            self.feature_extractor = self._get_feature_extractor()
            # 固定在CPU上以避免CUDA多进程问题
            self.feature_extractor = self.feature_extractor.to(self.device)

    def _compute_attention_weights(self, features):
        """计算特征的自注意力权重"""
        # 计算特征之间的相似度矩阵(使用余弦相似度更稳定)
        features_norm = F.normalize(features, p=2, dim=1)
        similarity = torch.matmul(features_norm, features_norm.transpose(0, 1))

        # 应用softmax获取注意力权重
        attention_weights = F.softmax(similarity, dim=-1)

        # 计算全局注意力得分 (对每行求和)
        attention_scores = attention_weights.sum(dim=1)

        return attention_scores

    def _diversity_sampling(self, features, weights, max_frames):
        """基于多样性的帧采样"""
        if len(features) <= max_frames:
            return list(range(len(features)))

        selected_indices = []
        remaining_indices = list(range(len(features)))

        # 选择权重最高的帧作为第一帧
        first_idx = weights.argmax().item()
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # 迭代选择剩余帧
        while len(selected_indices) < max_frames and remaining_indices:
            best_idx = -1
            best_score = -float('inf')

            # 计算每个剩余帧与已选帧的最小距离
            for idx in remaining_indices:
                min_dist = float('inf')
                for sel_idx in selected_indices:
                    dist = torch.norm(features[idx] - features[sel_idx], p=2).item()
                    min_dist = min(min_dist, dist)

                # 结合注意力权重和多样性得分
                diversity_score = min_dist
                attention_score = weights[idx].item()
                # 平衡注意力与多样性
                combined_score = attention_score * 0.7 + diversity_score * 0.3

                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx

            if best_idx != -1:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break

        return sorted(selected_indices)

    def _diversity_sampling_multimodal(self, visual_features, motion_features, weights, max_frames):
        """多模态多样性帧采样"""
        if len(visual_features) <= max_frames:
            return list(range(len(visual_features)))

        selected_indices = []
        remaining_indices = list(range(len(visual_features)))

        # 选择权重最高的帧作为第一帧
        first_idx = weights.argmax().item()
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # 迭代选择剩余帧
        while len(selected_indices) < max_frames and remaining_indices:
            best_idx = -1
            best_score = -float('inf')

            # 计算每个剩余帧与已选帧的多模态距离
            for idx in remaining_indices:
                # 视觉特征距离
                min_visual_dist = float('inf')
                for sel_idx in selected_indices:
                    dist = torch.norm(visual_features[idx] - visual_features[sel_idx], p=2).item()
                    min_visual_dist = min(min_visual_dist, dist)

                # 运动特征差异
                motion_diff = abs(motion_features[idx].item() -
                                  sum(motion_features[sel_idx].item() for sel_idx in selected_indices) / len(selected_indices))

                # 结合注意力权重、视觉多样性和运动差异
                diversity_score = min_visual_dist * 0.7 + motion_diff * 0.3
                attention_score = weights[idx].item()

                # 平衡注意力与多样性 (调整权重以获得最佳结果)
                combined_score = attention_score * 0.6 + diversity_score * 0.4

                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx

            if best_idx != -1:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break

        return sorted(selected_indices)

    def detect_keyframes(self, vr, max_frames=10, threshold=0.15, mode='hmdb51_optimized'):
        """检测视频中的关键帧"""
        import time
        import os
        import fcntl
        
        self.keyframe_extraction_count += 1  # 增加调用计数
        process_id = os.getpid()
        
        # 创建统计目录
        stats_dir = os.path.join('/tmp', 'keyframe_stats')
        os.makedirs(stats_dir, exist_ok=True)
        stats_file = os.path.join(stats_dir, 'keyframe_stats.txt')
        
        # 打印日志用于调试
        print(f"开始关键帧提取 #{self.keyframe_extraction_count}, 模式: {mode}, 最大帧数: {max_frames}, 进程ID: {process_id}")
        start_time = time.time()
        
        length = len(vr)
        if length <= max_frames:
            elapsed = time.time() - start_time
            self.keyframe_time += elapsed
            print(f"视频长度 {length} 小于最大帧数 {max_frames}，耗时: {elapsed:.6f}秒")
            return list(range(length))

        # 使用注意力机制的动态关键帧选择
        if mode == 'attention_based':
            try:
                # 获取均匀采样的候选帧
                num_candidates = min(30, length)
                step = length / num_candidates
                candidate_indices = [min(length - 1, int(i * step)) for i in range(num_candidates)]

                # 获取候选帧
                frames = vr.get_batch(candidate_indices).asnumpy()

                # 使用CPU处理以避免CUDA问题
                self.device = "cpu"
                self._ensure_extractor_loaded()

                # 预处理帧并提取特征
                batch_frames = []
                for frame in frames:
                    # 预处理
                    img = cv2.resize(frame, (224, 224))
                    img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                    img = torch.unsqueeze(img, 0)
                    # 归一化
                    img = (img - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / \
                          torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                    batch_frames.append(img)

                with torch.no_grad():
                    # 批量提取特征
                    features = []
                    for img in batch_frames:
                        img = img.to(self.device)
                        feat = self.feature_extractor(img)
                        feat = feat.squeeze().cpu()
                        features.append(feat)

                    features = torch.stack(features)
                    features_flat = features.view(features.size(0), -1)

                    # 计算注意力权重
                    attention_weights = self._compute_attention_weights(features_flat)

                    # 基于注意力权重和多样性采样选择最终关键帧
                    selected_indices = self._diversity_sampling(
                        features_flat,
                        attention_weights,
                        max_frames
                    )

                    # 将选择的索引映射回原始帧索引
                    final_indices = [candidate_indices[idx] for idx in selected_indices]

                    elapsed = time.time() - start_time
                    self.keyframe_time += elapsed
                    print(f"关键帧提取完成 #{self.keyframe_extraction_count}, 模式: {mode}, 耗时: {elapsed:.6f}秒")
                    
                    # 写入统计信息到文件（使用文件锁确保并发安全）
                    try:
                        with open(stats_file, 'a+') as f:
                            fcntl.flock(f, fcntl.LOCK_EX)  # 获取排他锁
                            f.write(f"{process_id},{elapsed},{mode},1\n")  # 格式: 进程ID,耗时,模式,计数
                            fcntl.flock(f, fcntl.LOCK_UN)  # 释放锁
                    except Exception as e:
                        print(f"无法写入统计信息: {e}")
                    
                    return sorted(final_indices)
            except Exception as e:
                print(f"注意力关键帧检测失败，使用默认方法: {str(e)}")
                # 如果出错，回退到默认方法
                elapsed = time.time() - start_time
                self.keyframe_time += elapsed
                mode = 'hmdb51_optimized'

        elif mode == 'multimodal_attention':
            try:
                # 1. 获取均匀采样的候选帧和特征
                num_candidates = min(60, length)
                step = length / num_candidates
                candidate_indices = [min(length - 1, int(i * step)) for i in range(num_candidates)]
                frames = vr.get_batch(candidate_indices).asnumpy()

                # 2. 使用CPU处理
                self.device = "cpu"
                self._ensure_extractor_loaded()

                # 3. 提取多模态特征
                visual_features = []    # 视觉特征
                motion_features = []    # 运动特征
                temporal_weights = []   # 时序权重

                # 视觉特征提取
                batch_frames = []
                for frame in frames:
                    img = cv2.resize(frame, (224, 224))
                    img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                    img = torch.unsqueeze(img, 0)
                    img = (img - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / \
                          torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                    batch_frames.append(img)

                # 提取视觉特征
                with torch.no_grad():
                    for img in batch_frames:
                        img = img.to(self.device)
                        feat = self.feature_extractor(img)
                        feat = feat.squeeze().cpu()
                        visual_features.append(feat)

                    visual_features = torch.stack(visual_features)
                    visual_features_flat = visual_features.view(visual_features.size(0), -1)

                # 提取运动特征 (帧差异)
                for i in range(1, len(frames)):
                    curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                    prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)

                    # 计算光流
                    try:
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_gray, curr_gray, None,
                            0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        # 计算光流大小
                        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        motion_score = np.mean(magnitude)
                    except:
                        # 如果光流计算失败，使用帧差异
                        diff = np.abs(curr_gray.astype(float) - prev_gray.astype(float))
                        motion_score = np.mean(diff)

                    motion_features.append(motion_score)

                # 为第一帧添加运动特征
                motion_features.insert(0, motion_features[0] if motion_features else 0)
                motion_features = torch.tensor(motion_features)

                # 时序权重 - 关注视频中间部分
                video_len = len(candidate_indices)
                center = video_len // 2
                temporal_weights = [1.0 - 0.8 * abs(i - center) / max(center, 1) for i in range(video_len)]
                temporal_weights = torch.tensor(temporal_weights)

                # 4. 融合多模态信息
                # 计算视觉特征的注意力权重
                visual_attention = self._compute_attention_weights(visual_features_flat)

                # 归一化运动特征
                if motion_features.max() > motion_features.min():
                    motion_features = (motion_features - motion_features.min()) / (motion_features.max() - motion_features.min())

                # 融合注意力权重（视觉特征权重、运动特征权重和时序权重）
                # 根据实验调整权重
                fusion_weights = visual_attention * 0.5 + motion_features * 0.3 + temporal_weights * 0.2

                # 5. 进行多样性采样
                selected_indices = self._diversity_sampling_multimodal(
                    visual_features_flat,
                    motion_features,
                    fusion_weights,
                    max_frames
                )

                # 将选择的索引映射回原始帧
                final_indices = [candidate_indices[idx] for idx in selected_indices]

                elapsed = time.time() - start_time
                self.keyframe_time += elapsed
                print(f"关键帧提取完成 #{self.keyframe_extraction_count}, 模式: {mode}, 耗时: {elapsed:.6f}秒")
                
                # 写入统计信息到文件（使用文件锁确保并发安全）
                try:
                    with open(stats_file, 'a+') as f:
                        fcntl.flock(f, fcntl.LOCK_EX)  # 获取排他锁
                        f.write(f"{process_id},{elapsed},{mode},1\n")  # 格式: 进程ID,耗时,模式,计数
                        fcntl.flock(f, fcntl.LOCK_UN)  # 释放锁
                except Exception as e:
                    print(f"无法写入统计信息: {e}")
                
                return sorted(final_indices)
            except Exception as e:
                print(f"多模态注意力关键帧检测失败: {str(e)}")
                print("回退到普通注意力方法...")
                elapsed = time.time() - start_time
                self.keyframe_time += elapsed
                return self.detect_keyframes(vr, max_frames, threshold, 'attention_based')

        # 其他现有的关键帧提取方法保持不变
        elif mode == 'hmdb51_optimized':
            # 原有的实现...
            frame_diffs = []
            prev_frame = None

            sample_indices = np.linspace(0, length-1, min(100, length)).astype(int)
            frames = vr.get_batch(sample_indices).asnumpy()

            for i in range(1, len(frames)):
                curr_frame = frames[i]
                if prev_frame is not None:
                    curr_gray = np.mean(curr_frame, axis=2)
                    prev_gray = np.mean(prev_frame, axis=2)

                    diff = np.mean(np.abs(curr_gray - prev_gray))
                    frame_diffs.append((sample_indices[i], diff))
                prev_frame = curr_frame

            keyframe_indices = [0]
            sorted_diffs = sorted(frame_diffs, key=lambda x: x[1], reverse=True)

            for idx, diff in sorted_diffs:
                if diff >= threshold and idx not in keyframe_indices:
                    keyframe_indices.append(idx)
                if len(keyframe_indices) >= max_frames:
                    break

            if len(keyframe_indices) < max_frames:
                remaining_indices = [i for i in range(length) if i not in keyframe_indices]
                step = len(remaining_indices) // (max_frames - len(keyframe_indices))
                if step > 0:
                    additional_indices = remaining_indices[::step][:max_frames - len(keyframe_indices)]
                    keyframe_indices.extend(additional_indices)

            elapsed = time.time() - start_time
            self.keyframe_time += elapsed
            print(f"关键帧提取完成 #{self.keyframe_extraction_count}, 模式: {mode}, 耗时: {elapsed:.6f}秒")
            
            # 写入统计信息到文件（使用文件锁确保并发安全）
            try:
                with open(stats_file, 'a+') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)  # 获取排他锁
                    f.write(f"{process_id},{elapsed},{mode},1\n")  # 格式: 进程ID,耗时,模式,计数
                    fcntl.flock(f, fcntl.LOCK_UN)  # 释放锁
            except Exception as e:
                print(f"无法写入统计信息: {e}")
            
            return sorted(keyframe_indices[:max_frames])

        elif mode == 'optical_flow':
            sample_indices = np.linspace(0, length-1, min(100, length)).astype(int)
            frames = vr.get_batch(sample_indices).asnumpy()

            flow_magnitudes = []
            prev_frame_gray = None

            for i in range(len(frames)):
                curr_frame = frames[i]
                curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

                if prev_frame_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_frame_gray, curr_frame_gray, None,
                        0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    mean_magnitude = np.mean(magnitude)
                    flow_magnitudes.append((sample_indices[i], mean_magnitude))

                prev_frame_gray = curr_frame_gray

            flow_magnitudes.sort(key=lambda x: x[1], reverse=True)
            keyframe_indices = [0]

            for idx, _ in flow_magnitudes:
                if idx not in keyframe_indices:
                    keyframe_indices.append(idx)
                if len(keyframe_indices) >= max_frames:
                    break

            elapsed = time.time() - start_time
            self.keyframe_time += elapsed
            print(f"关键帧提取完成 #{self.keyframe_extraction_count}, 模式: {mode}, 耗时: {elapsed:.6f}秒")
            
            # 写入统计信息到文件（使用文件锁确保并发安全）
            try:
                with open(stats_file, 'a+') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)  # 获取排他锁
                    f.write(f"{process_id},{elapsed},{mode},1\n")  # 格式: 进程ID,耗时,模式,计数
                    fcntl.flock(f, fcntl.LOCK_UN)  # 释放锁
            except Exception as e:
                print(f"无法写入统计信息: {e}")
            
            return sorted(keyframe_indices[:max_frames])

        elif mode == 'scene_change':
            sample_indices = np.linspace(0, length-1, min(100, length)).astype(int)
            frames = vr.get_batch(sample_indices).asnumpy()

            scene_changes = []
            prev_hist = None

            for i in range(len(frames)):
                curr_frame = frames[i]
                hsv = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()

                if prev_hist is not None:
                    diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    scene_changes.append((sample_indices[i], 1 - diff))

                prev_hist = hist

            scene_changes.sort(key=lambda x: x[1], reverse=True)
            keyframe_indices = [0]

            for idx, diff in scene_changes:
                if diff > threshold and idx not in keyframe_indices:
                    keyframe_indices.append(idx)
                if len(keyframe_indices) >= max_frames:
                    break

            if len(keyframe_indices) < max_frames:
                remaining_indices = [i for i in range(length) if i not in keyframe_indices]
                step = len(remaining_indices) // (max_frames - len(keyframe_indices))
                if step > 0:
                    additional_indices = remaining_indices[::step][:max_frames - len(keyframe_indices)]
                    keyframe_indices.extend(additional_indices)

            elapsed = time.time() - start_time
            self.keyframe_time += elapsed
            print(f"关键帧提取完成 #{self.keyframe_extraction_count}, 模式: {mode}, 耗时: {elapsed:.6f}秒")
            
            # 写入统计信息到文件（使用文件锁确保并发安全）
            try:
                with open(stats_file, 'a+') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)  # 获取排他锁
                    f.write(f"{process_id},{elapsed},{mode},1\n")  # 格式: 进程ID,耗时,模式,计数
                    fcntl.flock(f, fcntl.LOCK_UN)  # 释放锁
            except Exception as e:
                print(f"无法写入统计信息: {e}")
            
            return sorted(keyframe_indices[:max_frames])

        elif mode == 'content_aware':
            sample_indices = np.linspace(0, length-1, min(100, length)).astype(int)
            frames = vr.get_batch(sample_indices).asnumpy()

            frame_scores = []
            prev_frame = None
            prev_gray = None

            for i in range(len(frames)):
                curr_frame = frames[i]
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

                clarity = cv2.Laplacian(curr_gray, cv2.CV_64F).var()
                brightness = np.mean(curr_gray)
                contrast = np.std(curr_gray)

                motion_score = 0
                if prev_gray is not None:
                    diff = np.abs(curr_gray - prev_gray)
                    motion_score = np.mean(diff)

                    if prev_frame is not None and i % 5 == 0:
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_gray, curr_gray, None,
                            0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        motion_score = max(motion_score, np.mean(magnitude))

                score = clarity * 0.4 + contrast * 0.3 + motion_score * 0.3
                frame_scores.append((sample_indices[i], score))

                prev_frame = curr_frame
                prev_gray = curr_gray

            frame_scores.sort(key=lambda x: x[1], reverse=True)
            keyframe_indices = [0]
            last_selected = 0
            min_interval = length // (max_frames * 2)

            for idx, _ in frame_scores:
                if idx not in keyframe_indices and (idx - last_selected) > min_interval:
                    keyframe_indices.append(idx)
                    last_selected = idx
                if len(keyframe_indices) >= max_frames:
                    break

            if len(keyframe_indices) < max_frames:
                remaining_indices = [i for i in range(length) if i not in keyframe_indices]
                step = len(remaining_indices) // (max_frames - len(keyframe_indices))
                if step > 0:
                    additional_indices = remaining_indices[::step][:max_frames - len(keyframe_indices)]
                    keyframe_indices.extend(additional_indices)

            elapsed = time.time() - start_time
            self.keyframe_time += elapsed
            print(f"关键帧提取完成 #{self.keyframe_extraction_count}, 模式: {mode}, 耗时: {elapsed:.6f}秒")
            
            # 写入统计信息到文件（使用文件锁确保并发安全）
            try:
                with open(stats_file, 'a+') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)  # 获取排他锁
                    f.write(f"{process_id},{elapsed},{mode},1\n")  # 格式: 进程ID,耗时,模式,计数
                    fcntl.flock(f, fcntl.LOCK_UN)  # 释放锁
            except Exception as e:
                print(f"无法写入统计信息: {e}")
            
            return sorted(keyframe_indices[:max_frames])

        elif mode == 'uniform':
            elapsed = time.time() - start_time
            self.keyframe_time += elapsed
            print(f"关键帧提取完成 #{self.keyframe_extraction_count}, 模式: {mode}, 耗时: {elapsed:.6f}秒")
            
            # 写入统计信息到文件（使用文件锁确保并发安全）
            try:
                with open(stats_file, 'a+') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)  # 获取排他锁
                    f.write(f"{process_id},{elapsed},{mode},1\n")  # 格式: 进程ID,耗时,模式,计数
                    fcntl.flock(f, fcntl.LOCK_UN)  # 释放锁
            except Exception as e:
                print(f"无法写入统计信息: {e}")
            
            return list(np.linspace(0, length-1, max_frames).astype(int))

        elif mode == 'saliency_map':
            sample_indices = np.linspace(0, length-1, min(100, length)).astype(int)
            frames = vr.get_batch(sample_indices).asnumpy()

            saliency_scores = []
            saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()

            for i in range(len(frames)):
                curr_frame = frames[i]
                curr_frame_bgr = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2BGR)

                success, saliency_map = saliency_detector.computeSaliency(curr_frame_bgr)
                if success:
                    score = np.mean(saliency_map)
                    saliency_scores.append((sample_indices[i], score))

            saliency_scores.sort(key=lambda x: x[1], reverse=True)
            keyframe_indices = [0]

            for idx, _ in saliency_scores:
                if idx not in keyframe_indices:
                    keyframe_indices.append(idx)
                if len(keyframe_indices) >= max_frames:
                    break

            elapsed = time.time() - start_time
            self.keyframe_time += elapsed
            print(f"关键帧提取完成 #{self.keyframe_extraction_count}, 模式: {mode}, 耗时: {elapsed:.6f}秒")
            
            # 写入统计信息到文件（使用文件锁确保并发安全）
            try:
                with open(stats_file, 'a+') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)  # 获取排他锁
                    f.write(f"{process_id},{elapsed},{mode},1\n")  # 格式: 进程ID,耗时,模式,计数
                    fcntl.flock(f, fcntl.LOCK_UN)  # 释放锁
            except Exception as e:
                print(f"无法写入统计信息: {e}")
            
            return sorted(keyframe_indices[:max_frames])

        else:
            elapsed = time.time() - start_time
            self.keyframe_time += elapsed
            print(f"关键帧提取完成 #{self.keyframe_extraction_count}, 模式: {mode}, 耗时: {elapsed:.6f}秒")
            
            # 写入统计信息到文件（使用文件锁确保并发安全）
            try:
                with open(stats_file, 'a+') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)  # 获取排他锁
                    f.write(f"{process_id},{elapsed},{mode},1\n")  # 格式: 进程ID,耗时,模式,计数
                    fcntl.flock(f, fcntl.LOCK_UN)  # 释放锁
            except Exception as e:
                print(f"无法写入统计信息: {e}")
            
            return list(np.linspace(0, length-1, max_frames).astype(int))
            
    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args
            scale_t = 1

            sample = self.dataset_samples[index]
            # T H W C
            buffer = self.load_video(sample, sample_rate_scale=scale_t)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(
                        "video {} not correctly loaded during training".format(
                            sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.load_video(sample, sample_rate_scale=scale_t)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)

            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer = self.load_video(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(
                        "video {} not correctly loaded during validation".
                        format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.load_video(sample)
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], sample.split(
                "/")[-1].split(".")[0]

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.load_video(sample)

            while len(buffer) == 0:
                warnings.warn(
                    "video {}, temporal {}, spatial {} not found during testing"
                    .format(str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.load_video(sample)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            if self.sparse_sample:
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) -
                                      self.short_side_size) / (
                                          self.test_num_crop - 1)
                temporal_start = chunk_nb
                spatial_start = int(split_nb * spatial_step)
                if buffer.shape[1] >= buffer.shape[2]:
                    buffer = buffer[temporal_start::self.test_num_segment,
                                    spatial_start:spatial_start +
                                    self.short_side_size, :, :]
                else:
                    buffer = buffer[temporal_start::self.test_num_segment, :,
                                    spatial_start:spatial_start +
                                    self.short_side_size, :]
            else:
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) -
                                      self.short_side_size) / (
                                          self.test_num_crop - 1)
                temporal_step = max(
                    1.0 * (buffer.shape[0] - self.clip_len) /
                    (self.test_num_segment - 1), 0)
                temporal_start = int(chunk_nb * temporal_step)
                spatial_start = int(split_nb * spatial_step)
                if buffer.shape[1] >= buffer.shape[2]:
                    buffer = buffer[temporal_start:temporal_start +
                                    self.clip_len,
                                    spatial_start:spatial_start +
                                    self.short_side_size, :, :]
                else:
                    buffer = buffer[temporal_start:temporal_start +
                                    self.clip_len, :,
                                    spatial_start:spatial_start +
                                    self.short_side_size, :]

            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample.split(
                "/")[-1].split(".")[0], chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(self, buffer, args):
        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        # T H W C
        buffer = tensor_normalize(buffer, [0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            # crop_size=224,
            crop_size=args.input_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False)

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)  # C T H W -> T C H W
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)  # T C H W -> C T H W

        return buffer

    def load_video(self, sample, sample_rate_scale=1):
        fname = sample

        try:
            vr = self.video_loader(fname)
        except Exception as e:
            print(f"Failed to load video from {fname} with error {e}!")
            return []

        length = len(vr)

        # 检查是否启用了关键帧模式
        use_keyframes = hasattr(self.args, 'keyframe_mode') and self.args.keyframe_mode

        if self.mode == 'test':
            # 测试模式下，如果启用了关键帧且在评估状态，使用关键帧
            if use_keyframes and self.args.eval:
                import os
                process_id = os.getpid()
                self.total_videos_processed += 1  # 记录处理的视频数
                print(f"处理视频 #{self.total_videos_processed}: {fname}, 进程ID: {process_id}")
                
                # 创建统计目录
                stats_dir = os.path.join('/tmp', 'keyframe_stats')
                os.makedirs(stats_dir, exist_ok=True)
                videos_file = os.path.join(stats_dir, 'videos_processed.txt')
                
                # 记录处理的视频数
                try:
                    import fcntl
                    with open(videos_file, 'a+') as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        f.write(f"{process_id},1\n")  # 进程ID,视频数
                        fcntl.flock(f, fcntl.LOCK_UN)
                except Exception as e:
                    print(f"无法写入视频统计信息: {e}")
                
                import time
                keyframe_start = time.time()
                all_index = self.detect_keyframes(
                    vr, 
                    max_frames=self.args.keyframe_max_frames,
                    threshold=self.args.keyframe_threshold,
                    mode=self.args.keyframe_mode
                )
                self.keyframe_time += time.time() - keyframe_start
                
                # 确保有足够的帧
                if len(all_index) < self.clip_len:
                    additional_indices = [i for i in range(length) if i not in all_index]
                    all_index.extend(additional_indices[:self.clip_len - len(all_index)])
                
                vr.seek(0)
                buffer = vr.get_batch(all_index).asnumpy()
                return buffer
            
            # 原始测试模式逻辑
            if self.sparse_sample:
                tick = length / float(self.num_segment)
                all_index = []
                for t_seg in range(self.test_num_segment):
                    tmp_index = [
                        int(t_seg * tick / self.test_num_segment + tick * x)
                        for x in range(self.num_segment)
                    ]
                    all_index.extend(tmp_index)
                all_index = list(np.sort(np.array(all_index)))
            else:
                all_index = [
                    x for x in range(0, length, self.frame_sample_rate)
                ]
                while len(all_index) < self.clip_len:
                    all_index.append(all_index[-1])

            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # 训练或验证模式下，如果启用了关键帧，也使用关键帧
        if use_keyframes:
            import time
            keyframe_start = time.time()
            all_index = self.detect_keyframes(
                vr, 
                max_frames=self.args.keyframe_max_frames,
                threshold=self.args.keyframe_threshold,
                mode=self.args.keyframe_mode
            )
            self.keyframe_time += time.time() - keyframe_start
            
            # 确保有足够的帧
            if len(all_index) < self.clip_len:
                additional_indices = [i for i in range(length) if i not in all_index]
                all_index.extend(additional_indices[:self.clip_len - len(all_index)])
            
            # 如果帧太多，取子集
            if len(all_index) > self.clip_len:
                interval = len(all_index) // self.clip_len
                all_index = all_index[::interval][:self.clip_len]
            
            all_index = all_index[::int(sample_rate_scale)]
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # 原始训练模式逻辑
        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = length // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(
                    0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate(
                    (index,
                     np.ones(self.clip_len - seg_len // self.frame_sample_rate)
                     * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                if self.mode == 'validation':
                    end_idx = (converted_len + seg_len) // 2
                else:
                    end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i * seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


class RawFrameClsDataset(Dataset):
    """Load your own raw frame classification dataset."""

    def __init__(self,
                 anno_path,
                 data_root,
                 mode='train',
                 clip_len=8,
                 crop_size=224,
                 short_side_size=256,
                 new_height=256,
                 new_width=340,
                 keep_aspect_ratio=True,
                 num_segment=1,
                 num_crop=1,
                 test_num_segment=10,
                 test_num_crop=3,
                 filename_tmpl='img_{:05}.jpg',
                 start_idx=1,
                 args=None):
        self.anno_path = anno_path
        self.data_root = data_root
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.filename_tmpl = filename_tmpl
        self.start_idx = start_idx
        self.args = args
        self.aug = False
        self.rand_erase = False

        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        self.image_loader = get_image_loader()

        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        self.dataset_samples = list(
            cleaned[0].apply(lambda row: os.path.join(self.data_root, row)))
        self.total_frames = list(cleaned.values[:, 1])
        self.label_array = list(cleaned.values[:, -1])

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(
                    self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(
                    size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(
                    size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_total_frames = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        self.test_seg.append((ck, cp))
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_total_frames.append(self.total_frames[idx])
                        self.test_label_array.append(self.label_array[idx])

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args
            scale_t = 1

            sample = self.dataset_samples[index]
            total_frame = self.total_frames[index]
            buffer = self.load_frame(
                sample, total_frame, sample_rate_scale=scale_t)  # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(
                        "video {} not correctly loaded during training".format(
                            sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    total_frame = self.total_frames[index]
                    buffer = self.load_frame(
                        sample, total_frame, sample_rate_scale=scale_t)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)

            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            total_frame = self.total_frames[index]
            buffer = self.load_frame(sample, total_frame)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(
                        "video {} not correctly loaded during validation".
                        format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.load_frame(sample, total_frame)
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], sample.split(
                "/")[-1].split(".")[0]

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            total_frame = self.test_total_frames[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.load_frame(sample, total_frame)

            while len(buffer) == 0:
                warnings.warn(
                    "video {}, temporal {}, spatial {} not found during testing"
                    .format(str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                total_frame = self.test_total_frames[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.load_frame(sample, total_frame)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) -
                                  self.short_side_size) / (
                                      self.test_num_crop - 1)
            temporal_start = chunk_nb
            spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start::self.test_num_segment,
                                spatial_start:spatial_start +
                                self.short_side_size, :, :]
            else:
                buffer = buffer[temporal_start::self.test_num_segment, :,
                                spatial_start:spatial_start +
                                self.short_side_size, :]

            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample.split(
                "/")[-1].split(".")[0], chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(self, buffer, args):
        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        # T H W C
        buffer = tensor_normalize(buffer, [0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False)

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def load_frame(self, sample, num_frames, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if self.mode == 'test':
            tick = num_frames / float(self.num_segment)
            all_index = []
            for t_seg in range(self.test_num_segment):
                tmp_index = [
                    int(t_seg * tick / self.test_num_segment + tick * x)
                    for x in range(self.num_segment)
                ]
                all_index.extend(tmp_index)
            all_index = list(np.sort(np.array(all_index) + self.start_idx))
            imgs = []
            for idx in all_index:
                frame_fname = os.path.join(fname,
                                           self.filename_tmpl.format(idx))
                img = self.image_loader(frame_fname)
                imgs.append(img)
            buffer = np.array(imgs)
            return buffer

        # handle temporal segments
        average_duration = num_frames // self.num_segment
        all_index = []
        if average_duration > 0:
            if self.mode == 'validation':
                all_index = list(
                    np.multiply(
                        list(range(self.num_segment)), average_duration) +
                    np.ones(self.num_segment, dtype=int) *
                    (average_duration // 2))
            else:
                all_index = list(
                    np.multiply(
                        list(range(self.num_segment)), average_duration) +
                    np.random.randint(average_duration, size=self.num_segment))
        elif num_frames > self.num_segment:
            if self.mode == 'validation':
                all_index = list(range(self.num_segment))
            else:
                all_index = list(
                    np.sort(
                        np.random.randint(num_frames, size=self.num_segment)))
        else:
            all_index = [0] * (self.num_segment - num_frames) + list(
                range(num_frames))
        all_index = list(np.array(all_index) + self.start_idx)
        imgs = []
        for idx in all_index:
            frame_fname = os.path.join(fname, self.filename_tmpl.format(idx))
            img = self.image_loader(frame_fname)
            imgs.append(img)
        buffer = np.array(imgs)
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift else video_transforms.random_resized_crop)
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale)
        frames, _ = video_transforms.uniform_crop(frames, crop_size,
                                                  spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor
