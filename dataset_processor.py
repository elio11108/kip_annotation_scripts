#!/usr/bin/env python3
"""
1956 TI 数据集处理脚本
将标注数据转换为训练格式，创建训练/验证/测试集
"""

import json
import os
import shutil
from pathlib import Path
import random
from PIL import Image
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetProcessor:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.raw_images_dir = self.dataset_dir / "raw_images"
        self.annotations_dir = self.dataset_dir / "annotations"
        self.processed_dir = self.dataset_dir / "processed_data"
        
        # 创建处理后的数据目录
        for split in ['train', 'val', 'test']:
            (self.processed_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.processed_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    def clean_split_dirs(self):
        """清空旧的分割结果，防止多次运行时图像跨分割累积（见 README 的 Data-split note）"""
        for split in ['train', 'val', 'test']:
            for sub in ['images', 'labels']:
                split_sub_dir = self.processed_dir / split / sub
                if split_sub_dir.exists():
                    shutil.rmtree(split_sub_dir)
                split_sub_dir.mkdir(parents=True, exist_ok=True)
    
    def load_annotations(self):
        """加载所有标注文件"""
        annotations = []
        annotation_files = list(self.annotations_dir.glob("*.json"))
        
        logger.info(f"找到 {len(annotation_files)} 个标注文件")
        
        for annotation_file in annotation_files:
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 验证图像文件是否存在
                image_path = self.raw_images_dir / data['image_name']
                if image_path.exists() and data['bboxes']:
                    annotations.append(data)
                else:
                    logger.warning(f"跳过 {annotation_file}: 图像不存在或无标注")
                    
            except Exception as e:
                logger.error(f"读取标注文件 {annotation_file} 失败: {e}")
                
        logger.info(f"成功加载 {len(annotations)} 个有效标注")
        return annotations
    
    def convert_to_yolo_format(self, bbox, image_width, image_height):
        """将边界框转换为YOLO格式 (归一化的中心点坐标和宽高)"""
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        # 计算中心点坐标和宽高
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # 归一化
        center_x /= image_width
        center_y /= image_height
        width /= image_width
        height /= image_height
        
        return center_x, center_y, width, height
    
    def split_dataset(self, annotations, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """分割数据集"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "分割比例之和必须等于1"

        # 随机打乱（固定种子，保证分割可复现；见 README 的 Data-split note）
        random.seed(42)
        random.shuffle(annotations)
        
        # 计算分割点
        total = len(annotations)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_data = annotations[:train_end]
        val_data = annotations[train_end:val_end]
        test_data = annotations[val_end:]
        
        logger.info(f"数据集分割: 训练集 {len(train_data)}, 验证集 {len(val_data)}, 测试集 {len(test_data)}")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def process_split(self, split_name, annotations):
        """处理单个数据集分割"""
        split_dir = self.processed_dir / split_name
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        logger.info(f"处理 {split_name} 集: {len(annotations)} 个样本")
        
        for i, annotation in enumerate(annotations):
            try:
                # 复制图像文件
                source_image = self.raw_images_dir / annotation['image_name']
                dest_image = images_dir / annotation['image_name']
                shutil.copy2(source_image, dest_image)
                
                # 创建YOLO格式的标签文件
                label_file = labels_dir / f"{Path(annotation['image_name']).stem}.txt"
                
                with open(label_file, 'w') as f:
                    for bbox in annotation['bboxes']:
                        # 转换为YOLO格式
                        center_x, center_y, width, height = self.convert_to_yolo_format(
                            bbox, 
                            annotation['image_size']['width'],
                            annotation['image_size']['height']
                        )
                        
                        # YOLO格式: class_id center_x center_y width height
                        # 我们只有一个类别(图像)，所以class_id = 0
                        f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                        
            except Exception as e:
                logger.error(f"处理 {annotation['image_name']} 失败: {e}")
    
    def create_yaml_config(self):
        """创建YOLO训练配置文件"""
        config = {
            'path': str(self.processed_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,  # 类别数量
            'names': ['image']  # 类别名称
        }
        
        config_file = self.dataset_dir / 'dataset_config.yaml'
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"YOLO配置文件已保存到: {config_file}")
        return config_file
    
    def generate_statistics(self, dataset_splits):
        """生成数据集统计信息"""
        stats = {
            'total_images': sum(len(split) for split in dataset_splits.values()),
            'total_annotations': 0,
            'splits': {},
            'bbox_statistics': {
                'areas': [],
                'aspect_ratios': [],
                'widths': [],
                'heights': []
            }
        }
        
        for split_name, annotations in dataset_splits.items():
            split_stats = {
                'num_images': len(annotations),
                'num_annotations': sum(len(ann['bboxes']) for ann in annotations),
                'avg_annotations_per_image': 0
            }
            
            if split_stats['num_images'] > 0:
                split_stats['avg_annotations_per_image'] = split_stats['num_annotations'] / split_stats['num_images']
            
            stats['splits'][split_name] = split_stats
            stats['total_annotations'] += split_stats['num_annotations']
            
            # 收集边界框统计信息
            for annotation in annotations:
                for bbox in annotation['bboxes']:
                    width = bbox['x2'] - bbox['x1']
                    height = bbox['y2'] - bbox['y1']
                    area = width * height
                    aspect_ratio = width / height if height > 0 else 0
                    
                    stats['bbox_statistics']['areas'].append(area)
                    stats['bbox_statistics']['aspect_ratios'].append(aspect_ratio)
                    stats['bbox_statistics']['widths'].append(width)
                    stats['bbox_statistics']['heights'].append(height)
        
        # 计算统计量
        for key in ['areas', 'aspect_ratios', 'widths', 'heights']:
            values = stats['bbox_statistics'][key]
            if values:
                stats['bbox_statistics'][key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        # 保存统计信息
        stats_file = self.dataset_dir / 'dataset_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据集统计信息已保存到: {stats_file}")
        return stats
    
    def process_dataset(self):
        """处理整个数据集"""
        logger.info("开始处理数据集...")
        
        # 加载标注
        annotations = self.load_annotations()
        if not annotations:
            logger.error("没有找到有效的标注数据")
            return
        
        # 分割数据集
        dataset_splits = self.split_dataset(annotations)

        # 清空旧的分割目录后再写入
        self.clean_split_dirs()

        # 处理各个分割
        for split_name, split_annotations in dataset_splits.items():
            self.process_split(split_name, split_annotations)
        
        # 创建配置文件
        config_file = self.create_yaml_config()
        
        # 生成统计信息
        stats = self.generate_statistics(dataset_splits)
        
        logger.info("数据集处理完成!")
        logger.info(f"总计: {stats['total_images']} 张图像, {stats['total_annotations']} 个标注")
        
        return config_file, stats

def main():
    dataset_dir = str(Path(__file__).resolve().parent)
    processor = DatasetProcessor(dataset_dir)
    
    try:
        config_file, stats = processor.process_dataset()
        
        print("\n=== 数据集处理完成 ===")
        print(f"配置文件: {config_file}")
        print(f"训练集: {stats['splits']['train']['num_images']} 张图像")
        print(f"验证集: {stats['splits']['val']['num_images']} 张图像")
        print(f"测试集: {stats['splits']['test']['num_images']} 张图像")
        print(f"总标注数: {stats['total_annotations']}")
        
    except Exception as e:
        logger.error(f"处理失败: {e}")

if __name__ == "__main__":
    main()

