#!/usr/bin/env python3
"""
创建示例标注数据用于演示Pipeline
这个脚本会为前几张图像创建示例标注，展示完整流程
"""

import json
import os
from pathlib import Path
from PIL import Image
import random

def create_sample_annotations(dataset_dir, num_samples=10):
    """为前几张图像创建示例标注"""
    dataset_dir = Path(dataset_dir)
    raw_images_dir = dataset_dir / "raw_images"
    annotations_dir = dataset_dir / "annotations"
    
    # 确保标注目录存在
    annotations_dir.mkdir(exist_ok=True)
    
    # 获取图像文件列表
    image_files = list(raw_images_dir.glob("*.png"))
    image_files.sort()
    
    print(f"找到 {len(image_files)} 张图像，将为前 {num_samples} 张创建示例标注")
    
    # 为前几张图像创建示例标注
    for i, image_file in enumerate(image_files[:num_samples]):
        try:
            # 加载图像获取尺寸
            with Image.open(image_file) as img:
                width, height = img.size
            
            # 创建示例边界框（模拟图像区域）
            # 通常图像区域位于文档的中央或特定位置
            num_boxes = random.randint(1, 3)  # 每张图1-3个图像区域
            bboxes = []
            
            for j in range(num_boxes):
                # 创建随机但合理的边界框
                # 确保不会太小，也不会覆盖整个图像
                min_size = min(width, height) // 10  # 最小尺寸
                max_width = width * 0.8
                max_height = height * 0.8
                
                # 随机生成边界框
                box_width = random.randint(min_size, int(max_width))
                box_height = random.randint(min_size, int(max_height))
                
                x1 = random.randint(0, width - box_width)
                y1 = random.randint(0, height - box_height)
                x2 = x1 + box_width
                y2 = y1 + box_height
                
                bbox = {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'label': 'image',
                    'area': box_width * box_height
                }
                bboxes.append(bbox)
            
            # 创建标注数据
            annotation_data = {
                'image_path': str(image_file),
                'image_name': image_file.name,
                'image_size': {'width': width, 'height': height},
                'bboxes': bboxes,
                'num_annotations': len(bboxes)
            }
            
            # 保存标注文件
            annotation_file = annotations_dir / f"{image_file.stem}.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 创建标注: {image_file.name} -> {len(bboxes)} 个区域")
            
        except Exception as e:
            print(f"❌ 处理 {image_file.name} 失败: {e}")
    
    print(f"\n✨ 示例标注创建完成！创建了 {num_samples} 个标注文件")
    print(f"📁 标注文件位置: {annotations_dir}")

if __name__ == "__main__":
    dataset_dir = "/Users/zhaoye/Desktop/1956_TI_Dataset"
    create_sample_annotations(dataset_dir, num_samples=20)  # 创建20个示例标注

