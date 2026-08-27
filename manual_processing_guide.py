#!/usr/bin/env python3
"""
手动数据集处理指南和工具
展示如何手动完成数据集处理的每个步骤
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image
import random

def manual_step1_convert_annotations():
    """
    步骤1: 手动转换标注格式
    从JSON格式转换为YOLO格式
    """
    print("=== 步骤1: 手动转换标注格式 ===")
    
    annotations_dir = Path("/Users/zhaoye/Desktop/1956_TI_Dataset/annotations")
    
    # 读取一个标注文件示例
    sample_file = list(annotations_dir.glob("*.json"))[0]
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始JSON格式 ({sample_file.name}):")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    
    # 转换为YOLO格式
    image_width = data['image_size']['width']
    image_height = data['image_size']['height']
    
    print(f"\n转换为YOLO格式:")
    for i, bbox in enumerate(data['bboxes']):
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        # 计算中心点和宽高
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # 归一化 (0-1范围)
        center_x_norm = center_x / image_width
        center_y_norm = center_y / image_height
        width_norm = width / image_width
        height_norm = height / image_height
        
        print(f"边界框{i+1}: 原始({x1},{y1},{x2},{y2}) -> YOLO(0 {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f})")

def manual_step2_split_dataset():
    """
    步骤2: 手动分割数据集
    """
    print("\n=== 步骤2: 手动分割数据集 ===")
    
    annotations_dir = Path("/Users/zhaoye/Desktop/1956_TI_Dataset/annotations")
    annotation_files = list(annotations_dir.glob("*.json"))
    
    print(f"总标注文件: {len(annotation_files)}")
    
    # 随机打乱
    random.shuffle(annotation_files)
    
    # 计算分割点
    total = len(annotation_files)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.2)
    
    train_files = annotation_files[:train_end]
    val_files = annotation_files[train_end:val_end]
    test_files = annotation_files[val_end:]
    
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    print(f"测试集: {len(test_files)} 个文件")
    
    return {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

def manual_step3_create_directories():
    """
    步骤3: 手动创建目录结构
    """
    print("\n=== 步骤3: 手动创建目录结构 ===")
    
    base_dir = Path("/Users/zhaoye/Desktop/1956_TI_Dataset/processed_data")
    
    # 创建目录
    for split in ['train', 'val', 'test']:
        images_dir = base_dir / split / 'images'
        labels_dir = base_dir / split / 'labels'
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"创建目录: {images_dir}")
        print(f"创建目录: {labels_dir}")

def manual_step4_copy_and_convert():
    """
    步骤4: 手动复制图像和转换标签
    """
    print("\n=== 步骤4: 手动复制图像和转换标签 ===")
    
    # 这里展示手动处理一个文件的完整流程
    sample_annotation = "/Users/zhaoye/Desktop/1956_TI_Dataset/annotations/1956_Issue-1_12_page_0_fig0_fig1.json"
    
    with open(sample_annotation, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_name = data['image_name']
    
    print(f"处理文件: {image_name}")
    
    # 1. 复制图像文件
    source_image = f"/Users/zhaoye/Desktop/1956_TI_Dataset/raw_images/{image_name}"
    dest_image = f"/Users/zhaoye/Desktop/1956_TI_Dataset/processed_data/train/images/{image_name}"
    
    print(f"复制图像: {source_image} -> {dest_image}")
    
    # 2. 创建YOLO标签文件
    label_file = f"/Users/zhaoye/Desktop/1956_TI_Dataset/processed_data/train/labels/{Path(image_name).stem}.txt"
    
    print(f"创建标签: {label_file}")
    
    # 转换边界框
    image_width = data['image_size']['width']
    image_height = data['image_size']['height']
    
    yolo_lines = []
    for bbox in data['bboxes']:
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        center_x = (x1 + x2) / 2.0 / image_width
        center_y = (y1 + y2) / 2.0 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        
        # YOLO格式: class_id center_x center_y width height
        yolo_line = f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)
        print(f"  YOLO行: {yolo_line}")
    
    print(f"标签内容:\n" + "\n".join(yolo_lines))

def diagnose_training_issues():
    """
    诊断训练问题
    """
    print("\n=== 训练问题诊断 ===")
    
    # 检查数据集大小
    train_dir = Path("/Users/zhaoye/Desktop/1956_TI_Dataset/processed_data/train/images")
    train_images = list(train_dir.glob("*.png"))
    
    print(f"训练图像数量: {len(train_images)}")
    
    if len(train_images) < 100:
        print("⚠️  警告: 训练数据太少！建议至少100张图像")
        print("   解决方案:")
        print("   1. 标注更多图像")
        print("   2. 使用数据增强")
        print("   3. 降低置信度阈值")
    
    # 检查标注质量
    labels_dir = Path("/Users/zhaoye/Desktop/1956_TI_Dataset/processed_data/train/labels")
    label_files = list(labels_dir.glob("*.txt"))
    
    total_annotations = 0
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            total_annotations += len(lines)
    
    print(f"总标注数量: {total_annotations}")
    print(f"平均每张图标注数: {total_annotations/len(train_images):.2f}")
    
    if total_annotations < 50:
        print("⚠️  警告: 标注数量太少！")

def suggest_improvements():
    """
    改进建议
    """
    print("\n=== 改进建议 ===")
    
    print("1. 增加训练数据:")
    print("   - 标注更多图像 (建议至少100-200张)")
    print("   - 确保标注质量和准确性")
    
    print("\n2. 调整训练参数:")
    print("   - 增加训练轮数 (epochs=200)")
    print("   - 减小学习率 (lr0=0.001)")
    print("   - 使用更大的模型 (yolov8s 或 yolov8m)")
    
    print("\n3. 调整推理参数:")
    print("   - 降低置信度阈值 (conf=0.1)")
    print("   - 调整IoU阈值")
    print("   - 减小最小区域面积")
    
    print("\n4. 数据增强:")
    print("   - 启用更多数据增强选项")
    print("   - 使用mixup和cutmix")

def create_improved_inference_script():
    """
    创建改进的推理脚本
    """
    print("\n=== 创建改进的推理脚本 ===")
    
    script_content = '''
# 改进的推理参数
inferencer.conf_threshold = 0.1  # 降低置信度阈值
inferencer.iou_threshold = 0.3   # 降低IoU阈值

# 在 extract_image_regions 函数中
min_area = 500  # 降低最小面积要求
'''
    
    print("建议的参数调整:")
    print(script_content)

def main():
    """主函数 - 运行所有手动处理步骤的演示"""
    print("1956 TI 数据集手动处理指南")
    print("=" * 50)
    
    # 步骤1: 转换标注格式
    manual_step1_convert_annotations()
    
    # 步骤2: 分割数据集
    splits = manual_step2_split_dataset()
    
    # 步骤3: 创建目录
    manual_step3_create_directories()
    
    # 步骤4: 复制和转换
    manual_step4_copy_and_convert()
    
    # 诊断问题
    diagnose_training_issues()
    
    # 改进建议
    suggest_improvements()
    
    # 改进脚本
    create_improved_inference_script()

if __name__ == "__main__":
    main()
