#!/usr/bin/env python3
"""
优化的模型训练脚本
基于108张手动标注图像的高质量数据集
"""

import os
import sys
from pathlib import Path
import json
import logging
import torch
from ultralytics import YOLO
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedTrainer:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.models_dir = self.dataset_dir / "models"
        self.results_dir = self.dataset_dir / "results"
        self.config_file = self.dataset_dir / "dataset_config.yaml"
        
        # 创建目录
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
    
    def get_optimized_training_params(self):
        """获取针对1956 TI数据集优化的训练参数"""
        return {
            # 基础参数
            'data': str(self.config_file),
            'epochs': 150,           # 增加训练轮数
            'batch': 8,              # 适中的批次大小
            'imgsz': 640,            # 标准图像大小
            
            # 学习率策略
            'lr0': 0.001,            # 降低初始学习率
            'lrf': 0.01,             # 最终学习率因子
            'momentum': 0.937,       # 动量
            'weight_decay': 0.0005,  # 权重衰减
            
            # 优化器
            'optimizer': 'AdamW',    # 使用AdamW优化器
            
            # 数据增强（针对文档图像优化）
            'hsv_h': 0.005,          # 色调变化（文档图像色调变化小）
            'hsv_s': 0.3,            # 饱和度变化
            'hsv_v': 0.2,            # 亮度变化
            'degrees': 2.0,          # 旋转角度（文档通常不大幅旋转）
            'translate': 0.05,       # 平移（小幅平移）
            'scale': 0.3,            # 缩放
            'shear': 1.0,            # 剪切
            'perspective': 0.0001,   # 透视变换（文档图像透视变化小）
            'flipud': 0.0,           # 垂直翻转（文档不翻转）
            'fliplr': 0.2,           # 水平翻转（少量翻转）
            'mosaic': 0.8,           # Mosaic增强
            'mixup': 0.1,            # Mixup增强
            'copy_paste': 0.1,       # Copy-paste增强
            
            # 损失函数权重
            'box': 7.5,              # 边界框损失权重
            'cls': 0.5,              # 分类损失权重
            'dfl': 1.5,              # 分布焦点损失权重
            
            # 训练策略
            'patience': 30,          # 早停耐心值
            'save_period': 10,       # 每10轮保存一次
            'val': True,             # 启用验证
            'plots': True,           # 生成训练图表
            'deterministic': True,   # 确定性训练
            'single_cls': False,     # 多类别（虽然我们只有一个类别）
            'rect': False,           # 矩形训练
            'cos_lr': True,          # 余弦学习率调度
            'close_mosaic': 20,      # 关闭mosaic的轮数
            'amp': True,             # 混合精度训练
            
            # GPU设置（Mac MPS支持）
            'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
            
            # 输出设置
            'project': str(self.results_dir),
            'name': 'optimized_1956_TI',
            'exist_ok': True,
            'save': True,
            'verbose': True,
        }
    
    def train_optimized_model(self, model_size='s'):
        """使用优化参数训练模型"""
        logger.info(f"开始优化训练 YOLOv8{model_size} 模型...")
        logger.info(f"数据集: 108张图像, 197个标注")
        
        # 获取训练参数
        train_params = self.get_optimized_training_params()
        
        # 保存训练配置
        config_save_path = self.models_dir / f"optimized_train_config_yolov8{model_size}.json"
        with open(config_save_path, 'w', encoding='utf-8') as f:
            # 转换Path对象为字符串以便JSON序列化
            json_params = {k: str(v) if isinstance(v, Path) else v for k, v in train_params.items()}
            json.dump(json_params, f, ensure_ascii=False, indent=2)
        
        # 初始化模型
        model_name = f'yolov8{model_size}.pt'
        model = YOLO(model_name)
        
        logger.info(f"使用预训练模型: {model_name}")
        logger.info(f"训练参数已保存到: {config_save_path}")
        
        # 开始训练
        try:
            results = model.train(**train_params)
            
            logger.info("优化训练完成!")
            
            # 保存最佳模型
            best_model_src = self.results_dir / 'optimized_1956_TI' / 'weights' / 'best.pt'
            best_model_dst = self.models_dir / f'optimized_best_yolov8{model_size}.pt'
            
            if best_model_src.exists():
                import shutil
                shutil.copy2(best_model_src, best_model_dst)
                logger.info(f"优化模型已保存到: {best_model_dst}")
            
            return results, best_model_dst
            
        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            raise
    
    def validate_model(self, model_path):
        """验证模型性能"""
        logger.info(f"验证模型: {model_path}")
        
        model = YOLO(model_path)
        
        # 在测试集上验证
        results = model.val(
            data=str(self.config_file),
            split='test',
            save_json=True,
            conf=0.1,  # 使用较低的置信度阈值
            iou=0.6,
            plots=True
        )
        
        logger.info("模型验证完成")
        return results

def main():
    """主训练函数"""
    dataset_dir = "/Users/zhaoye/Desktop/1956_TI_Dataset"
    
    # 检查数据集是否准备就绪
    config_file = Path(dataset_dir) / "dataset_config.yaml"
    if not config_file.exists():
        logger.error("数据集配置文件不存在，请先运行 dataset_processor.py")
        return
    
    # 创建优化训练器
    trainer = OptimizedTrainer(dataset_dir)
    
    # 选择模型大小（s比n更大，适合我们的数据量）
    model_size = 's'  # 使用YOLOv8s，比n版本更强大
    
    try:
        logger.info("=" * 50)
        logger.info("开始优化训练流程...")
        logger.info("=" * 50)
        
        # 训练优化模型
        results, model_path = trainer.train_optimized_model(model_size=model_size)
        
        # 验证模型
        val_results = trainer.validate_model(model_path)
        
        print("\n" + "=" * 50)
        print("🎉 优化训练完成!")
        print("=" * 50)
        print(f"📁 最佳模型: {model_path}")
        print(f"📊 模型大小: {model_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"📈 训练结果: {trainer.results_dir / 'optimized_1956_TI'}")
        
        # 显示性能指标
        if hasattr(val_results, 'box'):
            print(f"🎯 mAP50: {val_results.box.map50:.4f}")
            print(f"🎯 mAP50-95: {val_results.box.map:.4f}")
            print(f"🎯 Precision: {val_results.box.mp:.4f}")
            print(f"🎯 Recall: {val_results.box.mr:.4f}")
        
        print("\n📋 下一步:")
        print("1. 检查训练结果图表")
        print("2. 运行推理测试")
        print("3. 调整推理参数（如果需要）")
        
    except Exception as e:
        logger.error(f"优化训练失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
