#!/usr/bin/env python3
"""
优化的1956 TI数据集模型训练脚本
基于108个高质量手动标注优化训练参数
"""

import os
import sys
from pathlib import Path
import yaml
import json
import logging
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedModelTrainer:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.models_dir = self.dataset_dir / "models"
        self.results_dir = self.dataset_dir / "results"
        
        # 创建目录
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # 配置文件路径
        self.config_file = self.dataset_dir / "dataset_config.yaml"
    
    def train_optimized_model(self, model_size='s', epochs=150, batch_size=8):
        """使用优化参数训练模型"""
        logger.info(f"开始优化训练 YOLOv8{model_size} 模型...")
        logger.info(f"数据集: 108张图像, 197个标注")
        
        # 检查GPU可用性
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        logger.info(f"使用设备: {device}")
        
        # 初始化模型
        model_name = f'yolov8{model_size}.pt'
        model = YOLO(model_name)
        
        logger.info(f"使用预训练模型: {model_name}")
        logger.info(f"训练参数: epochs={epochs}, batch_size={batch_size}")
        
        # 优化的训练参数
        try:
            results = model.train(
                # 数据配置
                data=str(self.config_file),
                epochs=epochs,
                batch=batch_size,
                imgsz=640,
                
                # 优化器配置
                optimizer='AdamW',
                lr0=0.002,           # 降低初始学习率
                lrf=0.01,            # 最终学习率因子
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=5,     # 增加预热轮数
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                
                # 损失函数权重
                box=7.5,
                cls=0.5,
                dfl=1.5,
                
                # 数据增强（适中设置）
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=0.0,         # 不旋转（文档图像）
                translate=0.1,
                scale=0.5,
                shear=0.0,           # 不剪切（文档图像）
                perspective=0.0,     # 不透视变换
                flipud=0.0,          # 不上下翻转
                fliplr=0.5,          # 50%左右翻转
                mosaic=1.0,          # 启用mosaic增强
                mixup=0.1,           # 适度mixup
                copy_paste=0.1,      # 适度copy-paste
                
                # 训练配置
                patience=50,         # 早停耐心
                save=True,
                save_period=-1,
                cache=False,
                device=device,
                workers=4,           # 减少worker数量
                project=str(self.results_dir),
                name=f'1956_TI_optimized_yolov8{model_size}',
                exist_ok=True,
                pretrained=True,
                verbose=True,
                seed=42,             # 固定随机种子
                deterministic=True,
                single_cls=False,
                rect=False,
                cos_lr=True,         # 启用余弦学习率调度
                close_mosaic=15,     # 最后15轮关闭mosaic
                resume=False,
                amp=True,            # 自动混合精度
                fraction=1.0,
                profile=False,
                freeze=None,
                multi_scale=True,    # 多尺度训练
                overlap_mask=True,
                mask_ratio=4,
                dropout=0.0,
                val=True,
                plots=True,
            )
            
            logger.info("优化训练完成!")
            
            # 保存最佳模型
            best_model_src = self.results_dir / f'1956_TI_optimized_yolov8{model_size}' / 'weights' / 'best.pt'
            best_model_dst = self.models_dir / f'optimized_best_yolov8{model_size}.pt'
            
            if best_model_src.exists():
                import shutil
                shutil.copy2(best_model_src, best_model_dst)
                logger.info(f"优化模型已保存到: {best_model_dst}")
            
            return results, best_model_dst
            
        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            raise

def main():
    """主函数"""
    dataset_dir = "/Users/zhaoye/Desktop/1956_TI_Dataset"
    
    # 检查数据集配置
    config_file = Path(dataset_dir) / "dataset_config.yaml"
    if not config_file.exists():
        logger.error("数据集配置文件不存在")
        return
    
    # 创建优化训练器
    trainer = OptimizedModelTrainer(dataset_dir)
    
    # 优化的训练参数
    model_size = 's'      # 使用稍大的模型
    epochs = 150          # 增加训练轮数
    batch_size = 8        # 适中的批次大小
    
    try:
        logger.info("开始优化训练流程...")
        logger.info(f"数据集规模: 108张图像, 197个标注")
        logger.info(f"训练配置: YOLOv8{model_size}, {epochs} epochs, batch_size={batch_size}")
        
        # 训练模型
        results, model_path = trainer.train_optimized_model(
            model_size=model_size,
            epochs=epochs,
            batch_size=batch_size
        )
        
        print("\n=== 优化训练完成 ===")
        print(f"最佳模型: {model_path}")
        print(f"训练结果目录: {trainer.results_dir}")
        
        # 显示关键指标
        results_csv = trainer.results_dir / f'1956_TI_optimized_yolov8{model_size}' / 'results.csv'
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            final_map50 = df['metrics/mAP50(B)'].iloc[-1]
            final_map50_95 = df['metrics/mAP50-95(B)'].iloc[-1]
            print(f"最终 mAP50: {final_map50:.4f}")
            print(f"最终 mAP50-95: {final_map50_95:.4f}")
        
    except Exception as e:
        logger.error(f"优化训练失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
