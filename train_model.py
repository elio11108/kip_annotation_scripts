#!/usr/bin/env python3
"""
1956 TI 数据集模型训练脚本
使用YOLOv8训练自定义的图像检测模型
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
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.models_dir = self.dataset_dir / "models"
        self.results_dir = self.dataset_dir / "results"
        
        # 创建目录
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # 配置文件路径
        self.config_file = self.dataset_dir / "dataset_config.yaml"
        
    def load_dataset_config(self):
        """加载数据集配置"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"数据集配置文件不存在: {self.config_file}")
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"加载数据集配置: {config}")
        return config
    
    def create_training_config(self, model_size='n', epochs=100, batch_size=16, img_size=640):
        """创建训练配置"""
        config = {
            'model_size': model_size,
            'epochs': epochs,
            'batch_size': batch_size,
            'img_size': img_size,
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'lrf': 0.1,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 2.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'save': True,
            'save_period': -1,
            'cache': False,
            'device': '',
            'workers': 8,
            'project': str(self.results_dir),
            'name': f'1956_TI_yolov8{model_size}',
            'exist_ok': False,
            'pretrained': True,
            'verbose': True,
            'seed': 0,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
        }
        
        return config
    
    def train_model(self, model_size='n', epochs=100, batch_size=16, img_size=640):
        """训练模型"""
        logger.info(f"开始训练 YOLOv8{model_size} 模型...")
        
        # 加载数据集配置
        dataset_config = self.load_dataset_config()
        
        # 创建训练配置
        train_config = self.create_training_config(model_size, epochs, batch_size, img_size)
        
        # 保存训练配置
        config_save_path = self.models_dir / f"train_config_yolov8{model_size}.json"
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(train_config, f, ensure_ascii=False, indent=2)
        
        # 初始化模型
        model_name = f'yolov8{model_size}.pt'
        model = YOLO(model_name)
        
        logger.info(f"使用预训练模型: {model_name}")
        logger.info(f"数据集路径: {self.config_file}")
        logger.info(f"训练参数: epochs={epochs}, batch_size={batch_size}, img_size={img_size}")
        
        # 开始训练
        try:
            results = model.train(
                data=str(self.config_file),
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                save=True,
                save_period=-1,
                cache=False,
                device='',
                workers=8,
                project=str(self.results_dir),
                name=f'1956_TI_yolov8{model_size}',
                exist_ok=True,
                pretrained=True,
                optimizer='AdamW',
                verbose=True,
                seed=0,
                deterministic=True,
                single_cls=False,
                rect=False,
                cos_lr=False,
                close_mosaic=10,
                resume=False,
                amp=True,
                fraction=1.0,
                profile=False,
                freeze=None,
                lr0=0.01,
                lrf=0.1,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                box=7.5,
                cls=0.5,
                dfl=1.5,
                dropout=0.0,
                val=True,
                plots=True,
                overlap_mask=True,
                mask_ratio=4,
            )
            
            logger.info("训练完成!")
            
            # 保存最佳模型到models目录
            best_model_src = self.results_dir / f'1956_TI_yolov8{model_size}' / 'weights' / 'best.pt'
            best_model_dst = self.models_dir / f'best_yolov8{model_size}.pt'
            
            if best_model_src.exists():
                import shutil
                shutil.copy2(best_model_src, best_model_dst)
                logger.info(f"最佳模型已保存到: {best_model_dst}")
            
            return results, best_model_dst
            
        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            raise
    
    def evaluate_model(self, model_path, test_data_path=None):
        """评估模型"""
        logger.info(f"评估模型: {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型
        model = YOLO(model_path)
        
        # 使用测试集评估
        if test_data_path is None:
            test_data_path = str(self.config_file)
        
        # 运行验证
        results = model.val(
            data=test_data_path,
            save_json=True,
            save_hybrid=True,
            conf=0.25,
            iou=0.6,
            max_det=300,
            half=True,
            device='',
            dnn=False,
            plots=True,
            rect=False,
            split='test'
        )
        
        logger.info("模型评估完成")
        return results
    
    def create_training_report(self, results, model_path, model_size):
        """创建训练报告"""
        report = {
            'model_info': {
                'model_size': model_size,
                'model_path': str(model_path),
                'total_parameters': 0,
                'model_size_mb': 0
            },
            'training_results': {},
            'evaluation_metrics': {}
        }
        
        # 获取模型信息
        if Path(model_path).exists():
            model_size_bytes = Path(model_path).stat().st_size
            report['model_info']['model_size_mb'] = model_size_bytes / (1024 * 1024)
        
        # 保存训练结果
        if results:
            # 这里可以添加更多的结果处理逻辑
            report['training_results'] = {
                'status': 'completed',
                'message': 'Training completed successfully'
            }
        
        # 保存报告
        report_file = self.results_dir / f'training_report_yolov8{model_size}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练报告已保存到: {report_file}")
        return report

def main():
    """主函数"""
    dataset_dir = "/Users/zhaoye/Desktop/1956_TI_Dataset"
    
    # 检查数据集是否已准备
    config_file = Path(dataset_dir) / "dataset_config.yaml"
    if not config_file.exists():
        logger.error("数据集配置文件不存在，请先运行 dataset_processor.py")
        return
    
    # 创建训练器
    trainer = ModelTrainer(dataset_dir)
    
    # 训练参数
    model_size = 'n'  # 可选: n, s, m, l, x
    epochs = 50  # 可以根据需要调整
    batch_size = 16
    img_size = 640
    
    try:
        logger.info("开始训练流程...")
        
        # 训练模型
        results, model_path = trainer.train_model(
            model_size=model_size,
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size
        )
        
        # 评估模型
        eval_results = trainer.evaluate_model(model_path)
        
        # 创建训练报告
        report = trainer.create_training_report(results, model_path, model_size)
        
        print("\n=== 训练完成 ===")
        print(f"最佳模型: {model_path}")
        print(f"模型大小: {report['model_info']['model_size_mb']:.2f} MB")
        print(f"训练报告: {trainer.results_dir / f'training_report_yolov8{model_size}.json'}")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

