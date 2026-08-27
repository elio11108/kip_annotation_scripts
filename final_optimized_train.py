#!/usr/bin/env python3
"""
最终优化训练脚本
包含Early Stop和完整的训练时间预估
"""

import os
import sys
import time
from pathlib import Path
import json
import logging
import torch
from ultralytics import YOLO
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalOptimizedTrainer:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.models_dir = self.dataset_dir / "models"
        self.results_dir = self.dataset_dir / "results"
        self.config_file = self.dataset_dir / "dataset_config.yaml"
        
        # 创建目录
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # 检查GPU状态
        self.device = self.get_best_device()
        
    def get_best_device(self):
        """获取最佳计算设备"""
        if torch.backends.mps.is_available():
            device = 'mps'
            device_name = 'Apple M2 MPS GPU'
        elif torch.cuda.is_available():
            device = 'cuda'
            device_name = f'CUDA GPU: {torch.cuda.get_device_name()}'
        else:
            device = 'cpu'
            device_name = f'CPU ({torch.get_num_threads()} threads)'
        
        logger.info(f"使用计算设备: {device_name}")
        return device
    
    def estimate_training_time(self, model_size='s', epochs=150):
        """预估训练时间"""
        logger.info("=== 训练时间预估 ===")
        
        # 基础时间估算（每个epoch的秒数）
        base_times = {
            'n': {'mps': 8, 'cpu': 25},
            's': {'mps': 12, 'cpu': 40}, 
            'm': {'mps': 20, 'cpu': 70},
            'l': {'mps': 35, 'cpu': 120}
        }
        
        device_type = 'mps' if self.device == 'mps' else 'cpu'
        seconds_per_epoch = base_times[model_size][device_type]
        
        # 计算总时间
        total_seconds = seconds_per_epoch * epochs
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        
        logger.info(f"模型大小: YOLOv8{model_size}")
        logger.info(f"计算设备: {self.device.upper()}")
        logger.info(f"数据集大小: 75张训练图像")
        logger.info(f"预计训练轮数: {epochs}")
        logger.info(f"每轮预估时间: {seconds_per_epoch}秒")
        logger.info(f"总预估时间: {hours}小时{minutes}分钟")
        
        return total_seconds
    
    def get_optimized_params_with_early_stop(self, model_size='s'):
        """获取包含Early Stop的优化训练参数"""
        
        # Early Stop配置
        early_stop_config = {
            'patience': 20,          # 20轮没有改善就停止
            'min_delta': 0.001,      # 最小改善阈值
            'monitor': 'val_loss',   # 监控验证损失
        }
        
        params = {
            # 基础参数
            'data': str(self.config_file),
            'epochs': 200,           # 设置较大值，让early stop决定何时停止
            'batch': 8,              # 适合M2芯片的批次大小
            'imgsz': 640,            # 标准图像大小
            'device': self.device,   # 使用最佳设备
            
            # Early Stop设置
            'patience': early_stop_config['patience'],  # 关键：Early Stop耐心值
            
            # 学习率策略（更保守，适合Early Stop）
            'lr0': 0.0005,           # 较低的初始学习率
            'lrf': 0.01,             # 最终学习率因子
            'momentum': 0.937,       # 动量
            'weight_decay': 0.0005,  # 权重衰减
            'warmup_epochs': 3,      # 预热轮数
            'warmup_momentum': 0.8,  # 预热动量
            'warmup_bias_lr': 0.1,   # 预热偏置学习率
            
            # 优化器
            'optimizer': 'AdamW',    # AdamW优化器
            
            # 损失函数权重
            'box': 7.5,              # 边界框损失权重
            'cls': 0.5,              # 分类损失权重
            'dfl': 1.5,              # 分布焦点损失权重
            
            # 数据增强（适合文档图像）
            'hsv_h': 0.01,           # 轻微色调变化
            'hsv_s': 0.4,            # 适度饱和度变化
            'hsv_v': 0.3,            # 适度亮度变化
            'degrees': 3.0,          # 小角度旋转
            'translate': 0.1,        # 轻微平移
            'scale': 0.4,            # 缩放范围
            'shear': 2.0,            # 剪切变换
            'perspective': 0.0002,   # 轻微透视变换
            'flipud': 0.0,           # 不垂直翻转
            'fliplr': 0.3,           # 适度水平翻转
            'mosaic': 0.8,           # Mosaic增强
            'mixup': 0.1,            # Mixup增强
            'copy_paste': 0.1,       # Copy-paste增强
            
            # 训练策略
            'save_period': 10,       # 每10轮保存一次
            'val': True,             # 启用验证
            'plots': True,           # 生成训练图表
            'deterministic': True,   # 确定性训练
            'single_cls': False,     # 多类别支持
            'rect': False,           # 矩形训练
            'cos_lr': True,          # 余弦学习率调度
            'close_mosaic': 15,      # 最后15轮关闭mosaic
            'amp': True,             # 混合精度训练
            'fraction': 1.0,         # 使用全部数据
            
            # 输出设置
            'project': str(self.results_dir),
            'name': f'final_optimized_yolov8{model_size}',
            'exist_ok': True,
            'save': True,
            'verbose': True,
        }
        
        return params, early_stop_config
    
    def train_with_early_stop(self, model_size='s'):
        """使用Early Stop训练模型"""
        logger.info("=" * 60)
        logger.info("🚀 开始最终优化训练（包含Early Stop）")
        logger.info("=" * 60)
        
        # 预估训练时间
        estimated_time = self.estimate_training_time(model_size)
        
        # 获取优化参数
        train_params, early_stop_config = self.get_optimized_params_with_early_stop(model_size)
        
        # 保存训练配置
        config_save_path = self.models_dir / f"final_optimized_config_yolov8{model_size}.json"
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json_params = {k: str(v) if isinstance(v, Path) else v for k, v in train_params.items()}
            json_params['early_stop_config'] = early_stop_config
            json_params['estimated_time_seconds'] = estimated_time
            json.dump(json_params, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练配置已保存到: {config_save_path}")
        
        # 初始化模型
        model_name = f'yolov8{model_size}.pt'
        model = YOLO(model_name)
        
        logger.info(f"使用预训练模型: {model_name}")
        logger.info(f"Early Stop设置: {early_stop_config['patience']}轮无改善自动停止")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 开始训练
            logger.info("🏁 开始训练...")
            results = model.train(**train_params)
            
            # 计算实际训练时间
            actual_time = time.time() - start_time
            actual_hours = actual_time // 3600
            actual_minutes = (actual_time % 3600) // 60
            
            logger.info("🎉 训练完成!")
            logger.info(f"⏱️ 实际训练时间: {actual_hours:.0f}小时{actual_minutes:.0f}分钟")
            
            # 保存最佳模型
            best_model_src = self.results_dir / f'final_optimized_yolov8{model_size}' / 'weights' / 'best.pt'
            best_model_dst = self.models_dir / f'final_optimized_best_yolov8{model_size}.pt'
            
            if best_model_src.exists():
                import shutil
                shutil.copy2(best_model_src, best_model_dst)
                logger.info(f"最佳模型已保存到: {best_model_dst}")
                
                # 获取模型大小
                model_size_mb = best_model_dst.stat().st_size / (1024 * 1024)
                logger.info(f"模型文件大小: {model_size_mb:.2f} MB")
            
            return results, best_model_dst, actual_time
            
        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            raise
    
    def analyze_training_results(self, results_dir, model_size):
        """分析训练结果"""
        results_path = self.results_dir / f'final_optimized_yolov8{model_size}'
        
        # 读取训练结果
        results_csv = results_path / 'results.csv'
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            
            # 获取最佳指标
            best_epoch = df['metrics/mAP50(B)'].idxmax()
            best_map50 = df['metrics/mAP50(B)'].max()
            best_map50_95 = df['metrics/mAP50-95(B)'].max()
            final_loss = df['train/box_loss'].iloc[-1]
            
            logger.info("📊 训练结果分析:")
            logger.info(f"   最佳轮次: {best_epoch + 1}")
            logger.info(f"   最佳mAP50: {best_map50:.4f}")
            logger.info(f"   最佳mAP50-95: {best_map50_95:.4f}")
            logger.info(f"   最终损失: {final_loss:.4f}")
            
            # 判断是否Early Stop生效
            total_epochs = len(df)
            if total_epochs < 180:  # 如果少于设定的200轮
                logger.info(f"✅ Early Stop生效: 在第{total_epochs}轮自动停止")
            else:
                logger.info(f"⏰ 完成全部{total_epochs}轮训练")
            
            return {
                'best_epoch': best_epoch,
                'best_map50': best_map50,
                'best_map50_95': best_map50_95,
                'final_loss': final_loss,
                'total_epochs': total_epochs
            }
        
        return None

def main():
    """主训练函数"""
    dataset_dir = "/Users/zhaoye/Desktop/1956_TI_Dataset"
    
    # 检查数据集
    config_file = Path(dataset_dir) / "dataset_config.yaml"
    if not config_file.exists():
        logger.error("数据集配置文件不存在")
        return
    
    # 创建训练器
    trainer = FinalOptimizedTrainer(dataset_dir)
    
    # 用户确认
    print("🎯 训练配置确认:")
    print(f"📊 数据集: 75张训练图像, 21张验证图像")
    print(f"🖥️ 计算设备: {trainer.device.upper()}")
    
    # 模型选择
    model_size = 's'  # 使用YOLOv8s平衡性能和速度
    
    # 预估时间
    estimated_time = trainer.estimate_training_time(model_size)
    
    print(f"\n⏱️ 开始训练...")
    print(f"🔧 Early Stop: 20轮无改善自动停止")
    print(f"📈 最大轮数: 200轮")
    
    try:
        # 开始训练
        results, model_path, actual_time = trainer.train_with_early_stop(model_size)
        
        # 分析结果
        analysis = trainer.analyze_training_results(trainer.results_dir, model_size)
        
        # 最终报告
        print("\n" + "=" * 60)
        print("🎉 最终优化训练完成!")
        print("=" * 60)
        print(f"📁 最佳模型: {model_path}")
        print(f"⏱️ 实际训练时间: {actual_time/60:.1f}分钟")
        
        if analysis:
            print(f"📊 最佳mAP50: {analysis['best_map50']:.4f}")
            print(f"📊 训练轮数: {analysis['total_epochs']}")
            if analysis['total_epochs'] < 180:
                print("✅ Early Stop成功生效!")
        
        print(f"📈 详细结果: {trainer.results_dir / f'final_optimized_yolov8{model_size}'}")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
