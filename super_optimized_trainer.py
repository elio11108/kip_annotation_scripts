#!/usr/bin/env python3
"""
超级优化训练系统
针对高分辨率图像和小数据集的极速训练解决方案
解决问题：
1. 高分辨率图像处理
2. 内存压力优化
3. 模型大小优化
4. 批次大小调整
5. Early Stop激进策略
"""

import os
import sys
import time
import threading
from pathlib import Path
import json
import logging
import datetime
import torch
from ultralytics import YOLO
import yaml
import psutil
from typing import Optional, Dict, Any, List
from PIL import Image
import shutil
import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(Path(__file__).resolve().parent / 'super_optimized_training.log'), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class SuperOptimizedTrainer:
    def __init__(self, dataset_dir: str, timeout_minutes: int = 15):
        self.dataset_dir = Path(dataset_dir)
        self.models_dir = self.dataset_dir / "models"
        self.results_dir = self.dataset_dir / "results"
        self.config_file = self.dataset_dir / "dataset_config.yaml"
        self.optimized_data_dir = self.dataset_dir / "optimized_data"
        
        # 超时设置（减少到15分钟）
        self.timeout_seconds = timeout_minutes * 60
        self.start_time = None
        self.training_stopped = False
        
        # 创建目录
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.optimized_data_dir.mkdir(exist_ok=True)
        
        # 检查计算设备
        self.device = self._get_optimal_device()
        
        # 系统监控
        self.system_monitor = SystemMonitor()
        
    def _get_optimal_device(self) -> str:
        """获取最优计算设备"""
        if torch.backends.mps.is_available():
            device = 'mps'
            device_info = 'Apple M2 MPS GPU'
        elif torch.cuda.is_available():
            device = 'cuda'
            device_info = f'CUDA GPU: {torch.cuda.get_device_name()}'
        else:
            device = 'cpu'
            device_info = f'CPU ({torch.get_num_threads()} threads)'
        
        logger.info(f"🖥️ 计算设备: {device_info}")
        return device
    
    def preprocess_images_for_speed(self):
        """预处理图像以提高训练速度"""
        logger.info("🔄 开始图像预处理优化...")
        
        # 创建优化后的数据目录结构
        for split in ['train', 'val', 'test']:
            (self.optimized_data_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.optimized_data_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        total_size_before = 0
        total_size_after = 0
        
        # 处理每个数据集分割
        for split in ['train', 'val', 'test']:
            source_images_dir = self.dataset_dir / 'processed_data' / split / 'images'
            source_labels_dir = self.dataset_dir / 'processed_data' / split / 'labels'
            
            target_images_dir = self.optimized_data_dir / split / 'images'
            target_labels_dir = self.optimized_data_dir / split / 'labels'
            
            if not source_images_dir.exists():
                continue
                
            for image_file in source_images_dir.glob('*.png'):
                try:
                    # 读取原图像
                    with Image.open(image_file) as img:
                        original_size = image_file.stat().st_size
                        total_size_before += original_size
                        
                        # 获取原始尺寸
                        width, height = img.size
                        
                        # 计算优化尺寸 - 保持宽高比，最大边不超过1024
                        max_size = 1024
                        if max(width, height) > max_size:
                            if width > height:
                                new_width = max_size
                                new_height = int(height * max_size / width)
                            else:
                                new_height = max_size
                                new_width = int(width * max_size / height)
                        else:
                            new_width, new_height = width, height
                        
                        # Resize图像
                        if (new_width, new_height) != (width, height):
                            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        else:
                            img_resized = img
                        
                        # 转换为RGB（如果是RGBA）并保存为JPG格式
                        if img_resized.mode in ('RGBA', 'LA'):
                            background = Image.new('RGB', img_resized.size, (255, 255, 255))
                            background.paste(img_resized, mask=img_resized.split()[-1] if img_resized.mode == 'RGBA' else None)
                            img_resized = background
                        elif img_resized.mode != 'RGB':
                            img_resized = img_resized.convert('RGB')
                        
                        # 保存优化后的图像（JPG格式，质量85）
                        output_image_file = target_images_dir / f"{image_file.stem}.jpg"
                        img_resized.save(output_image_file, 'JPEG', quality=85, optimize=True)
                        
                        # 计算压缩比
                        new_size = output_image_file.stat().st_size
                        total_size_after += new_size
                        
                        # 处理对应的标签文件
                        label_file = source_labels_dir / f"{image_file.stem}.txt"
                        target_label_file = target_labels_dir / f"{image_file.stem}.txt"
                        
                        if label_file.exists():
                            # 如果图像尺寸改变了，需要调整标签
                            if (new_width, new_height) != (width, height):
                                self._adjust_label_coordinates(
                                    label_file, target_label_file, 
                                    (width, height), (new_width, new_height)
                                )
                            else:
                                shutil.copy2(label_file, target_label_file)
                        
                        processed_count += 1
                        
                except Exception as e:
                    logger.error(f"处理图像 {image_file} 失败: {e}")
        
        # 统计信息
        compression_ratio = (1 - total_size_after / total_size_before) * 100 if total_size_before > 0 else 0
        
        logger.info(f"✅ 图像预处理完成:")
        logger.info(f"   处理图像数: {processed_count}")
        logger.info(f"   原始大小: {total_size_before / (1024*1024):.1f} MB")
        logger.info(f"   优化大小: {total_size_after / (1024*1024):.1f} MB")
        logger.info(f"   压缩比: {compression_ratio:.1f}%")
        
        # 创建优化后的配置文件
        self._create_optimized_config()
        
        return processed_count
    
    def _adjust_label_coordinates(self, source_label: Path, target_label: Path, 
                                original_size: tuple, new_size: tuple):
        """调整标签坐标以适应新的图像尺寸"""
        try:
            with open(source_label, 'r') as f:
                lines = f.readlines()
            
            adjusted_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, center_x, center_y, width, height = parts
                    # YOLO格式的坐标是相对的，不需要调整
                    adjusted_lines.append(line)
                else:
                    adjusted_lines.append(line)
            
            with open(target_label, 'w') as f:
                f.writelines(adjusted_lines)
                
        except Exception as e:
            logger.error(f"调整标签坐标失败 {source_label}: {e}")
            # 如果调整失败，直接复制原文件
            shutil.copy2(source_label, target_label)
    
    def _create_optimized_config(self):
        """创建优化后的配置文件"""
        config = {
            'path': str(self.optimized_data_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,
            'names': ['image']
        }
        
        optimized_config_file = self.dataset_dir / 'optimized_dataset_config.yaml'
        with open(optimized_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"📝 优化配置文件: {optimized_config_file}")
        return optimized_config_file
    
    def get_super_optimized_params(self) -> Dict[str, Any]:
        """获取超级优化的训练参数"""
        
        # 根据可用内存动态调整批次大小
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if available_memory < 2:
            batch_size = 2
        elif available_memory < 4:
            batch_size = 4
        else:
            batch_size = 8
        
        params = {
            # 核心参数 - 超级优化
            'data': str(self.dataset_dir / 'optimized_dataset_config.yaml'),
            'epochs': 60,            # 减少轮数
            'batch': batch_size,     # 动态批次大小
            'imgsz': 416,           # 降低训练分辨率（从640到416）
            'device': self.device,
            
            # 激进的Early Stop
            'patience': 8,           # 8轮无改善就停止
            
            # 优化的学习率策略
            'lr0': 0.002,           # 稍高的初始学习率（快速收敛）
            'lrf': 0.01,            # 最终学习率因子
            'momentum': 0.9,        # 标准动量
            'weight_decay': 0.0001, # 减少权重衰减
            'warmup_epochs': 1,     # 最小预热
            'warmup_momentum': 0.5,
            'warmup_bias_lr': 0.05,
            
            # 优化器
            'optimizer': 'SGD',     # SGD比AdamW更快，内存占用更少
            
            # 损失函数权重
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            
            # 最小化数据增强（提高速度）
            'hsv_h': 0.005,         # 最小色调变化
            'hsv_s': 0.1,           # 最小饱和度变化
            'hsv_v': 0.1,           # 最小亮度变化
            'degrees': 1.0,         # 最小旋转
            'translate': 0.02,      # 最小平移
            'scale': 0.1,           # 最小缩放
            'shear': 0.5,           # 最小剪切
            'perspective': 0.0,     # 关闭透视变换
            'flipud': 0.0,          # 关闭垂直翻转
            'fliplr': 0.1,          # 最小水平翻转
            'mosaic': 0.3,          # 减少mosaic
            'mixup': 0.0,           # 关闭mixup
            'copy_paste': 0.0,      # 关闭copy_paste
            
            # 速度优化设置
            'save_period': 30,      # 减少保存频率
            'val': True,            # 保持验证
            'plots': False,         # 关闭图表生成
            'deterministic': False, # 关闭确定性（提高速度）
            'single_cls': False,
            'rect': True,           # 启用矩形训练
            'cos_lr': False,        # 关闭余弦学习率（简化）
            'close_mosaic': 5,      # 提前关闭mosaic
            'amp': True,            # 混合精度
            'fraction': 1.0,
            'cache': True,          # 缓存图像到内存
            'workers': 2,           # 减少worker数量（节省内存）
            
            # 输出设置
            'project': str(self.results_dir),
            'name': 'super_optimized_yolov8n',
            'exist_ok': True,
            'save': True,
            'verbose': False,
        }
        
        return params
    
    def estimate_optimized_time(self) -> float:
        """预估优化后的训练时间"""
        logger.info("📊 优化训练时间预估")
        
        # 基于优化的时间估算（每epoch秒数）
        time_estimates = {
            'mps': 3,    # Apple M2 MPS - 大幅优化
            'cuda': 2,   # CUDA GPU - 大幅优化
            'cpu': 12    # CPU - 优化后
        }
        
        device_key = 'mps' if self.device == 'mps' else ('cuda' if self.device == 'cuda' else 'cpu')
        seconds_per_epoch = time_estimates[device_key]
        
        # 考虑early stop，估算实际训练轮数
        estimated_epochs = 25  # 基于激进early stop的估计
        total_seconds = seconds_per_epoch * estimated_epochs
        
        logger.info(f"⚡ 模型: YOLOv8n (超轻量)")
        logger.info(f"🖥️ 设备: {self.device.upper()}")
        logger.info(f"📏 分辨率: 416×416 (优化)")
        logger.info(f"📈 预估轮数: {estimated_epochs} (激进early stop)")
        logger.info(f"⏱️ 预估时间: {total_seconds/60:.1f}分钟")
        
        return total_seconds
    
    def _timeout_handler(self):
        """超时处理线程"""
        time.sleep(self.timeout_seconds)
        if not self.training_stopped:
            logger.warning(f"⚠️ 训练超时！已运行{self.timeout_seconds/60:.1f}分钟")
            self.training_stopped = True
            self._force_stop_training()
    
    def _force_stop_training(self):
        """强制停止训练"""
        try:
            current_process = psutil.Process()
            for child in current_process.children(recursive=True):
                try:
                    child.terminate()
                    child.wait(timeout=5)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
            logger.info("🛑 训练进程已强制停止")
        except Exception as e:
            logger.error(f"强制停止训练时出错: {e}")
    
    def train_super_optimized(self) -> Dict[str, Any]:
        """执行超级优化训练"""
        logger.info("🚀" + "=" * 60)
        logger.info("🚀 启动超级优化训练系统")
        logger.info("🚀" + "=" * 60)
        
        # 记录开始时间
        self.start_time = time.time()
        
        # 预处理图像
        processed_images = self.preprocess_images_for_speed()
        if processed_images == 0:
            logger.error("❌ 图像预处理失败，无法继续训练")
            return {'success': False, 'error': 'Image preprocessing failed'}
        
        # 预估训练时间
        estimated_time = self.estimate_optimized_time()
        
        # 获取超级优化参数
        train_params = self.get_super_optimized_params()
        
        # 启动超时监控
        timeout_thread = threading.Thread(target=self._timeout_handler, daemon=True)
        timeout_thread.start()
        
        # 启动系统监控
        self.system_monitor.start_monitoring()
        
        # 保存配置
        config_path = self.models_dir / "super_optimized_config_yolov8n.json"
        self._save_training_config(config_path, train_params)
        
        # 使用YOLOv8n模型（最轻量）
        model_name = 'yolov8n.pt'
        model = YOLO(model_name)
        
        logger.info(f"🎯 模型: {model_name} (6.5MB)")
        logger.info(f"⏰ 超时设置: {self.timeout_seconds/60:.1f}分钟")
        logger.info(f"🛑 Early Stop: {train_params['patience']}轮无改善自动停止")
        logger.info(f"📏 训练分辨率: {train_params['imgsz']}×{train_params['imgsz']}")
        logger.info(f"📦 批次大小: {train_params['batch']}")
        
        training_results = {
            'success': False,
            'model_path': None,
            'training_time': 0,
            'stopped_reason': 'unknown',
            'final_metrics': {},
            'system_stats': {},
            'optimization_stats': {
                'processed_images': processed_images,
                'estimated_time': estimated_time,
                'actual_speedup': 0
            }
        }
        
        try:
            logger.info("🏁 开始超级优化训练...")
            
            # 执行训练
            results = model.train(**train_params)
            
            # 计算训练时间
            training_time = time.time() - self.start_time
            training_results['training_time'] = training_time
            training_results['optimization_stats']['actual_speedup'] = estimated_time / training_time if training_time > 0 else 1
            
            if self.training_stopped:
                training_results['stopped_reason'] = 'timeout'
                logger.warning(f"⚠️ 训练因超时停止 ({training_time/60:.1f}分钟)")
            else:
                training_results['stopped_reason'] = 'early_stop' if training_time < estimated_time * 0.8 else 'completed'
                logger.info(f"✅ 训练完成 ({training_time/60:.1f}分钟)")
            
            # 保存最佳模型
            model_path = self._save_best_model(results)
            training_results['model_path'] = model_path
            training_results['success'] = True
            
            # 分析训练结果
            metrics = self._analyze_training_results()
            training_results['final_metrics'] = metrics
            
        except Exception as e:
            logger.error(f"❌ 训练过程出错: {e}")
            training_results['stopped_reason'] = 'error'
            training_results['error'] = str(e)
        
        finally:
            # 停止系统监控
            self.system_monitor.stop_monitoring()
            training_results['system_stats'] = self.system_monitor.get_stats()
            
            # 生成最终报告
            self._generate_final_report(training_results)
        
        return training_results
    
    def _save_training_config(self, config_path: Path, params: Dict):
        """保存训练配置"""
        config_data = {
            'model_size': 'n',
            'timestamp': datetime.datetime.now().isoformat(),
            'timeout_minutes': self.timeout_seconds / 60,
            'device': self.device,
            'optimization_features': [
                '图像预处理优化',
                '分辨率降低到416',
                '使用YOLOv8n轻量模型',
                '动态批次大小',
                '激进Early Stop',
                '最小化数据增强',
                '内存优化'
            ],
            'training_params': {k: str(v) if isinstance(v, Path) else v for k, v in params.items()},
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                'python_version': sys.version,
                'torch_version': torch.__version__
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 配置已保存: {config_path}")
    
    def _save_best_model(self, results) -> Optional[Path]:
        """保存最佳模型"""
        try:
            source_path = self.results_dir / 'super_optimized_yolov8n' / 'weights' / 'best.pt'
            dest_path = self.models_dir / 'super_optimized_best_yolov8n.pt'
            
            if source_path.exists():
                shutil.copy2(source_path, dest_path)
                model_size_mb = dest_path.stat().st_size / (1024 * 1024)
                logger.info(f"💾 优化模型已保存: {dest_path} ({model_size_mb:.2f} MB)")
                return dest_path
            else:
                logger.warning("⚠️ 未找到最佳模型文件")
                return None
        except Exception as e:
            logger.error(f"保存模型时出错: {e}")
            return None
    
    def _analyze_training_results(self) -> Dict[str, Any]:
        """分析训练结果"""
        results_path = self.results_dir / 'super_optimized_yolov8n'
        metrics = {}
        
        try:
            results_csv = results_path / 'results.csv'
            if results_csv.exists():
                import pandas as pd
                df = pd.read_csv(results_csv)
                
                if not df.empty:
                    metrics = {
                        'total_epochs': len(df),
                        'best_map50': float(df['metrics/mAP50(B)'].max()) if 'metrics/mAP50(B)' in df.columns else 0,
                        'best_map50_95': float(df['metrics/mAP50-95(B)'].max()) if 'metrics/mAP50-95(B)' in df.columns else 0,
                        'final_train_loss': float(df['train/box_loss'].iloc[-1]) if 'train/box_loss' in df.columns else 0,
                        'final_val_loss': float(df['val/box_loss'].iloc[-1]) if 'val/box_loss' in df.columns else 0,
                        'convergence_epoch': int(df['metrics/mAP50(B)'].idxmax()) if 'metrics/mAP50(B)' in df.columns else 0
                    }
                    
                    logger.info(f"📊 训练指标:")
                    logger.info(f"   总轮数: {metrics['total_epochs']}")
                    logger.info(f"   最佳mAP50: {metrics['best_map50']:.4f}")
                    logger.info(f"   收敛轮次: {metrics['convergence_epoch']}")
                    
                    if metrics['total_epochs'] < 30:
                        logger.info("✅ Early Stop成功生效！")
                    
        except Exception as e:
            logger.error(f"分析训练结果时出错: {e}")
        
        return metrics
    
    def _generate_final_report(self, results: Dict[str, Any]):
        """生成最终训练报告"""
        report = {
            'super_optimized_training': {
                'timestamp': datetime.datetime.now().isoformat(),
                'model': 'YOLOv8n',
                'device': self.device,
                'optimizations_applied': [
                    '图像预处理：PNG->JPG，分辨率优化',
                    '模型选择：YOLOv8s->YOLOv8n (75%减少)',
                    '训练分辨率：640->416 (45%减少)',
                    '激进Early Stop：15->8轮',
                    '数据增强最小化',
                    '内存优化：动态批次大小',
                    '优化器：AdamW->SGD'
                ],
                'dataset_info': {
                    'train_images': 75,
                    'val_images': 21,
                    'test_images': 12,
                    'total_annotations': 197,
                    'processed_images': results['optimization_stats']['processed_images']
                }
            },
            'performance_results': results,
            'speed_analysis': self._analyze_speed_improvement(results),
            'recommendations': self._get_optimization_recommendations(results)
        }
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f'super_optimized_report_{timestamp}.json'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📋 超级优化报告: {report_path}")
    
    def _analyze_speed_improvement(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析速度提升"""
        analysis = {}
        
        if results['success']:
            training_time_minutes = results['training_time'] / 60
            estimated_original_time = 30  # 原始预估时间
            
            speed_improvement = estimated_original_time / training_time_minutes if training_time_minutes > 0 else 1
            
            analysis = {
                'estimated_original_time_minutes': estimated_original_time,
                'actual_training_time_minutes': training_time_minutes,
                'speed_improvement_factor': speed_improvement,
                'time_saved_minutes': estimated_original_time - training_time_minutes,
                'efficiency_rating': 'excellent' if speed_improvement > 4 else 'good' if speed_improvement > 2 else 'moderate'
            }
            
            logger.info(f"🚀 速度提升分析:")
            logger.info(f"   原始预估: {estimated_original_time}分钟")
            logger.info(f"   实际用时: {training_time_minutes:.1f}分钟")
            logger.info(f"   速度提升: {speed_improvement:.1f}倍")
            logger.info(f"   节省时间: {analysis['time_saved_minutes']:.1f}分钟")
        
        return analysis
    
    def _get_optimization_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """获取优化建议"""
        recommendations = []
        
        if not results['success']:
            recommendations.append("检查错误日志，解决训练失败问题")
            return recommendations
        
        training_time = results['training_time'] / 60
        
        if training_time < 5:
            recommendations.extend([
                "训练速度极佳！可以考虑稍微增加模型复杂度",
                "如果精度满足需求，当前配置是最优的"
            ])
        elif training_time < 10:
            recommendations.extend([
                "训练速度良好，配置合理",
                "可以进行推理测试验证模型效果"
            ])
        else:
            recommendations.extend([
                "训练时间仍较长，考虑进一步优化",
                "检查系统资源使用情况"
            ])
        
        if 'final_metrics' in results and results['final_metrics']:
            metrics = results['final_metrics']
            if metrics.get('best_map50', 0) < 0.3:
                recommendations.append("模型精度较低，考虑增加训练数据或调整参数")
            elif metrics.get('best_map50', 0) > 0.7:
                recommendations.append("模型精度良好，可以投入使用")
        
        recommendations.append("建议在实际数据上测试模型性能")
        
        return recommendations


class SystemMonitor:
    """系统性能监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'start_time': None,
            'end_time': None
        }
    
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.stats['start_time'] = time.time()
        
        def monitor_loop():
            while self.monitoring:
                try:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_percent = psutil.virtual_memory().percent
                    
                    self.stats['cpu_usage'].append(cpu_percent)
                    self.stats['memory_usage'].append(memory_percent)
                    
                    time.sleep(5)  # 每5秒记录一次（更频繁）
                except Exception as e:
                    logger.error(f"系统监控出错: {e}")
                    break
        
        threading.Thread(target=monitor_loop, daemon=True).start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        self.stats['end_time'] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.stats['cpu_usage']:
            return {}
        
        import numpy as np
        
        return {
            'avg_cpu_usage': float(np.mean(self.stats['cpu_usage'])),
            'max_cpu_usage': float(np.max(self.stats['cpu_usage'])),
            'avg_memory_usage': float(np.mean(self.stats['memory_usage'])),
            'max_memory_usage': float(np.max(self.stats['memory_usage'])),
            'monitoring_duration': self.stats['end_time'] - self.stats['start_time'] if self.stats['end_time'] else 0
        }


def main():
    """主函数"""
    dataset_dir = str(Path(__file__).resolve().parent)

    # 检查数据集
    config_file = Path(dataset_dir) / "dataset_config.yaml"
    if not config_file.exists():
        logger.error("❌ 数据集配置文件不存在，请先运行数据处理")
        return
    
    # 创建超级优化训练系统
    trainer = SuperOptimizedTrainer(dataset_dir, timeout_minutes=15)
    
    print("🚀 超级优化训练系统")
    print("=" * 50)
    print("🎯 优化特性:")
    print("  📏 图像预处理：高分辨率->优化分辨率")
    print("  🎯 模型选择：YOLOv8s->YOLOv8n (轻量化)")
    print("  📐 训练分辨率：640->416 (速度优化)")
    print("  🛑 激进Early Stop：8轮无改善停止")
    print("  💾 内存优化：动态批次大小")
    print("  ⚡ 数据增强：最小化处理")
    print("=" * 50)
    print(f"📊 数据集: 75张训练, 21张验证, 12张测试")
    print(f"🖥️ 计算设备: {trainer.device.upper()}")
    print(f"⏰ 超时限制: 15分钟")
    print(f"🎯 预期训练时间: 3-8分钟")
    
    try:
        print(f"\n🚀 开始超级优化训练...")
        
        # 执行超级优化训练
        results = trainer.train_super_optimized()
        
        # 显示结果
        print("\n" + "🎉" + "=" * 60)
        print("🎉 超级优化训练完成!")
        print("🎉" + "=" * 60)
        
        if results['success']:
            training_time = results['training_time'] / 60
            print(f"✅ 训练状态: 成功")
            print(f"⏱️ 训练时间: {training_time:.1f}分钟")
            print(f"🛑 停止原因: {results['stopped_reason']}")
            
            if 'optimization_stats' in results:
                speedup = results['optimization_stats'].get('actual_speedup', 1)
                print(f"🚀 速度提升: {speedup:.1f}倍")
            
            if results['model_path']:
                print(f"💾 模型路径: {results['model_path']}")
            
            if results['final_metrics']:
                metrics = results['final_metrics']
                print(f"📊 训练轮数: {metrics.get('total_epochs', 'N/A')}")
                print(f"📊 最佳mAP50: {metrics.get('best_map50', 0):.4f}")
                
                if metrics.get('total_epochs', 0) < 30:
                    print("✅ Early Stop成功生效!")
            
            if training_time < 10:
                print("🏆 训练速度优异！优化效果显著")
            
        else:
            print(f"❌ 训练失败: {results.get('error', '未知错误')}")
        
        print(f"\n📋 详细报告: {trainer.results_dir}")
        print("📝 查看super_optimized_training.log获取完整日志")
        
    except KeyboardInterrupt:
        logger.info("用户中断训练")
    except Exception as e:
        logger.error(f"超级优化训练系统出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
