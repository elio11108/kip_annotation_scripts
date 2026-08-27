#!/usr/bin/env python3
"""
增强训练系统
集成高速训练、Early Stop和30分钟超时控制
专为1956 TI数据集优化
"""

import os
import sys
import time
import signal
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

# 设置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/Users/zhaoye/Desktop/1956_TI_Dataset/training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    """训练超时异常"""
    pass

class EnhancedTrainingSystem:
    def __init__(self, dataset_dir: str, timeout_minutes: int = 30):
        self.dataset_dir = Path(dataset_dir)
        self.models_dir = self.dataset_dir / "models"
        self.results_dir = self.dataset_dir / "results"
        self.config_file = self.dataset_dir / "dataset_config.yaml"
        
        # 超时设置
        self.timeout_seconds = timeout_minutes * 60
        self.start_time = None
        self.training_stopped = False
        
        # 创建目录
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # 检查计算设备
        self.device = self._get_optimal_device()
        
        # 系统性能监控
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
    
    def _timeout_handler(self):
        """超时处理线程"""
        time.sleep(self.timeout_seconds)
        if not self.training_stopped:
            logger.warning(f"⚠️ 训练超时！已运行{self.timeout_seconds/60:.1f}分钟")
            self.training_stopped = True
            # 强制停止训练进程
            self._force_stop_training()
    
    def _force_stop_training(self):
        """强制停止训练"""
        try:
            # 获取当前进程
            current_process = psutil.Process()
            
            # 终止所有子进程
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
    
    def get_high_speed_training_params(self, model_size: str = 's') -> Dict[str, Any]:
        """获取高速训练参数配置"""
        
        # 基础高速配置
        base_params = {
            # 核心参数
            'data': str(self.config_file),
            'epochs': 100,           # 适中的轮数，让early stop决定
            'batch': 16 if self.device == 'mps' else 8,  # 根据设备优化批次大小
            'imgsz': 640,            # 标准图像大小
            'device': self.device,
            
            # Early Stop配置 - 关键！
            'patience': 15,          # 15轮无改善停止（更激进的early stop）
            
            # 高效学习率策略
            'lr0': 0.001,            # 初始学习率
            'lrf': 0.1,              # 最终学习率因子
            'momentum': 0.937,       # 动量
            'weight_decay': 0.0005,  # 权重衰减
            'warmup_epochs': 2,      # 减少预热轮数
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # 优化器设置
            'optimizer': 'AdamW',    # 高效优化器
            
            # 损失函数权重优化
            'box': 7.5,              # 边界框损失
            'cls': 0.5,              # 分类损失
            'dfl': 1.5,              # 分布焦点损失
            
            # 高效数据增强（减少计算开销）
            'hsv_h': 0.01,           # 最小色调变化
            'hsv_s': 0.3,            # 适度饱和度
            'hsv_v': 0.2,            # 适度亮度
            'degrees': 2.0,          # 小角度旋转
            'translate': 0.05,       # 轻微平移
            'scale': 0.3,            # 缩放
            'shear': 1.0,            # 剪切
            'perspective': 0.0001,   # 最小透视
            'flipud': 0.0,           # 不垂直翻转
            'fliplr': 0.2,           # 少量水平翻转
            'mosaic': 0.6,           # 减少mosaic（节省计算）
            'mixup': 0.05,           # 减少mixup
            'copy_paste': 0.05,      # 减少copy_paste
            
            # 高速训练策略
            'save_period': 20,       # 减少保存频率
            'val': True,             # 保持验证
            'plots': False,          # 关闭训练图表生成（节省时间）
            'deterministic': False,  # 关闭确定性（提高速度）
            'single_cls': False,
            'rect': True,            # 启用矩形训练（提高效率）
            'cos_lr': True,          # 余弦学习率
            'close_mosaic': 10,      # 提前关闭mosaic
            'amp': True,             # 混合精度训练（加速）
            'fraction': 1.0,         # 使用全部数据
            'cache': 'ram',          # 缓存到内存（加速）
            
            # 输出设置
            'project': str(self.results_dir),
            'name': f'enhanced_high_speed_yolov8{model_size}',
            'exist_ok': True,
            'save': True,
            'verbose': False,        # 减少输出（提高速度）
        }
        
        return base_params
    
    def estimate_training_time(self, model_size: str = 's') -> float:
        """预估训练时间"""
        logger.info("📊 训练时间预估")
        
        # 基于实际测试的时间估算（每epoch秒数）
        time_estimates = {
            'n': {'mps': 6, 'cuda': 4, 'cpu': 20},
            's': {'mps': 9, 'cuda': 6, 'cpu': 35},
            'm': {'mps': 15, 'cuda': 10, 'cpu': 60},
            'l': {'mps': 25, 'cuda': 18, 'cpu': 100}
        }
        
        device_key = 'mps' if self.device == 'mps' else ('cuda' if self.device == 'cuda' else 'cpu')
        seconds_per_epoch = time_estimates[model_size][device_key]
        
        # 考虑early stop，估算实际训练轮数
        estimated_epochs = 40  # 基于early stop的保守估计
        total_seconds = seconds_per_epoch * estimated_epochs
        
        logger.info(f"⚡ 模型: YOLOv8{model_size}")
        logger.info(f"🖥️ 设备: {self.device.upper()}")
        logger.info(f"📈 预估轮数: {estimated_epochs} (early stop)")
        logger.info(f"⏱️ 预估时间: {total_seconds/60:.1f}分钟")
        
        return total_seconds
    
    def train_with_enhanced_control(self, model_size: str = 's') -> Dict[str, Any]:
        """执行增强控制的训练"""
        logger.info("🚀" + "=" * 50)
        logger.info("🚀 启动增强训练系统")
        logger.info("🚀" + "=" * 50)
        
        # 记录开始时间
        self.start_time = time.time()
        
        # 预估训练时间
        estimated_time = self.estimate_training_time(model_size)
        
        if estimated_time > self.timeout_seconds:
            logger.warning(f"⚠️ 预估训练时间({estimated_time/60:.1f}分钟)超过超时限制({self.timeout_seconds/60:.1f}分钟)")
            logger.info("📉 自动调整为更激进的early stop设置")
        
        # 获取训练参数
        train_params = self.get_high_speed_training_params(model_size)
        
        # 启动超时监控线程
        timeout_thread = threading.Thread(target=self._timeout_handler, daemon=True)
        timeout_thread.start()
        
        # 启动系统监控
        self.system_monitor.start_monitoring()
        
        # 保存配置
        config_path = self.models_dir / f"enhanced_train_config_yolov8{model_size}.json"
        self._save_training_config(config_path, train_params, model_size)
        
        # 初始化模型
        model_name = f'yolov8{model_size}.pt'
        model = YOLO(model_name)
        
        logger.info(f"🎯 模型: {model_name}")
        logger.info(f"⏰ 超时设置: {self.timeout_seconds/60:.1f}分钟")
        logger.info(f"🛑 Early Stop: {train_params['patience']}轮无改善自动停止")
        
        training_results = {
            'success': False,
            'model_path': None,
            'training_time': 0,
            'stopped_reason': 'unknown',
            'final_metrics': {},
            'system_stats': {}
        }
        
        try:
            logger.info("🏁 开始训练...")
            
            # 执行训练
            results = model.train(**train_params)
            
            # 计算训练时间
            training_time = time.time() - self.start_time
            training_results['training_time'] = training_time
            
            if self.training_stopped:
                training_results['stopped_reason'] = 'timeout'
                logger.warning(f"⚠️ 训练因超时停止 ({training_time/60:.1f}分钟)")
                self._generate_timeout_report(training_time)
            else:
                training_results['stopped_reason'] = 'early_stop' if training_time < estimated_time * 0.8 else 'completed'
                logger.info(f"✅ 训练正常完成 ({training_time/60:.1f}分钟)")
            
            # 保存最佳模型
            model_path = self._save_best_model(model_size, results)
            training_results['model_path'] = model_path
            training_results['success'] = True
            
            # 分析训练结果
            metrics = self._analyze_training_results(model_size)
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
            self._generate_final_report(training_results, model_size)
        
        return training_results
    
    def _save_training_config(self, config_path: Path, params: Dict, model_size: str):
        """保存训练配置"""
        config_data = {
            'model_size': model_size,
            'timestamp': datetime.datetime.now().isoformat(),
            'timeout_minutes': self.timeout_seconds / 60,
            'device': self.device,
            'training_params': {k: str(v) if isinstance(v, Path) else v for k, v in params.items()},
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': sys.version,
                'torch_version': torch.__version__
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 配置已保存: {config_path}")
    
    def _save_best_model(self, model_size: str, results) -> Optional[Path]:
        """保存最佳模型"""
        try:
            source_path = self.results_dir / f'enhanced_high_speed_yolov8{model_size}' / 'weights' / 'best.pt'
            dest_path = self.models_dir / f'enhanced_best_yolov8{model_size}.pt'
            
            if source_path.exists():
                import shutil
                shutil.copy2(source_path, dest_path)
                
                model_size_mb = dest_path.stat().st_size / (1024 * 1024)
                logger.info(f"💾 模型已保存: {dest_path} ({model_size_mb:.2f} MB)")
                return dest_path
            else:
                logger.warning("⚠️ 未找到最佳模型文件")
                return None
                
        except Exception as e:
            logger.error(f"保存模型时出错: {e}")
            return None
    
    def _analyze_training_results(self, model_size: str) -> Dict[str, Any]:
        """分析训练结果"""
        results_path = self.results_dir / f'enhanced_high_speed_yolov8{model_size}'
        metrics = {}
        
        try:
            # 读取结果CSV
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
                        'final_val_loss': float(df['val/box_loss'].iloc[-1]) if 'val/box_loss' in df.columns else 0
                    }
                    
                    logger.info(f"📊 训练指标:")
                    logger.info(f"   轮数: {metrics['total_epochs']}")
                    logger.info(f"   最佳mAP50: {metrics['best_map50']:.4f}")
                    logger.info(f"   最佳mAP50-95: {metrics['best_map50_95']:.4f}")
                    
        except Exception as e:
            logger.error(f"分析训练结果时出错: {e}")
        
        return metrics
    
    def _generate_timeout_report(self, training_time: float):
        """生成超时报告"""
        report = {
            'timeout_occurred': True,
            'training_time_minutes': training_time / 60,
            'timeout_limit_minutes': self.timeout_seconds / 60,
            'possible_reasons': [
                '数据集过大，需要更多训练时间',
                '模型复杂度较高，计算时间较长',
                '硬件性能限制',
                'Early stop参数设置过于宽松',
                '数据增强策略计算开销过大'
            ],
            'recommendations': [
                '使用更小的模型(如yolov8n)',
                '减少early stop patience值',
                '降低数据增强强度',
                '增加超时时间限制',
                '使用更强的硬件'
            ],
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        report_path = self.results_dir / 'timeout_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.warning("📝 超时报告已生成")
        logger.warning(f"⚠️ 训练超过{self.timeout_seconds/60:.1f}分钟限制")
        logger.warning("💡 建议: 调整模型大小或训练参数")
    
    def _generate_final_report(self, results: Dict[str, Any], model_size: str):
        """生成最终训练报告"""
        report = {
            'training_session': {
                'timestamp': datetime.datetime.now().isoformat(),
                'model_size': model_size,
                'device': self.device,
                'dataset_info': {
                    'train_images': 75,
                    'val_images': 21,
                    'test_images': 12,
                    'total_annotations': 197
                }
            },
            'results': results,
            'performance_analysis': self._get_performance_analysis(results),
            'recommendations': self._get_recommendations(results)
        }
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f'enhanced_training_report_{timestamp}.json'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📋 最终报告已生成: {report_path}")
    
    def _get_performance_analysis(self, results: Dict[str, Any]) -> Dict[str, str]:
        """性能分析"""
        analysis = {}
        
        if results['success']:
            training_time = results['training_time'] / 60  # 转换为分钟
            
            if training_time < 10:
                analysis['speed'] = 'excellent'
                analysis['speed_comment'] = '训练速度极快，优化效果显著'
            elif training_time < 20:
                analysis['speed'] = 'good'
                analysis['speed_comment'] = '训练速度良好，在合理范围内'
            else:
                analysis['speed'] = 'slow'
                analysis['speed_comment'] = '训练速度较慢，建议优化参数'
            
            if results['stopped_reason'] == 'early_stop':
                analysis['convergence'] = 'excellent'
                analysis['convergence_comment'] = 'Early stop成功生效，模型收敛良好'
            elif results['stopped_reason'] == 'completed':
                analysis['convergence'] = 'good'
                analysis['convergence_comment'] = '完成全部训练轮数'
            elif results['stopped_reason'] == 'timeout':
                analysis['convergence'] = 'interrupted'
                analysis['convergence_comment'] = '因超时中断，建议调整参数'
        else:
            analysis['speed'] = 'failed'
            analysis['convergence'] = 'failed'
        
        return analysis
    
    def _get_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """获取建议"""
        recommendations = []
        
        if not results['success']:
            recommendations.append("检查错误日志，解决训练失败问题")
            return recommendations
        
        training_time = results['training_time'] / 60
        
        if results['stopped_reason'] == 'timeout':
            recommendations.extend([
                "考虑使用更小的模型(如yolov8n)",
                "减少early stop的patience值",
                "降低数据增强强度",
                "增加超时时间限制"
            ])
        
        if training_time > 20:
            recommendations.extend([
                "优化训练参数以提高速度",
                "考虑使用更强的硬件",
                "减少不必要的数据增强"
            ])
        
        if 'final_metrics' in results and results['final_metrics']:
            metrics = results['final_metrics']
            if metrics.get('best_map50', 0) < 0.5:
                recommendations.extend([
                    "模型性能较低，考虑增加训练轮数",
                    "调整学习率和优化器参数",
                    "检查数据质量和标注准确性"
                ])
        
        if not recommendations:
            recommendations.append("训练效果良好，可以进行推理测试")
        
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
                    
                    time.sleep(10)  # 每10秒记录一次
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
    dataset_dir = "/Users/zhaoye/Desktop/1956_TI_Dataset"
    
    # 检查数据集
    config_file = Path(dataset_dir) / "dataset_config.yaml"
    if not config_file.exists():
        logger.error("❌ 数据集配置文件不存在，请先运行数据处理")
        return
    
    # 创建增强训练系统
    training_system = EnhancedTrainingSystem(dataset_dir, timeout_minutes=30)
    
    print("🎯 增强训练系统配置:")
    print(f"📊 数据集: 75张训练图像, 21张验证图像, 12张测试图像")
    print(f"🖥️ 计算设备: {training_system.device.upper()}")
    print(f"⏰ 超时限制: 30分钟")
    print(f"🛑 Early Stop: 15轮无改善自动停止")
    print(f"⚡ 高速优化: 启用")
    
    # 选择模型大小（平衡速度和性能）
    model_size = 's'
    
    print(f"\n🚀 开始训练 YOLOv8{model_size}...")
    
    try:
        # 执行增强训练
        results = training_system.train_with_enhanced_control(model_size)
        
        # 显示结果
        print("\n" + "🎉" + "=" * 50)
        print("🎉 增强训练系统执行完成!")
        print("🎉" + "=" * 50)
        
        if results['success']:
            print(f"✅ 训练状态: 成功")
            print(f"⏱️ 训练时间: {results['training_time']/60:.1f}分钟")
            print(f"🛑 停止原因: {results['stopped_reason']}")
            
            if results['model_path']:
                print(f"💾 模型路径: {results['model_path']}")
            
            if results['final_metrics']:
                metrics = results['final_metrics']
                print(f"📊 训练轮数: {metrics.get('total_epochs', 'N/A')}")
                print(f"📊 最佳mAP50: {metrics.get('best_map50', 0):.4f}")
            
            if results['stopped_reason'] == 'timeout':
                print("⚠️ 注意: 训练因超过30分钟时间限制而停止")
                print("💡 建议: 查看超时报告了解详细原因")
            elif results['stopped_reason'] == 'early_stop':
                print("✅ Early Stop成功生效，训练在最佳时机停止")
            
        else:
            print(f"❌ 训练失败: {results.get('error', '未知错误')}")
        
        print(f"\n📋 详细报告: {training_system.results_dir}")
        print("📝 查看training.log获取完整日志")
        
    except KeyboardInterrupt:
        logger.info("用户中断训练")
    except Exception as e:
        logger.error(f"训练系统出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
