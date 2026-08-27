#!/usr/bin/env python3
"""
1956 TI 数据集完整Pipeline管理脚本
统一管理整个数据集创建、训练和推理流程
"""

import os
import sys
import subprocess
from pathlib import Path
import json
import logging
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineManager:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(exist_ok=True)
        
        # 各个脚本路径
        self.annotation_tool = self.dataset_dir / "image_annotation_tool.py"
        self.dataset_processor = self.dataset_dir / "dataset_processor.py"
        self.model_trainer = self.dataset_dir / "train_model.py"
        self.inference_script = self.dataset_dir / "inference_model.py"
        
        # 状态文件
        self.status_file = self.dataset_dir / "pipeline_status.json"
        
        # 初始化状态
        self.status = self.load_status()
    
    def load_status(self):
        """加载Pipeline状态"""
        if self.status_file.exists():
            with open(self.status_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'steps': {
                    'annotation': {'completed': False, 'timestamp': None},
                    'dataset_processing': {'completed': False, 'timestamp': None},
                    'model_training': {'completed': False, 'timestamp': None, 'model_path': None},
                    'inference': {'completed': False, 'timestamp': None}
                },
                'statistics': {}
            }
    
    def save_status(self):
        """保存Pipeline状态"""
        self.status['last_updated'] = datetime.now().isoformat()
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.status, f, ensure_ascii=False, indent=2)
    
    def check_dependencies(self):
        """检查依赖"""
        logger.info("检查依赖...")
        
        required_packages = [
            'ultralytics', 'torch', 'torchvision', 'PIL', 
            'opencv-python', 'matplotlib', 'seaborn', 'scikit-learn',
            'numpy', 'pandas', 'pyyaml'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.warning(f"缺少以下依赖包: {missing_packages}")
            logger.info("请运行: pip install " + " ".join(missing_packages))
            return False
        
        logger.info("所有依赖检查通过")
        return True
    
    def step1_annotation(self):
        """步骤1: 图像标注"""
        logger.info("=== 步骤1: 图像标注 ===")
        
        if self.status['steps']['annotation']['completed']:
            logger.info("图像标注已完成，跳过此步骤")
            return True
        
        logger.info("启动图像标注工具...")
        logger.info(f"请运行: python {self.annotation_tool}")
        logger.info("标注完成后，请继续执行下一步")
        
        # 检查是否有标注文件
        annotations_dir = self.dataset_dir / "annotations"
        if annotations_dir.exists():
            annotation_files = list(annotations_dir.glob("*.json"))
            if annotation_files:
                logger.info(f"检测到 {len(annotation_files)} 个标注文件")
                self.status['steps']['annotation']['completed'] = True
                self.status['steps']['annotation']['timestamp'] = datetime.now().isoformat()
                self.status['statistics']['num_annotations'] = len(annotation_files)
                self.save_status()
                return True
        
        logger.warning("未检测到标注文件，请先完成图像标注")
        return False
    
    def step2_process_dataset(self):
        """步骤2: 处理数据集"""
        logger.info("=== 步骤2: 处理数据集 ===")
        
        if not self.status['steps']['annotation']['completed']:
            logger.error("请先完成图像标注")
            return False
        
        if self.status['steps']['dataset_processing']['completed']:
            logger.info("数据集处理已完成，跳过此步骤")
            return True
        
        try:
            logger.info("开始处理数据集...")
            
            # 运行数据集处理脚本
            result = subprocess.run([
                sys.executable, str(self.dataset_processor)
            ], capture_output=True, text=True, cwd=str(self.dataset_dir))
            
            if result.returncode == 0:
                logger.info("数据集处理完成")
                
                # 检查配置文件是否生成
                config_file = self.dataset_dir / "dataset_config.yaml"
                if config_file.exists():
                    self.status['steps']['dataset_processing']['completed'] = True
                    self.status['steps']['dataset_processing']['timestamp'] = datetime.now().isoformat()
                    
                    # 读取统计信息
                    stats_file = self.dataset_dir / "dataset_statistics.json"
                    if stats_file.exists():
                        with open(stats_file, 'r', encoding='utf-8') as f:
                            stats = json.load(f)
                            self.status['statistics'].update(stats)
                    
                    self.save_status()
                    return True
                else:
                    logger.error("数据集配置文件未生成")
                    return False
            else:
                logger.error(f"数据集处理失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"数据集处理过程中出现错误: {e}")
            return False
    
    def step3_train_model(self, model_size='n', epochs=50, batch_size=16):
        """步骤3: 训练模型"""
        logger.info("=== 步骤3: 训练模型 ===")
        
        if not self.status['steps']['dataset_processing']['completed']:
            logger.error("请先完成数据集处理")
            return False
        
        if self.status['steps']['model_training']['completed']:
            logger.info("模型训练已完成，跳过此步骤")
            return True
        
        try:
            logger.info(f"开始训练模型 (size={model_size}, epochs={epochs}, batch={batch_size})...")
            
            # 设置环境变量
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.dataset_dir)
            
            # 运行训练脚本
            result = subprocess.run([
                sys.executable, str(self.model_trainer)
            ], capture_output=True, text=True, cwd=str(self.dataset_dir), env=env)
            
            if result.returncode == 0:
                logger.info("模型训练完成")
                
                # 检查模型文件是否生成
                models_dir = self.dataset_dir / "models"
                model_files = list(models_dir.glob("best_*.pt"))
                
                if model_files:
                    best_model = model_files[0]
                    self.status['steps']['model_training']['completed'] = True
                    self.status['steps']['model_training']['timestamp'] = datetime.now().isoformat()
                    self.status['steps']['model_training']['model_path'] = str(best_model)
                    
                    # 获取模型大小
                    model_size_mb = best_model.stat().st_size / (1024 * 1024)
                    self.status['statistics']['model_size_mb'] = model_size_mb
                    
                    self.save_status()
                    return True
                else:
                    logger.error("训练完成但未找到模型文件")
                    return False
            else:
                logger.error(f"模型训练失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"模型训练过程中出现错误: {e}")
            return False
    
    def step4_inference(self, input_folder=None):
        """步骤4: 模型推理"""
        logger.info("=== 步骤4: 模型推理 ===")
        
        if not self.status['steps']['model_training']['completed']:
            logger.error("请先完成模型训练")
            return False
        
        try:
            logger.info("开始模型推理...")
            
            # 设置环境变量
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.dataset_dir)
            
            # 运行推理脚本
            result = subprocess.run([
                sys.executable, str(self.inference_script)
            ], capture_output=True, text=True, cwd=str(self.dataset_dir), env=env)
            
            if result.returncode == 0:
                logger.info("模型推理完成")
                
                # 检查推理结果
                results_dir = self.dataset_dir / "inference_results"
                results_file = results_dir / "inference_results.json"
                
                if results_file.exists():
                    with open(results_file, 'r', encoding='utf-8') as f:
                        inference_results = json.load(f)
                        
                    self.status['steps']['inference']['completed'] = True
                    self.status['steps']['inference']['timestamp'] = datetime.now().isoformat()
                    self.status['statistics']['inference_results'] = {
                        'total_processed': inference_results.get('total_images', 0),
                        'total_detections': inference_results.get('total_detections', 0),
                        'total_extracted': inference_results.get('total_extracted', 0)
                    }
                    
                    self.save_status()
                    return True
                else:
                    logger.error("推理完成但未找到结果文件")
                    return False
            else:
                logger.error(f"模型推理失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"模型推理过程中出现错误: {e}")
            return False
    
    def run_full_pipeline(self, model_size='n', epochs=50, batch_size=16):
        """运行完整Pipeline"""
        logger.info("开始运行完整Pipeline...")
        
        # 检查依赖
        if not self.check_dependencies():
            return False
        
        # 步骤1: 图像标注 (手动步骤)
        if not self.step1_annotation():
            logger.info("请先完成图像标注，然后重新运行Pipeline")
            return False
        
        # 步骤2: 处理数据集
        if not self.step2_process_dataset():
            return False
        
        # 步骤3: 训练模型
        if not self.step3_train_model(model_size, epochs, batch_size):
            return False
        
        # 步骤4: 模型推理
        if not self.step4_inference():
            return False
        
        logger.info("完整Pipeline执行成功!")
        self.print_summary()
        return True
    
    def print_summary(self):
        """打印摘要信息"""
        print("\n" + "="*50)
        print("1956 TI 数据集Pipeline摘要")
        print("="*50)
        
        stats = self.status.get('statistics', {})
        
        print(f"创建时间: {self.status['created_at']}")
        print(f"最后更新: {self.status['last_updated']}")
        print()
        
        print("步骤完成情况:")
        for step_name, step_info in self.status['steps'].items():
            status = "✅" if step_info['completed'] else "❌"
            timestamp = step_info.get('timestamp', 'N/A')
            print(f"  {status} {step_name}: {timestamp}")
        print()
        
        if stats:
            print("统计信息:")
            if 'num_annotations' in stats:
                print(f"  标注文件数: {stats['num_annotations']}")
            if 'total_images' in stats:
                print(f"  总图像数: {stats['total_images']}")
            if 'total_annotations' in stats:
                print(f"  总标注数: {stats['total_annotations']}")
            if 'model_size_mb' in stats:
                print(f"  模型大小: {stats['model_size_mb']:.2f} MB")
            if 'inference_results' in stats:
                ir = stats['inference_results']
                print(f"  推理处理数: {ir.get('total_processed', 0)}")
                print(f"  检测总数: {ir.get('total_detections', 0)}")
                print(f"  提取总数: {ir.get('total_extracted', 0)}")
        
        print("\n文件位置:")
        print(f"  数据集目录: {self.dataset_dir}")
        if self.status['steps']['model_training']['completed']:
            print(f"  最佳模型: {self.status['steps']['model_training']['model_path']}")
        if self.status['steps']['inference']['completed']:
            print(f"  推理结果: {self.dataset_dir / 'inference_results'}")

def main():
    parser = argparse.ArgumentParser(description='1956 TI 数据集Pipeline管理')
    parser.add_argument('--step', choices=['annotation', 'process', 'train', 'inference', 'full'], 
                       default='full', help='要执行的步骤')
    parser.add_argument('--model-size', choices=['n', 's', 'm', 'l', 'x'], default='n', 
                       help='YOLO模型大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--dataset-dir', default='/Users/zhaoye/Desktop/1956_TI_Dataset', 
                       help='数据集目录')
    
    args = parser.parse_args()
    
    # 创建Pipeline管理器
    manager = PipelineManager(args.dataset_dir)
    
    try:
        if args.step == 'annotation':
            manager.step1_annotation()
        elif args.step == 'process':
            manager.step2_process_dataset()
        elif args.step == 'train':
            manager.step3_train_model(args.model_size, args.epochs, args.batch_size)
        elif args.step == 'inference':
            manager.step4_inference()
        elif args.step == 'full':
            manager.run_full_pipeline(args.model_size, args.epochs, args.batch_size)
        
        manager.print_summary()
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.error(f"Pipeline执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

