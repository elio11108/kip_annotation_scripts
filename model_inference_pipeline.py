#!/usr/bin/env python3
"""
模型推理Pipeline
完整的模型预测、评估和可视化系统
使用训练集以外的数据（验证集和测试集）来评估模型性能
"""

import os
import sys
from pathlib import Path
import json
import logging
import datetime
import torch
from ultralytics import YOLO
import yaml
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Any
import shutil

# 设置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/Users/zhaoye/Desktop/1956_TI_Dataset/inference.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ModelInferencePipeline:
    def __init__(self, dataset_dir: str, model_path: str):
        self.dataset_dir = Path(dataset_dir)
        self.model_path = Path(model_path)
        self.results_dir = self.dataset_dir / "inference_results"
        self.optimized_data_dir = self.dataset_dir / "optimized_data"
        
        # 创建结果目录
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "predictions").mkdir(exist_ok=True)
        (self.results_dir / "visualizations").mkdir(exist_ok=True)
        (self.results_dir / "reports").mkdir(exist_ok=True)
        
        # 验证模型文件
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 加载模型
        self.model = None
        self.load_model()
        
        # 数据集信息
        self.dataset_splits = ['val', 'test']  # 使用验证集和测试集进行预测
        
    def load_model(self):
        """加载训练好的模型"""
        try:
            logger.info(f"🔄 加载模型: {self.model_path}")
            self.model = YOLO(str(self.model_path))
            
            # 获取模型信息
            model_size = self.model_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ 模型加载成功")
            logger.info(f"   模型大小: {model_size:.2f} MB")
            logger.info(f"   模型类型: YOLOv8n (超级优化版)")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def prepare_test_datasets(self) -> Dict[str, Dict]:
        """准备测试数据集信息"""
        logger.info("📊 准备测试数据集...")
        
        datasets_info = {}
        
        for split in self.dataset_splits:
            images_dir = self.optimized_data_dir / split / 'images'
            labels_dir = self.optimized_data_dir / split / 'labels'
            
            if not images_dir.exists():
                logger.warning(f"⚠️ {split}数据集不存在: {images_dir}")
                continue
            
            # 获取图像文件列表
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            
            datasets_info[split] = {
                'images_dir': images_dir,
                'labels_dir': labels_dir,
                'image_files': image_files,
                'count': len(image_files)
            }
            
            logger.info(f"   {split}集: {len(image_files)}张图像")
        
        total_images = sum(info['count'] for info in datasets_info.values())
        logger.info(f"📈 总测试图像: {total_images}张")
        
        return datasets_info
    
    def run_batch_inference(self, datasets_info: Dict[str, Dict]) -> Dict[str, Any]:
        """批量推理"""
        logger.info("🚀 开始批量推理...")
        
        all_results = {}
        inference_stats = {
            'total_images': 0,
            'total_predictions': 0,
            'average_confidence': 0,
            'processing_time': 0
        }
        
        start_time = datetime.datetime.now()
        
        for split_name, dataset_info in datasets_info.items():
            logger.info(f"🔍 处理{split_name}集...")
            
            split_results = []
            split_predictions = 0
            confidences = []
            
            # 创建分割结果目录
            split_results_dir = self.results_dir / "predictions" / split_name
            split_results_dir.mkdir(exist_ok=True)
            
            for i, image_file in enumerate(dataset_info['image_files']):
                try:
                    # 运行推理
                    results = self.model(str(image_file), conf=0.25, iou=0.45, verbose=False)
                    
                    # 处理结果
                    result = results[0]  # 单张图像结果
                    
                    # 提取预测信息
                    predictions = []
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标
                        confidences_batch = result.boxes.conf.cpu().numpy()  # 置信度
                        classes = result.boxes.cls.cpu().numpy()  # 类别
                        
                        for j, (box, conf, cls) in enumerate(zip(boxes, confidences_batch, classes)):
                            prediction = {
                                'box': box.tolist(),
                                'confidence': float(conf),
                                'class': int(cls),
                                'class_name': 'image'  # 我们只有一个类别
                            }
                            predictions.append(prediction)
                            confidences.append(float(conf))
                    
                    # 保存单张图像结果
                    image_result = {
                        'image_file': image_file.name,
                        'image_path': str(image_file),
                        'predictions': predictions,
                        'prediction_count': len(predictions)
                    }
                    split_results.append(image_result)
                    split_predictions += len(predictions)
                    
                    # 保存预测结果到JSON
                    result_file = split_results_dir / f"{image_file.stem}_predictions.json"
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(image_result, f, ensure_ascii=False, indent=2)
                    
                    # 进度显示
                    if (i + 1) % 10 == 0 or (i + 1) == len(dataset_info['image_files']):
                        logger.info(f"   进度: {i+1}/{len(dataset_info['image_files'])} "
                                  f"({(i+1)/len(dataset_info['image_files'])*100:.1f}%)")
                
                except Exception as e:
                    logger.error(f"处理图像 {image_file} 失败: {e}")
            
            # 保存分割汇总结果
            split_summary = {
                'split_name': split_name,
                'total_images': len(dataset_info['image_files']),
                'total_predictions': split_predictions,
                'average_predictions_per_image': split_predictions / len(dataset_info['image_files']) if dataset_info['image_files'] else 0,
                'average_confidence': np.mean(confidences) if confidences else 0,
                'results': split_results
            }
            
            all_results[split_name] = split_summary
            
            logger.info(f"✅ {split_name}集完成:")
            logger.info(f"   图像数: {len(dataset_info['image_files'])}")
            logger.info(f"   预测数: {split_predictions}")
            logger.info(f"   平均置信度: {split_summary['average_confidence']:.3f}")
        
        # 计算总体统计
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        total_images = sum(result['total_images'] for result in all_results.values())
        total_predictions = sum(result['total_predictions'] for result in all_results.values())
        all_confidences = []
        for result in all_results.values():
            for image_result in result['results']:
                for pred in image_result['predictions']:
                    all_confidences.append(pred['confidence'])
        
        inference_stats.update({
            'total_images': total_images,
            'total_predictions': total_predictions,
            'average_confidence': np.mean(all_confidences) if all_confidences else 0,
            'processing_time': processing_time,
            'images_per_second': total_images / processing_time if processing_time > 0 else 0
        })
        
        logger.info("🎉 批量推理完成!")
        logger.info(f"   总图像数: {total_images}")
        logger.info(f"   总预测数: {total_predictions}")
        logger.info(f"   处理时间: {processing_time:.1f}秒")
        logger.info(f"   处理速度: {inference_stats['images_per_second']:.1f}张/秒")
        
        return {
            'results': all_results,
            'statistics': inference_stats,
            'timestamp': end_time.isoformat()
        }
    
    def evaluate_model_performance(self, datasets_info: Dict[str, Dict]) -> Dict[str, Any]:
        """评估模型性能（与真实标签对比）"""
        logger.info("📊 评估模型性能...")
        
        evaluation_results = {}
        
        for split_name, dataset_info in datasets_info.items():
            logger.info(f"🔍 评估{split_name}集...")
            
            # 使用YOLO内置的验证功能
            try:
                # 创建临时配置文件
                temp_config = {
                    'path': str(self.optimized_data_dir.absolute()),
                    'train': 'train/images',
                    'val': f'{split_name}/images',
                    'test': 'test/images' if split_name == 'test' else f'{split_name}/images',
                    'nc': 1,
                    'names': ['image']
                }
                
                temp_config_file = self.results_dir / f'temp_{split_name}_config.yaml'
                with open(temp_config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(temp_config, f, default_flow_style=False, allow_unicode=True)
                
                # 运行验证
                metrics = self.model.val(
                    data=str(temp_config_file),
                    split=split_name if split_name in ['val', 'test'] else 'val',
                    conf=0.25,
                    iou=0.45,
                    save_json=True,
                    save_txt=False,
                    plots=True,
                    verbose=False
                )
                
                # 提取关键指标
                if hasattr(metrics, 'box'):
                    evaluation_results[split_name] = {
                        'mAP50': float(metrics.box.map50),
                        'mAP50_95': float(metrics.box.map),
                        'precision': float(metrics.box.mp),
                        'recall': float(metrics.box.mr),
                        'f1_score': float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr)) if (metrics.box.mp + metrics.box.mr) > 0 else 0
                    }
                    
                    logger.info(f"✅ {split_name}集评估结果:")
                    logger.info(f"   mAP50: {evaluation_results[split_name]['mAP50']:.4f}")
                    logger.info(f"   mAP50-95: {evaluation_results[split_name]['mAP50_95']:.4f}")
                    logger.info(f"   Precision: {evaluation_results[split_name]['precision']:.4f}")
                    logger.info(f"   Recall: {evaluation_results[split_name]['recall']:.4f}")
                    logger.info(f"   F1-Score: {evaluation_results[split_name]['f1_score']:.4f}")
                
                # 清理临时文件
                if temp_config_file.exists():
                    temp_config_file.unlink()
                    
            except Exception as e:
                logger.error(f"评估{split_name}集失败: {e}")
                evaluation_results[split_name] = {'error': str(e)}
        
        return evaluation_results
    
    def create_visualizations(self, inference_results: Dict[str, Any], datasets_info: Dict[str, Dict]) -> Dict[str, Path]:
        """创建可视化结果"""
        logger.info("🎨 创建可视化结果...")
        
        visualization_files = {}
        
        # 1. 创建预测示例图像
        self._create_prediction_samples(inference_results, datasets_info)
        
        # 2. 创建统计图表
        stats_file = self._create_statistics_plots(inference_results)
        visualization_files['statistics'] = stats_file
        
        # 3. 创建置信度分布图
        confidence_file = self._create_confidence_distribution(inference_results)
        visualization_files['confidence_distribution'] = confidence_file
        
        # 4. 创建性能对比图
        if len(inference_results['results']) > 1:
            comparison_file = self._create_performance_comparison(inference_results)
            visualization_files['performance_comparison'] = comparison_file
        
        logger.info("✅ 可视化结果创建完成")
        return visualization_files
    
    def _create_prediction_samples(self, inference_results: Dict[str, Any], datasets_info: Dict[str, Dict]):
        """创建预测示例图像"""
        logger.info("📸 创建预测示例图像...")
        
        samples_dir = self.results_dir / "visualizations" / "prediction_samples"
        samples_dir.mkdir(exist_ok=True)
        
        for split_name, split_results in inference_results['results'].items():
            split_samples_dir = samples_dir / split_name
            split_samples_dir.mkdir(exist_ok=True)
            
            # 选择前5个有预测结果的图像作为示例
            sample_count = 0
            max_samples = 5
            
            for image_result in split_results['results']:
                if sample_count >= max_samples:
                    break
                
                if image_result['prediction_count'] > 0:
                    self._draw_predictions_on_image(
                        image_result, 
                        split_samples_dir / f"sample_{sample_count+1}_{Path(image_result['image_file']).stem}.jpg"
                    )
                    sample_count += 1
    
    def _draw_predictions_on_image(self, image_result: Dict, output_path: Path):
        """在图像上绘制预测结果"""
        try:
            # 加载图像
            image = Image.open(image_result['image_path'])
            draw = ImageDraw.Draw(image)
            
            # 设置绘制参数
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
            
            for i, pred in enumerate(image_result['predictions']):
                box = pred['box']
                confidence = pred['confidence']
                color = colors[i % len(colors)]
                
                # 绘制边界框
                draw.rectangle(box, outline=color, width=3)
                
                # 绘制标签
                label = f"image: {confidence:.3f}"
                
                # 尝试使用系统字体，如果失败则使用默认字体
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                # 计算文本位置
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                text_x = box[0]
                text_y = box[1] - text_height - 5
                
                # 绘制文本背景
                draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                             fill=color, outline=color)
                
                # 绘制文本
                draw.text((text_x, text_y), label, fill='white', font=font)
            
            # 保存图像
            image.save(output_path, quality=95)
            
        except Exception as e:
            logger.error(f"绘制预测结果失败 {image_result['image_file']}: {e}")
    
    def _create_statistics_plots(self, inference_results: Dict[str, Any]) -> Path:
        """创建统计图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型推理统计分析', fontsize=16, fontweight='bold')
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 各数据集预测数量对比
        splits = list(inference_results['results'].keys())
        prediction_counts = [inference_results['results'][split]['total_predictions'] 
                           for split in splits]
        image_counts = [inference_results['results'][split]['total_images'] 
                       for split in splits]
        
        x = np.arange(len(splits))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, image_counts, width, label='图像数量', alpha=0.8)
        axes[0, 0].bar(x + width/2, prediction_counts, width, label='预测数量', alpha=0.8)
        axes[0, 0].set_xlabel('数据集')
        axes[0, 0].set_ylabel('数量')
        axes[0, 0].set_title('各数据集图像和预测数量对比')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(splits)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 平均置信度对比
        avg_confidences = [inference_results['results'][split]['average_confidence'] 
                          for split in splits]
        
        bars = axes[0, 1].bar(splits, avg_confidences, color=['skyblue', 'lightcoral'][:len(splits)])
        axes[0, 1].set_xlabel('数据集')
        axes[0, 1].set_ylabel('平均置信度')
        axes[0, 1].set_title('各数据集平均置信度对比')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 在柱子上添加数值标签
        for bar, conf in zip(bars, avg_confidences):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{conf:.3f}', ha='center', va='bottom')
        
        # 3. 每张图像平均预测数量
        avg_predictions = [inference_results['results'][split]['average_predictions_per_image'] 
                          for split in splits]
        
        axes[1, 0].bar(splits, avg_predictions, color=['lightgreen', 'orange'][:len(splits)])
        axes[1, 0].set_xlabel('数据集')
        axes[1, 0].set_ylabel('平均预测数/图像')
        axes[1, 0].set_title('每张图像平均预测数量')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 处理性能统计
        stats = inference_results['statistics']
        performance_data = {
            '总图像数': stats['total_images'],
            '总预测数': stats['total_predictions'],
            '处理时间(秒)': stats['processing_time'],
            '处理速度(张/秒)': stats['images_per_second']
        }
        
        # 创建性能表格
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        table_data = [[key, f"{value:.2f}" if isinstance(value, float) else str(value)] 
                     for key, value in performance_data.items()]
        
        table = axes[1, 1].table(cellText=table_data,
                               colLabels=['指标', '数值'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('处理性能统计')
        
        plt.tight_layout()
        
        # 保存图表
        stats_file = self.results_dir / "visualizations" / "inference_statistics.png"
        plt.savefig(stats_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return stats_file
    
    def _create_confidence_distribution(self, inference_results: Dict[str, Any]) -> Path:
        """创建置信度分布图"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('预测置信度分布分析', fontsize=16, fontweight='bold')
        
        # 收集所有置信度数据
        all_confidences = {}
        for split_name, split_results in inference_results['results'].items():
            confidences = []
            for image_result in split_results['results']:
                for pred in image_result['predictions']:
                    confidences.append(pred['confidence'])
            all_confidences[split_name] = confidences
        
        # 1. 置信度直方图
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        for i, (split_name, confidences) in enumerate(all_confidences.items()):
            if confidences:
                axes[0].hist(confidences, bins=20, alpha=0.7, 
                           label=f'{split_name}集 (n={len(confidences)})',
                           color=colors[i % len(colors)])
        
        axes[0].set_xlabel('置信度')
        axes[0].set_ylabel('频次')
        axes[0].set_title('置信度分布直方图')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 置信度箱线图
        if all_confidences:
            data_for_boxplot = []
            labels_for_boxplot = []
            for split_name, confidences in all_confidences.items():
                if confidences:
                    data_for_boxplot.append(confidences)
                    labels_for_boxplot.append(f'{split_name}集')
            
            if data_for_boxplot:
                axes[1].boxplot(data_for_boxplot, labels=labels_for_boxplot)
                axes[1].set_ylabel('置信度')
                axes[1].set_title('置信度分布箱线图')
                axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        confidence_file = self.results_dir / "visualizations" / "confidence_distribution.png"
        plt.savefig(confidence_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return confidence_file
    
    def _create_performance_comparison(self, inference_results: Dict[str, Any]) -> Path:
        """创建性能对比图"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('数据集性能对比', fontsize=16, fontweight='bold')
        
        splits = list(inference_results['results'].keys())
        metrics = ['total_images', 'total_predictions', 'average_confidence', 'average_predictions_per_image']
        metric_labels = ['图像数量', '预测数量', '平均置信度', '平均预测数/图像']
        
        # 标准化数据进行对比
        data_matrix = []
        for split in splits:
            split_data = []
            for metric in metrics:
                value = inference_results['results'][split][metric]
                split_data.append(value)
            data_matrix.append(split_data)
        
        # 创建热力图
        df = pd.DataFrame(data_matrix, index=splits, columns=metric_labels)
        
        # 标准化每列数据
        df_normalized = df.div(df.max(axis=0), axis=1)
        
        sns.heatmap(df_normalized, annot=True, cmap='YlOrRd', 
                   cbar_kws={'label': '标准化值'}, fmt='.3f')
        
        plt.title('各数据集性能指标对比（标准化）')
        plt.tight_layout()
        
        # 保存图表
        comparison_file = self.results_dir / "visualizations" / "performance_comparison.png"
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return comparison_file
    
    def generate_comprehensive_report(self, inference_results: Dict[str, Any], 
                                    evaluation_results: Dict[str, Any],
                                    visualization_files: Dict[str, Path]) -> Path:
        """生成综合报告"""
        logger.info("📋 生成综合推理报告...")
        
        report = {
            'inference_report': {
                'timestamp': datetime.datetime.now().isoformat(),
                'model_path': str(self.model_path),
                'model_size_mb': self.model_path.stat().st_size / (1024 * 1024),
                'datasets_used': list(inference_results['results'].keys())
            },
            'inference_results': inference_results,
            'evaluation_results': evaluation_results,
            'visualization_files': {k: str(v) for k, v in visualization_files.items()},
            'summary': self._create_summary(inference_results, evaluation_results),
            'recommendations': self._create_recommendations(inference_results, evaluation_results)
        }
        
        # 保存报告
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / "reports" / f"comprehensive_inference_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 创建简化的文本报告
        text_report_file = self.results_dir / "reports" / f"inference_summary_{timestamp}.txt"
        self._create_text_summary(report, text_report_file)
        
        logger.info(f"✅ 综合报告已生成: {report_file}")
        logger.info(f"📝 文本摘要已生成: {text_report_file}")
        
        return report_file
    
    def _create_summary(self, inference_results: Dict[str, Any], evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建结果摘要"""
        summary = {
            'total_images_processed': inference_results['statistics']['total_images'],
            'total_predictions_made': inference_results['statistics']['total_predictions'],
            'average_processing_speed': inference_results['statistics']['images_per_second'],
            'overall_average_confidence': inference_results['statistics']['average_confidence']
        }
        
        # 添加评估结果摘要
        if evaluation_results:
            best_map50 = 0
            best_split = None
            for split, metrics in evaluation_results.items():
                if 'mAP50' in metrics and metrics['mAP50'] > best_map50:
                    best_map50 = metrics['mAP50']
                    best_split = split
            
            summary.update({
                'best_performing_split': best_split,
                'best_mAP50': best_map50,
                'evaluation_available': True
            })
        else:
            summary['evaluation_available'] = False
        
        return summary
    
    def _create_recommendations(self, inference_results: Dict[str, Any], evaluation_results: Dict[str, Any]) -> List[str]:
        """创建建议"""
        recommendations = []
        
        # 基于置信度的建议
        avg_conf = inference_results['statistics']['average_confidence']
        if avg_conf > 0.8:
            recommendations.append("模型预测置信度很高，表现优秀")
        elif avg_conf > 0.6:
            recommendations.append("模型预测置信度良好，可以投入使用")
        else:
            recommendations.append("模型预测置信度较低，建议进一步优化或增加训练数据")
        
        # 基于处理速度的建议
        speed = inference_results['statistics']['images_per_second']
        if speed > 10:
            recommendations.append("处理速度很快，适合实时应用")
        elif speed > 5:
            recommendations.append("处理速度良好，适合批量处理")
        else:
            recommendations.append("处理速度较慢，考虑优化模型或硬件")
        
        # 基于评估结果的建议
        if evaluation_results:
            for split, metrics in evaluation_results.items():
                if 'mAP50' in metrics:
                    if metrics['mAP50'] > 0.9:
                        recommendations.append(f"{split}集上表现优异 (mAP50: {metrics['mAP50']:.3f})")
                    elif metrics['mAP50'] > 0.7:
                        recommendations.append(f"{split}集上表现良好 (mAP50: {metrics['mAP50']:.3f})")
                    else:
                        recommendations.append(f"{split}集上表现需要改进 (mAP50: {metrics['mAP50']:.3f})")
        
        recommendations.append("建议在实际应用场景中进一步测试模型性能")
        
        return recommendations
    
    def _create_text_summary(self, report: Dict[str, Any], output_file: Path):
        """创建文本摘要"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("模型推理综合报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 基本信息
            f.write("📋 基本信息\n")
            f.write("-" * 30 + "\n")
            f.write(f"模型路径: {report['inference_report']['model_path']}\n")
            f.write(f"模型大小: {report['inference_report']['model_size_mb']:.2f} MB\n")
            f.write(f"处理时间: {report['inference_report']['timestamp']}\n")
            f.write(f"测试数据集: {', '.join(report['inference_report']['datasets_used'])}\n\n")
            
            # 推理结果
            f.write("🚀 推理结果\n")
            f.write("-" * 30 + "\n")
            summary = report['summary']
            f.write(f"总处理图像: {summary['total_images_processed']} 张\n")
            f.write(f"总预测数量: {summary['total_predictions_made']} 个\n")
            f.write(f"处理速度: {summary['average_processing_speed']:.2f} 张/秒\n")
            f.write(f"平均置信度: {summary['overall_average_confidence']:.3f}\n\n")
            
            # 各数据集详情
            f.write("📊 各数据集详情\n")
            f.write("-" * 30 + "\n")
            for split_name, split_results in report['inference_results']['results'].items():
                f.write(f"{split_name}集:\n")
                f.write(f"  图像数量: {split_results['total_images']}\n")
                f.write(f"  预测数量: {split_results['total_predictions']}\n")
                f.write(f"  平均置信度: {split_results['average_confidence']:.3f}\n")
                f.write(f"  平均预测数/图像: {split_results['average_predictions_per_image']:.2f}\n\n")
            
            # 评估结果
            if report['evaluation_results']:
                f.write("📈 性能评估\n")
                f.write("-" * 30 + "\n")
                for split_name, metrics in report['evaluation_results'].items():
                    if 'mAP50' in metrics:
                        f.write(f"{split_name}集:\n")
                        f.write(f"  mAP50: {metrics['mAP50']:.4f}\n")
                        f.write(f"  mAP50-95: {metrics['mAP50_95']:.4f}\n")
                        f.write(f"  Precision: {metrics['precision']:.4f}\n")
                        f.write(f"  Recall: {metrics['recall']:.4f}\n")
                        f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n\n")
            
            # 建议
            f.write("💡 建议\n")
            f.write("-" * 30 + "\n")
            for i, recommendation in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {recommendation}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("报告生成完成\n")
            f.write("=" * 60 + "\n")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """运行完整的推理pipeline"""
        logger.info("🚀 启动完整的模型推理Pipeline")
        logger.info("=" * 60)
        
        try:
            # 1. 准备测试数据
            datasets_info = self.prepare_test_datasets()
            
            # 2. 批量推理
            inference_results = self.run_batch_inference(datasets_info)
            
            # 3. 性能评估
            evaluation_results = self.evaluate_model_performance(datasets_info)
            
            # 4. 创建可视化
            visualization_files = self.create_visualizations(inference_results, datasets_info)
            
            # 5. 生成综合报告
            report_file = self.generate_comprehensive_report(
                inference_results, evaluation_results, visualization_files
            )
            
            logger.info("🎉 完整推理Pipeline执行成功!")
            logger.info(f"📋 综合报告: {report_file}")
            logger.info(f"📁 结果目录: {self.results_dir}")
            
            return {
                'success': True,
                'report_file': report_file,
                'results_directory': self.results_dir,
                'inference_results': inference_results,
                'evaluation_results': evaluation_results,
                'visualization_files': visualization_files
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline执行失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """主函数"""
    dataset_dir = "/Users/zhaoye/Desktop/1956_TI_Dataset"
    model_path = "/Users/zhaoye/Desktop/1956_TI_Dataset/models/super_optimized_best_yolov8n.pt"
    
    print("🔍 模型推理Pipeline")
    print("=" * 50)
    print(f"📊 数据集目录: {dataset_dir}")
    print(f"🎯 模型路径: {model_path}")
    print("📈 使用数据集: 验证集 + 测试集")
    print("🎨 包含功能: 批量推理 + 性能评估 + 可视化 + 综合报告")
    print("=" * 50)
    
    try:
        # 创建推理pipeline
        pipeline = ModelInferencePipeline(dataset_dir, model_path)
        
        # 运行完整pipeline
        results = pipeline.run_complete_pipeline()
        
        if results['success']:
            print("\n🎉 推理Pipeline执行成功!")
            print("=" * 50)
            print(f"📋 综合报告: {results['report_file']}")
            print(f"📁 结果目录: {results['results_directory']}")
            print("\n📊 主要结果:")
            
            stats = results['inference_results']['statistics']
            print(f"  总处理图像: {stats['total_images']} 张")
            print(f"  总预测数量: {stats['total_predictions']} 个")
            print(f"  处理速度: {stats['images_per_second']:.2f} 张/秒")
            print(f"  平均置信度: {stats['average_confidence']:.3f}")
            
            if results['evaluation_results']:
                print("\n📈 性能评估:")
                for split, metrics in results['evaluation_results'].items():
                    if 'mAP50' in metrics:
                        print(f"  {split}集 mAP50: {metrics['mAP50']:.4f}")
            
            print(f"\n📸 可视化文件:")
            for name, path in results['visualization_files'].items():
                print(f"  {name}: {path}")
            
            print("\n💡 下一步建议:")
            print("1. 查看综合报告了解详细结果")
            print("2. 检查可视化图像验证预测质量")
            print("3. 根据性能指标决定是否部署模型")
            print("4. 在新的真实数据上测试模型")
        else:
            print(f"❌ Pipeline执行失败: {results['error']}")
    
    except Exception as e:
        logger.error(f"主程序执行失败: {e}")
        print(f"❌ 执行失败: {e}")


if __name__ == "__main__":
    main()
