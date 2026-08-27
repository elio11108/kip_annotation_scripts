#!/usr/bin/env python3
"""
1962 TI数据集图像提取器
使用训练好的模型对1962 TI文件夹中的所有图像进行处理
提取检测到的图像区域并保存到"1962 TI_Extracted_Dataset"文件夹
专为大规模数据集处理优化
"""

import os
import sys
from pathlib import Path
import json
import logging
import datetime
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import shutil
from typing import Dict, List, Tuple, Any
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/Users/zhaoye/Desktop/1956_TI_Dataset/ti_1962_extraction.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class TI1962DatasetExtractor:
    def __init__(self, model_path: str, source_dir: str, output_dir: str):
        self.model_path = Path(model_path)
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # 验证输入
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        if not self.source_dir.exists():
            raise FileNotFoundError(f"源目录不存在: {self.source_dir}")
        
        # 创建输出目录结构（优化的大数据集结构）
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "extracted_images").mkdir(exist_ok=True)
        (self.output_dir / "high_confidence").mkdir(exist_ok=True)    # ≥0.8
        (self.output_dir / "good_confidence").mkdir(exist_ok=True)    # ≥0.6
        (self.output_dir / "medium_confidence").mkdir(exist_ok=True)  # ≥0.4
        (self.output_dir / "low_confidence").mkdir(exist_ok=True)     # ≥0.25
        (self.output_dir / "sample_annotations").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        (self.output_dir / "statistics").mkdir(exist_ok=True)
        
        # 加载模型
        self.model = None
        self.load_model()
        
        # 置信度阈值设置（更细化的分级）
        self.confidence_thresholds = {
            'high': 0.8,      # 高置信度
            'good': 0.6,      # 良好置信度
            'medium': 0.4,    # 中等置信度
            'low': 0.25       # 低置信度（检测阈值）
        }
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'total_detections': 0,
            'confidence_distribution': {'high': 0, 'good': 0, 'medium': 0, 'low': 0},
            'total_extracted': 0,
            'processing_time': 0,
            'failed_images': [],
            'dataset_quality_score': 0,
            'size_distribution': {'small': 0, 'medium': 0, 'large': 0, 'extra_large': 0},
            'processing_speed': 0
        }
        
        # 处理进度跟踪
        self.progress = {
            'current_batch': 0,
            'total_batches': 0,
            'processed_images': 0,
            'start_time': None
        }
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            logger.info(f"🔄 加载模型: {self.model_path}")
            self.model = YOLO(str(self.model_path))
            
            model_size = self.model_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ 模型加载成功 ({model_size:.2f} MB)")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def get_image_files(self) -> List[Path]:
        """获取源目录中的所有图像文件"""
        logger.info(f"📂 扫描源目录: {self.source_dir}")
        
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.source_dir.glob(f'*{ext}'))
            image_files.extend(self.source_dir.glob(f'*{ext.upper()}'))
        
        # 排序以确保处理顺序一致
        image_files.sort()
        
        logger.info(f"📊 找到 {len(image_files)} 个图像文件")
        return image_files
    
    def categorize_by_confidence(self, confidence: float) -> str:
        """根据置信度分类（细化分级）"""
        if confidence >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence >= self.confidence_thresholds['good']:
            return 'good'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def categorize_by_size(self, area: int) -> str:
        """根据区域大小分类"""
        if area < 25000:
            return 'small'
        elif area < 100000:
            return 'medium'
        elif area < 500000:
            return 'large'
        else:
            return 'extra_large'
    
    def process_single_image(self, image_path: Path) -> Dict[str, Any]:
        """处理单个图像"""
        try:
            # 运行模型推理
            results = self.model(str(image_path), 
                               conf=self.confidence_thresholds['low'], 
                               iou=0.45, 
                               verbose=False)
            result = results[0]
            
            # 加载原图像
            original_image = cv2.imread(str(image_path))
            if original_image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            original_pil = Image.open(image_path)
            
            detections = []
            extracted_images = []
            confidence_counts = {'high': 0, 'good': 0, 'medium': 0, 'low': 0}
            size_counts = {'small': 0, 'medium': 0, 'large': 0, 'extra_large': 0}
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                # 创建带标注的原图副本
                annotated_image = original_image.copy()
                
                for i, (box, conf) in enumerate(zip(boxes, confidences)):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 确保坐标在图像范围内
                    height, width = original_image.shape[:2]
                    x1 = max(0, min(x1, width-1))
                    y1 = max(0, min(y1, height-1))
                    x2 = max(0, min(x2, width-1))
                    y2 = max(0, min(y2, height-1))
                    
                    # 检查边界框是否有效
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # 确保提取区域有最小尺寸
                    min_size = 30
                    if (x2 - x1) < min_size or (y2 - y1) < min_size:
                        continue
                    
                    # 计算区域面积
                    area = (x2 - x1) * (y2 - y1)
                    
                    # 提取图像区域
                    extracted_region = original_pil.crop((x1, y1, x2, y2))
                    
                    # 根据置信度和大小分类
                    confidence_category = self.categorize_by_confidence(conf)
                    size_category = self.categorize_by_size(area)
                    
                    confidence_counts[confidence_category] += 1
                    size_counts[size_category] += 1
                    
                    # 生成提取图像的文件名
                    base_name = image_path.stem
                    extracted_filename = f"{base_name}_extracted_{i+1}_{confidence_category}_conf{conf:.3f}.jpg"
                    
                    # 保存到对应的置信度文件夹
                    category_dir = self.output_dir / f"{confidence_category}_confidence"
                    extracted_path = category_dir / extracted_filename
                    
                    # 同时保存到总的extracted_images文件夹
                    all_extracted_path = self.output_dir / "extracted_images" / extracted_filename
                    
                    # 保存提取的图像（高质量）
                    extracted_region.save(extracted_path, 'JPEG', quality=95)
                    extracted_region.save(all_extracted_path, 'JPEG', quality=95)
                    
                    # 记录检测信息
                    detection_info = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'confidence_category': confidence_category,
                        'size_category': size_category,
                        'area': area,
                        'extracted_file': extracted_filename,
                        'width': x2 - x1,
                        'height': y2 - y1
                    }
                    detections.append(detection_info)
                    extracted_images.append(extracted_path)
                    
                    # 在原图上绘制边界框（颜色根据置信度）
                    color_map = {
                        'high': (0, 255, 0),      # 绿色
                        'good': (0, 255, 255),    # 黄色
                        'medium': (0, 165, 255),  # 橙色
                        'low': (0, 0, 255)        # 红色
                    }
                    color = color_map[confidence_category]
                    
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    # 添加置信度标签
                    label = f"{confidence_category.upper()}: {conf:.3f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 8), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_image, label, (x1, y1 - 4), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 保存带标注的原图（仅当有检测结果时，且限制数量）
                if detections and len(detections) > 0:
                    # 只保存前100个带标注的图像作为样本
                    if self.stats['total_processed'] < 100:
                        annotated_filename = f"{image_path.stem}_annotated.jpg"
                        annotated_path = self.output_dir / "sample_annotations" / annotated_filename
                        cv2.imwrite(str(annotated_path), annotated_image, 
                                   [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # 更新统计
            for category, count in confidence_counts.items():
                self.stats['confidence_distribution'][category] += count
            for category, count in size_counts.items():
                self.stats['size_distribution'][category] += count
            
            # 返回处理结果
            return {
                'success': True,
                'original_file': image_path.name,
                'detections_count': len(detections),
                'confidence_distribution': confidence_counts,
                'size_distribution': size_counts,
                'detections': detections,
                'extracted_files': [str(p) for p in extracted_images]
            }
            
        except Exception as e:
            logger.error(f"处理图像 {image_path} 失败: {e}")
            return {
                'success': False,
                'original_file': image_path.name,
                'error': str(e)
            }
    
    def process_batch(self, image_files: List[Path], batch_size: int = 50) -> Dict[str, Any]:
        """批量处理图像（优化大数据集处理）"""
        logger.info(f"🚀 开始批量处理 {len(image_files)} 个图像文件")
        logger.info(f"📦 批次大小: {batch_size}")
        logger.info(f"🎯 置信度阈值: 高≥{self.confidence_thresholds['high']}, 良好≥{self.confidence_thresholds['good']}, 中≥{self.confidence_thresholds['medium']}, 低≥{self.confidence_thresholds['low']}")
        
        start_time = datetime.datetime.now()
        self.progress['start_time'] = start_time
        all_results = []
        
        # 计算总批次数
        total_batches = (len(image_files) + batch_size - 1) // batch_size
        self.progress['total_batches'] = total_batches
        
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            self.progress['current_batch'] = batch_num
            
            batch_start_time = time.time()
            
            logger.info(f"📦 处理批次 {batch_num}/{total_batches} ({len(batch_files)} 个文件)")
            
            batch_results = []
            for j, image_file in enumerate(batch_files):
                result = self.process_single_image(image_file)
                batch_results.append(result)
                all_results.append(result)
                
                # 更新统计
                self.stats['total_processed'] += 1
                self.progress['processed_images'] = self.stats['total_processed']
                
                if result['success']:
                    self.stats['total_detections'] += result['detections_count']
                    self.stats['total_extracted'] += result['detections_count']
                else:
                    self.stats['failed_images'].append(result['original_file'])
                
                # 显示进度（每10个文件或批次结束）
                if (j + 1) % 10 == 0 or (j + 1) == len(batch_files):
                    overall_progress = ((i + j + 1) / len(image_files)) * 100
                    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                    avg_speed = self.stats['total_processed'] / elapsed_time if elapsed_time > 0 else 0
                    
                    logger.info(f"   批次进度: {j+1}/{len(batch_files)} "
                              f"(总进度: {overall_progress:.1f}%, 速度: {avg_speed:.2f}张/秒)")
            
            # 批次完成统计
            batch_detections = sum(r['detections_count'] for r in batch_results if r['success'])
            batch_failures = sum(1 for r in batch_results if not r['success'])
            batch_time = time.time() - batch_start_time
            
            logger.info(f"✅ 批次 {batch_num} 完成: "
                       f"{len(batch_files) - batch_failures} 成功, "
                       f"{batch_failures} 失败, "
                       f"{batch_detections} 个检测, "
                       f"用时: {batch_time:.1f}秒")
            
            # 每50个批次保存中间统计
            if batch_num % 10 == 0:
                self._save_intermediate_stats(batch_num, total_batches)
        
        # 计算总体统计
        end_time = datetime.datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        self.stats['processing_speed'] = len(image_files) / self.stats['processing_time'] if self.stats['processing_time'] > 0 else 0
        
        success_count = sum(1 for r in all_results if r['success'])
        failure_count = len(all_results) - success_count
        
        # 计算数据集质量分数（细化计算）
        total_detections = self.stats['total_detections']
        if total_detections > 0:
            high_ratio = self.stats['confidence_distribution']['high'] / total_detections
            good_ratio = self.stats['confidence_distribution']['good'] / total_detections
            medium_ratio = self.stats['confidence_distribution']['medium'] / total_detections
            low_ratio = self.stats['confidence_distribution']['low'] / total_detections
            
            # 质量分数计算（高置信度权重更大）
            self.stats['dataset_quality_score'] = (
                high_ratio * 1.0 + good_ratio * 0.8 + medium_ratio * 0.5 + low_ratio * 0.2
            ) * 100
        
        logger.info("🎉 批量处理完成!")
        logger.info(f"📊 最终统计:")
        logger.info(f"   总文件数: {len(image_files)}")
        logger.info(f"   成功处理: {success_count} ({success_count/len(image_files)*100:.1f}%)")
        logger.info(f"   处理失败: {failure_count}")
        logger.info(f"   总检测数: {self.stats['total_detections']}")
        logger.info(f"   高置信度: {self.stats['confidence_distribution']['high']} 个")
        logger.info(f"   良好置信度: {self.stats['confidence_distribution']['good']} 个")
        logger.info(f"   中等置信度: {self.stats['confidence_distribution']['medium']} 个")
        logger.info(f"   低置信度: {self.stats['confidence_distribution']['low']} 个")
        logger.info(f"   数据集质量分数: {self.stats['dataset_quality_score']:.1f}/100")
        logger.info(f"   总处理时间: {self.stats['processing_time']:.1f} 秒 ({self.stats['processing_time']/60:.1f} 分钟)")
        logger.info(f"   处理速度: {self.stats['processing_speed']:.2f} 张/秒")
        
        return {
            'results': all_results,
            'statistics': self.stats,
            'timestamp': end_time.isoformat()
        }
    
    def _save_intermediate_stats(self, current_batch: int, total_batches: int):
        """保存中间统计信息"""
        intermediate_stats = {
            'progress': {
                'current_batch': current_batch,
                'total_batches': total_batches,
                'completion_percentage': (current_batch / total_batches) * 100,
                'processed_images': self.stats['total_processed']
            },
            'current_stats': self.stats.copy(),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        stats_file = self.output_dir / "statistics" / f"intermediate_stats_batch_{current_batch}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(intermediate_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 中间统计已保存: batch_{current_batch}")
    
    def create_comprehensive_dataset_metadata(self, processing_results: Dict[str, Any]) -> Path:
        """创建综合数据集元数据"""
        logger.info("📋 创建综合数据集元数据...")
        
        # 分析数据集特征
        all_areas = []
        all_confidences = []
        all_widths = []
        all_heights = []
        
        for result in processing_results['results']:
            if result['success']:
                for detection in result['detections']:
                    all_areas.append(detection['area'])
                    all_confidences.append(detection['confidence'])
                    all_widths.append(detection['width'])
                    all_heights.append(detection['height'])
        
        # 创建综合元数据
        metadata = {
            'dataset_info': {
                'name': '1962 TI Extracted Dataset',
                'description': '从1962 TI文档中提取的大规模图像数据集',
                'created_date': datetime.datetime.now().isoformat(),
                'source_directory': str(self.source_dir),
                'extraction_model': str(self.model_path),
                'model_size_mb': self.model_path.stat().st_size / (1024 * 1024),
                'dataset_scale': 'large'
            },
            'processing_performance': {
                'total_processing_time_minutes': self.stats['processing_time'] / 60,
                'processing_speed_images_per_second': self.stats['processing_speed'],
                'success_rate_percentage': (self.stats['total_processed'] - len(self.stats['failed_images'])) / self.stats['total_processed'] * 100 if self.stats['total_processed'] > 0 else 0
            },
            'dataset_statistics': {
                'total_source_images': self.stats['total_processed'],
                'total_extracted_images': self.stats['total_extracted'],
                'average_detections_per_source': self.stats['total_detections'] / self.stats['total_processed'] if self.stats['total_processed'] > 0 else 0,
                'quality_score': self.stats['dataset_quality_score']
            },
            'confidence_distribution': {},
            'size_distribution': {},
            'image_characteristics': {},
            'quality_analysis': self._analyze_dataset_quality(),
            'usage_recommendations': self._generate_usage_recommendations()
        }
        
        # 置信度分布
        total_detections = self.stats['total_detections']
        for level, count in self.stats['confidence_distribution'].items():
            threshold_map = {'high': '≥0.8', 'good': '≥0.6', 'medium': '≥0.4', 'low': '≥0.25'}
            metadata['confidence_distribution'][f'{level}_confidence'] = {
                'count': count,
                'threshold': threshold_map[level],
                'percentage': count / total_detections * 100 if total_detections > 0 else 0
            }
        
        # 尺寸分布
        for size, count in self.stats['size_distribution'].items():
            metadata['size_distribution'][f'{size}_size'] = {
                'count': count,
                'percentage': count / total_detections * 100 if total_detections > 0 else 0
            }
        
        # 图像特征统计
        if all_areas and all_confidences:
            metadata['image_characteristics'] = {
                'area_stats': {
                    'mean': float(np.mean(all_areas)),
                    'std': float(np.std(all_areas)),
                    'min': float(np.min(all_areas)),
                    'max': float(np.max(all_areas)),
                    'median': float(np.median(all_areas))
                },
                'confidence_stats': {
                    'mean': float(np.mean(all_confidences)),
                    'std': float(np.std(all_confidences)),
                    'min': float(np.min(all_confidences)),
                    'max': float(np.max(all_confidences)),
                    'median': float(np.median(all_confidences))
                },
                'dimension_stats': {
                    'width_mean': float(np.mean(all_widths)),
                    'height_mean': float(np.mean(all_heights)),
                    'width_range': [float(np.min(all_widths)), float(np.max(all_widths))],
                    'height_range': [float(np.min(all_heights)), float(np.max(all_heights))]
                }
            }
        
        # 保存元数据
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_file = self.output_dir / "metadata" / f"comprehensive_metadata_{timestamp}.json"
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 创建详细的README文件
        readme_file = self.output_dir / "README.md"
        self._create_comprehensive_readme(metadata, readme_file)
        
        # 创建数据集统计图表数据
        self._create_statistics_data(metadata)
        
        logger.info(f"✅ 综合数据集元数据已生成: {metadata_file}")
        logger.info(f"📝 详细README文件已生成: {readme_file}")
        
        return metadata_file
    
    def _analyze_dataset_quality(self) -> Dict[str, Any]:
        """分析数据集质量"""
        total = self.stats['total_detections']
        if total == 0:
            return {'overall_quality': 'empty', 'quality_rating': 0}
        
        high_ratio = self.stats['confidence_distribution']['high'] / total
        good_ratio = self.stats['confidence_distribution']['good'] / total
        high_good_ratio = (high_ratio + good_ratio)
        
        analysis = {
            'high_confidence_ratio': high_ratio,
            'good_confidence_ratio': good_ratio,
            'high_good_combined_ratio': high_good_ratio,
            'total_useful_ratio': high_good_ratio
        }
        
        # 质量评级
        if high_good_ratio > 0.7:
            analysis['overall_quality'] = 'excellent'
            analysis['quality_rating'] = 5
        elif high_good_ratio > 0.5:
            analysis['overall_quality'] = 'good'
            analysis['quality_rating'] = 4
        elif high_good_ratio > 0.3:
            analysis['overall_quality'] = 'fair'
            analysis['quality_rating'] = 3
        else:
            analysis['overall_quality'] = 'poor'
            analysis['quality_rating'] = 2
        
        return analysis
    
    def _generate_usage_recommendations(self) -> List[str]:
        """生成使用建议"""
        recommendations = []
        
        total = self.stats['total_detections']
        if total == 0:
            return ["数据集为空，无法提供使用建议"]
        
        high_ratio = self.stats['confidence_distribution']['high'] / total
        good_ratio = self.stats['confidence_distribution']['good'] / total
        high_good_ratio = high_ratio + good_ratio
        
        # 基于质量的建议
        if high_good_ratio > 0.7:
            recommendations.extend([
                "数据集质量优秀，可直接用于生产环境训练",
                "高置信度和良好置信度图像可作为黄金标准数据集",
                "适合用于基准测试和模型评估"
            ])
        elif high_good_ratio > 0.5:
            recommendations.extend([
                "数据集质量良好，建议结合少量人工筛选",
                "高置信度图像可直接使用，良好置信度图像建议抽检"
            ])
        else:
            recommendations.extend([
                "建议对高置信度和良好置信度图像进行人工审核",
                "可作为弱监督学习或半监督学习的数据源"
            ])
        
        # 基于数据量的建议
        if total > 1000:
            recommendations.append("大规模数据集，适合深度学习模型训练")
        elif total > 500:
            recommendations.append("中等规模数据集，适合迁移学习和微调")
        else:
            recommendations.append("小规模数据集，建议结合数据增强技术")
        
        # 通用建议
        recommendations.extend([
            "建议根据具体应用场景选择合适的置信度等级",
            "可使用sample_annotations文件夹进行质量验证",
            "不同置信度等级可用于不同的训练策略",
            "建议定期更新和扩充数据集"
        ])
        
        return recommendations
    
    def _create_comprehensive_readme(self, metadata: Dict[str, Any], readme_file: Path):
        """创建综合README文件"""
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write("# 1962 TI Extracted Dataset\n\n")
            
            # 数据集概览
            f.write("## 数据集概览\n\n")
            f.write(f"**数据集名称**: {metadata['dataset_info']['name']}\n\n")
            f.write(f"**描述**: {metadata['dataset_info']['description']}\n\n")
            f.write(f"**规模**: 大规模数据集 ({metadata['dataset_statistics']['total_source_images']} 源图像 → {metadata['dataset_statistics']['total_extracted_images']} 提取图像)\n\n")
            f.write(f"**创建时间**: {metadata['dataset_info']['created_date']}\n\n")
            f.write(f"**提取模型**: {Path(metadata['dataset_info']['extraction_model']).name} ({metadata['dataset_info']['model_size_mb']:.2f} MB)\n\n")
            
            # 处理性能
            f.write("## 处理性能\n\n")
            perf = metadata['processing_performance']
            f.write(f"- **处理时间**: {perf['total_processing_time_minutes']:.1f} 分钟\n")
            f.write(f"- **处理速度**: {perf['processing_speed_images_per_second']:.2f} 张/秒\n")
            f.write(f"- **成功率**: {perf['success_rate_percentage']:.1f}%\n\n")
            
            # 数据集统计
            f.write("## 数据集统计\n\n")
            stats = metadata['dataset_statistics']
            f.write(f"- **源图像数量**: {stats['total_source_images']:,} 张\n")
            f.write(f"- **提取图像数量**: {stats['total_extracted_images']:,} 张\n")
            f.write(f"- **平均检测数/源图像**: {stats['average_detections_per_source']:.2f}\n")
            f.write(f"- **数据集质量分数**: {stats['quality_score']:.1f}/100\n\n")
            
            # 质量分析
            quality = metadata['quality_analysis']
            f.write("## 质量分析\n\n")
            f.write(f"- **整体质量**: {quality['overall_quality'].upper()}\n")
            f.write(f"- **质量评级**: {'★' * quality['quality_rating']}{'☆' * (5 - quality['quality_rating'])} ({quality['quality_rating']}/5)\n")
            f.write(f"- **高质量图像比例**: {quality['high_good_combined_ratio']*100:.1f}%\n\n")
            
            # 置信度分布
            f.write("## 置信度分布\n\n")
            f.write("| 置信度等级 | 数量 | 阈值 | 占比 |\n")
            f.write("|-----------|------|------|------|\n")
            conf_names = {'high_confidence': '高', 'good_confidence': '良好', 'medium_confidence': '中等', 'low_confidence': '低'}
            for level, info in metadata['confidence_distribution'].items():
                level_name = conf_names.get(level, level)
                f.write(f"| {level_name} | {info['count']:,} | {info['threshold']} | {info['percentage']:.1f}% |\n")
            f.write("\n")
            
            # 尺寸分布
            f.write("## 尺寸分布\n\n")
            f.write("| 尺寸类别 | 数量 | 占比 |\n")
            f.write("|---------|------|------|\n")
            size_names = {'small_size': '小', 'medium_size': '中', 'large_size': '大', 'extra_large_size': '超大'}
            for size, info in metadata['size_distribution'].items():
                size_name = size_names.get(size, size)
                f.write(f"| {size_name} | {info['count']:,} | {info['percentage']:.1f}% |\n")
            f.write("\n")
            
            # 文件夹结构
            f.write("## 文件夹结构\n\n")
            f.write("```\n")
            f.write("1962 TI_Extracted_Dataset/\n")
            f.write("├── extracted_images/          # 所有提取的图像\n")
            f.write("├── high_confidence/           # 高置信度图像 (≥0.8)\n")
            f.write("├── good_confidence/           # 良好置信度图像 (≥0.6)\n")
            f.write("├── medium_confidence/         # 中等置信度图像 (≥0.4)\n")
            f.write("├── low_confidence/            # 低置信度图像 (≥0.25)\n")
            f.write("├── sample_annotations/        # 前100个带标注的示例图像\n")
            f.write("├── metadata/                  # 数据集元数据和统计信息\n")
            f.write("├── statistics/                # 处理过程中的统计数据\n")
            f.write("└── README.md                  # 此文件\n")
            f.write("```\n\n")
            
            # 使用建议
            f.write("## 使用建议\n\n")
            for i, recommendation in enumerate(metadata['usage_recommendations'], 1):
                f.write(f"{i}. {recommendation}\n")
            f.write("\n")
            
            # 图像特征
            if 'area_stats' in metadata['image_characteristics']:
                f.write("## 图像特征统计\n\n")
                area_stats = metadata['image_characteristics']['area_stats']
                conf_stats = metadata['image_characteristics']['confidence_stats']
                dim_stats = metadata['image_characteristics']['dimension_stats']
                
                f.write("### 区域大小统计\n")
                f.write(f"- **平均面积**: {area_stats['mean']:,.0f} 像素²\n")
                f.write(f"- **面积范围**: {area_stats['min']:,.0f} - {area_stats['max']:,.0f} 像素²\n")
                f.write(f"- **中位数**: {area_stats['median']:,.0f} 像素²\n\n")
                
                f.write("### 置信度统计\n")
                f.write(f"- **平均置信度**: {conf_stats['mean']:.3f}\n")
                f.write(f"- **置信度范围**: {conf_stats['min']:.3f} - {conf_stats['max']:.3f}\n")
                f.write(f"- **中位数**: {conf_stats['median']:.3f}\n\n")
                
                f.write("### 尺寸统计\n")
                f.write(f"- **平均宽度**: {dim_stats['width_mean']:.0f} 像素\n")
                f.write(f"- **平均高度**: {dim_stats['height_mean']:.0f} 像素\n")
                f.write(f"- **宽度范围**: {dim_stats['width_range'][0]:.0f} - {dim_stats['width_range'][1]:.0f} 像素\n")
                f.write(f"- **高度范围**: {dim_stats['height_range'][0]:.0f} - {dim_stats['height_range'][1]:.0f} 像素\n\n")
            
            # 技术说明
            f.write("## 技术说明\n\n")
            f.write("### 置信度分级\n")
            f.write("- **高置信度 (≥0.8)**: 模型非常确信的检测结果，推荐直接使用\n")
            f.write("- **良好置信度 (≥0.6)**: 模型较为确信的检测结果，建议抽检后使用\n")
            f.write("- **中等置信度 (≥0.4)**: 模型中等确信的检测结果，建议人工审核\n")
            f.write("- **低置信度 (≥0.25)**: 模型不太确信的检测结果，需要仔细审核\n\n")
            
            f.write("### 文件命名规则\n")
            f.write("提取的图像文件命名格式：`原文件名_extracted_序号_置信度等级_conf置信度值.jpg`\n\n")
            f.write("示例：`1962_Issue_1_1.pdf_001_fig1_extracted_1_high_conf0.876.jpg`\n\n")
            
            # 注意事项
            f.write("## 注意事项\n\n")
            f.write("1. 这是一个大规模数据集，建议根据具体需求选择合适的置信度等级\n")
            f.write("2. sample_annotations文件夹包含前100个带标注的原图，可用于快速质量评估\n")
            f.write("3. statistics文件夹包含处理过程中的详细统计信息\n")
            f.write("4. 所有提取的图像都保存为高质量JPEG格式（质量95%）\n")
            f.write("5. 建议在使用前先查看sample_annotations进行质量验证\n\n")
            
            f.write("---\n")
            f.write(f"*数据集生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    def _create_statistics_data(self, metadata: Dict[str, Any]):
        """创建统计数据文件"""
        stats_data = {
            'confidence_distribution': metadata['confidence_distribution'],
            'size_distribution': metadata['size_distribution'],
            'processing_performance': metadata['processing_performance'],
            'quality_metrics': metadata['quality_analysis']
        }
        
        stats_file = self.output_dir / "statistics" / "dataset_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
    
    def run_large_scale_extraction_pipeline(self) -> Dict[str, Any]:
        """运行大规模数据集提取pipeline"""
        logger.info("🚀 启动1962 TI大规模数据集提取Pipeline")
        logger.info("=" * 80)
        
        try:
            # 1. 获取图像文件列表
            image_files = self.get_image_files()
            if not image_files:
                logger.error("❌ 未找到任何图像文件")
                return {'success': False, 'error': 'No image files found'}
            
            # 2. 批量处理图像
            processing_results = self.process_batch(image_files)
            
            # 3. 创建综合数据集元数据
            metadata_file = self.create_comprehensive_dataset_metadata(processing_results)
            
            logger.info("🎉 大规模数据集提取Pipeline执行成功!")
            logger.info(f"📋 综合元数据: {metadata_file}")
            logger.info(f"📁 数据集目录: {self.output_dir}")
            
            return {
                'success': True,
                'metadata_file': metadata_file,
                'output_directory': self.output_dir,
                'processing_results': processing_results
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline执行失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """主函数"""
    # 配置路径
    model_path = "/Users/zhaoye/Desktop/1956_TI_Dataset/models/super_optimized_best_yolov8n.pt"
    source_dir = "/Users/zhaoye/Desktop/1962 TI"
    output_dir = "/Users/zhaoye/Desktop/1962 TI_Extracted_Dataset"
    
    print("🔍 1962 TI大规模数据集提取Pipeline")
    print("=" * 80)
    print(f"🎯 使用模型: {Path(model_path).name}")
    print(f"📂 源目录: {source_dir}")
    print(f"📁 输出目录: {output_dir}")
    print("🎨 功能: 大规模处理 + 细化分类 + 质量分析 + 综合数据集构建")
    print("=" * 80)
    
    try:
        # 创建大规模数据集提取器
        extractor = TI1962DatasetExtractor(model_path, source_dir, output_dir)
        
        # 运行完整pipeline
        results = extractor.run_large_scale_extraction_pipeline()
        
        if results['success']:
            print("\n🎉 大规模数据集提取Pipeline执行成功!")
            print("=" * 80)
            
            stats = results['processing_results']['statistics']
            
            print(f"📊 处理结果:")
            print(f"  源图像数: {stats['total_processed']:,} 张")
            print(f"  成功处理: {stats['total_processed'] - len(stats['failed_images']):,} 张")
            print(f"  总提取数: {stats['total_extracted']:,} 个图像")
            print(f"  处理速度: {stats['processing_speed']:.2f} 张/秒")
            print(f"  处理时间: {stats['processing_time']/60:.1f} 分钟")
            
            print(f"\n📈 置信度分布:")
            conf_dist = stats['confidence_distribution']
            total_detections = stats['total_detections']
            print(f"  高置信度 (≥0.8): {conf_dist['high']:,} 个 ({conf_dist['high']/total_detections*100:.1f}%)")
            print(f"  良好置信度 (≥0.6): {conf_dist['good']:,} 个 ({conf_dist['good']/total_detections*100:.1f}%)")
            print(f"  中等置信度 (≥0.4): {conf_dist['medium']:,} 个 ({conf_dist['medium']/total_detections*100:.1f}%)")
            print(f"  低置信度 (≥0.25): {conf_dist['low']:,} 个 ({conf_dist['low']/total_detections*100:.1f}%)")
            
            print(f"\n🏆 数据集质量分数: {stats['dataset_quality_score']:.1f}/100")
            
            if stats['failed_images']:
                print(f"\n⚠️ 失败文件: {len(stats['failed_images'])} 个")
            
            print(f"\n📁 大规模数据集结构:")
            print(f"  📸 所有提取图像: extracted_images/ ({stats['total_extracted']:,} 个)")
            print(f"  🟢 高置信度图像: high_confidence/ ({conf_dist['high']:,} 个)")
            print(f"  🔵 良好置信度图像: good_confidence/ ({conf_dist['good']:,} 个)")
            print(f"  🟡 中等置信度图像: medium_confidence/ ({conf_dist['medium']:,} 个)")
            print(f"  🔴 低置信度图像: low_confidence/ ({conf_dist['low']:,} 个)")
            print(f"  🖼️ 示例标注: sample_annotations/ (前100个)")
            print(f"  📋 综合元数据: metadata/")
            print(f"  📊 处理统计: statistics/")
            print(f"  📝 详细README: README.md")
            
            print("\n💡 下一步建议:")
            print("1. 查看 README.md 了解数据集完整信息")
            print("2. 优先使用 high_confidence/ 和 good_confidence/ 中的图像")
            print("3. 使用 sample_annotations/ 进行快速质量验证")
            print("4. 根据应用需求选择合适的置信度等级")
            print("5. 这是一个大规模数据集，可直接用于深度学习训练")
            
        else:
            print(f"❌ Pipeline执行失败: {results['error']}")
    
    except Exception as e:
        logger.error(f"主程序执行失败: {e}")
        print(f"❌ 执行失败: {e}")


if __name__ == "__main__":
    main()
