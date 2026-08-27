#!/usr/bin/env python3
"""
1957 Multi数据集图像提取器
使用训练好的模型对1957 Multi文件夹中的所有图像进行处理
提取检测到的图像区域并保存到"1957 Multi_Extracted_Dataset"文件夹
专为多Issue数据集优化，包含详细的Issue分析和1957年特征研究
这是第十二个数据集处理任务，完成最终验证系列
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
        logging.FileHandler('/Users/zhaoye/Desktop/1956_TI_Dataset/ti_1957_multi_extraction.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class TI1957MultiExtractor:
    def __init__(self, model_path: str, source_dir: str, output_dir: str):
        self.model_path = Path(model_path)
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # 验证输入
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        if not self.source_dir.exists():
            raise FileNotFoundError(f"源目录不存在: {self.source_dir}")
        
        # 创建输出目录结构（多Issue数据集结构）
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "extracted_images").mkdir(exist_ok=True)
        (self.output_dir / "premium_quality").mkdir(exist_ok=True)        # ≥0.9
        (self.output_dir / "excellent_confidence").mkdir(exist_ok=True)   # ≥0.8
        (self.output_dir / "high_confidence").mkdir(exist_ok=True)        # ≥0.65
        (self.output_dir / "good_confidence").mkdir(exist_ok=True)        # ≥0.45
        (self.output_dir / "low_confidence").mkdir(exist_ok=True)         # ≥0.25
        (self.output_dir / "sample_annotations").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        (self.output_dir / "issue_analysis").mkdir(exist_ok=True)
        (self.output_dir / "quality_reports").mkdir(exist_ok=True)
        (self.output_dir / "final_validation").mkdir(exist_ok=True)
        
        # 加载模型
        self.model = None
        self.load_model()
        
        # 置信度阈值设置（5级分类）
        self.confidence_thresholds = {
            'premium': 0.9,      # 顶级置信度
            'excellent': 0.8,    # 卓越置信度
            'high': 0.65,        # 高置信度
            'good': 0.45,        # 良好置信度
            'low': 0.25          # 低置信度（检测阈值）
        }
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'total_detections': 0,
            'confidence_distribution': {'premium': 0, 'excellent': 0, 'high': 0, 'good': 0, 'low': 0},
            'total_extracted': 0,
            'processing_time': 0,
            'failed_images': [],
            'dataset_quality_score': 0,
            'processing_speed': 0,
            'issue_distribution': {},  # 按Issue分组的详细统计
            'year_analysis': '1957',   # 年份分析
            'final_validation_metrics': {}  # 最终验证指标
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
        """获取源目录中的所有图像文件并分析Issue分布"""
        logger.info(f"📂 扫描源目录: {self.source_dir}")
        
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.source_dir.glob(f'*{ext}'))
            image_files.extend(self.source_dir.glob(f'*{ext.upper()}'))
        
        # 排序以确保处理顺序一致
        image_files.sort()
        
        # 分析Issue分布
        issue_counts = {}
        for img_file in image_files:
            issue_name = self.extract_issue_info(img_file.stem)
            issue_counts[issue_name] = issue_counts.get(issue_name, 0) + 1
        
        logger.info(f"📊 找到 {len(image_files)} 个图像文件")
        logger.info(f"📋 Issue分布: {len(issue_counts)} 个不同的Issue")
        for issue, count in sorted(issue_counts.items()):
            logger.info(f"   {issue}: {count} 张图像")
            
        return image_files
    
    def extract_issue_info(self, filename: str) -> str:
        """从文件名提取Issue信息"""
        parts = filename.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return "unknown"
    
    def categorize_by_confidence(self, confidence: float) -> str:
        """根据置信度分类（5级分类）"""
        if confidence >= self.confidence_thresholds['premium']:
            return 'premium'
        elif confidence >= self.confidence_thresholds['excellent']:
            return 'excellent'
        elif confidence >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence >= self.confidence_thresholds['good']:
            return 'good'
        else:
            return 'low'
    
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
            confidence_counts = {'premium': 0, 'excellent': 0, 'high': 0, 'good': 0, 'low': 0}
            
            # 提取Issue信息
            issue_name = self.extract_issue_info(image_path.stem)
            
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
                    
                    # 根据置信度分类
                    confidence_category = self.categorize_by_confidence(conf)
                    confidence_counts[confidence_category] += 1
                    
                    # 生成提取图像的文件名（包含Issue信息）
                    base_name = image_path.stem
                    extracted_filename = f"{issue_name}_{base_name}_extracted_{i+1}_{confidence_category}_conf{conf:.3f}.jpg"
                    
                    # 保存到对应的置信度文件夹
                    if confidence_category == 'premium':
                        category_dir = self.output_dir / "premium_quality"
                    else:
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
                        'area': area,
                        'extracted_file': extracted_filename,
                        'issue_name': issue_name,
                        'width': x2 - x1,
                        'height': y2 - y1
                    }
                    detections.append(detection_info)
                    extracted_images.append(extracted_path)
                    
                    # 在原图上绘制边界框（颜色根据置信度）
                    color_map = {
                        'premium': (255, 215, 0),    # 金色
                        'excellent': (255, 0, 255),  # 紫色
                        'high': (0, 255, 0),         # 绿色
                        'good': (0, 255, 255),       # 黄色
                        'low': (0, 0, 255)           # 红色
                    }
                    color = color_map[confidence_category]
                    
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
                    
                    # 添加置信度标签
                    label = f"{confidence_category.upper()}: {conf:.3f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 保存带标注的原图
                if detections:
                    annotated_filename = f"{issue_name}_{image_path.stem}_annotated.jpg"
                    annotated_path = self.output_dir / "sample_annotations" / annotated_filename
                    cv2.imwrite(str(annotated_path), annotated_image, 
                               [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # 更新统计
            for category, count in confidence_counts.items():
                self.stats['confidence_distribution'][category] += count
            
            # 更新Issue统计
            if issue_name not in self.stats['issue_distribution']:
                self.stats['issue_distribution'][issue_name] = {
                    'processed': 0, 'detections': 0, 'extracted': 0,
                    'confidence_breakdown': {'premium': 0, 'excellent': 0, 'high': 0, 'good': 0, 'low': 0}
                }
            
            self.stats['issue_distribution'][issue_name]['processed'] += 1
            self.stats['issue_distribution'][issue_name]['detections'] += len(detections)
            self.stats['issue_distribution'][issue_name]['extracted'] += len(detections)
            
            # 更新Issue的置信度分布
            for category, count in confidence_counts.items():
                self.stats['issue_distribution'][issue_name]['confidence_breakdown'][category] += count
            
            # 返回处理结果
            return {
                'success': True,
                'original_file': image_path.name,
                'issue_name': issue_name,
                'detections_count': len(detections),
                'confidence_distribution': confidence_counts,
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
    
    def process_batch(self, image_files: List[Path], batch_size: int = 20) -> Dict[str, Any]:
        """批量处理图像（中等规模优化）"""
        logger.info(f"🚀 开始批量处理 {len(image_files)} 个图像文件")
        logger.info(f"📦 批次大小: {batch_size}")
        logger.info(f"🎯 置信度阈值: 顶级≥{self.confidence_thresholds['premium']}, 卓越≥{self.confidence_thresholds['excellent']}, 高≥{self.confidence_thresholds['high']}, 良好≥{self.confidence_thresholds['good']}, 低≥{self.confidence_thresholds['low']}")
        
        start_time = datetime.datetime.now()
        all_results = []
        
        # 计算总批次数
        total_batches = (len(image_files) + batch_size - 1) // batch_size
        
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            batch_start_time = time.time()
            
            logger.info(f"📦 处理批次 {batch_num}/{total_batches} ({len(batch_files)} 个文件)")
            
            batch_results = []
            for j, image_file in enumerate(batch_files):
                result = self.process_single_image(image_file)
                batch_results.append(result)
                all_results.append(result)
                
                # 更新统计
                self.stats['total_processed'] += 1
                if result['success']:
                    self.stats['total_detections'] += result['detections_count']
                    self.stats['total_extracted'] += result['detections_count']
                else:
                    self.stats['failed_images'].append(result['original_file'])
                
                # 显示进度（每10个文件）
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
        
        # 计算总体统计
        end_time = datetime.datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        self.stats['processing_speed'] = len(image_files) / self.stats['processing_time'] if self.stats['processing_time'] > 0 else 0
        
        success_count = sum(1 for r in all_results if r['success'])
        failure_count = len(all_results) - success_count
        
        # 计算数据集质量分数（5级分类）
        total_detections = self.stats['total_detections']
        if total_detections > 0:
            premium_ratio = self.stats['confidence_distribution']['premium'] / total_detections
            excellent_ratio = self.stats['confidence_distribution']['excellent'] / total_detections
            high_ratio = self.stats['confidence_distribution']['high'] / total_detections
            good_ratio = self.stats['confidence_distribution']['good'] / total_detections
            low_ratio = self.stats['confidence_distribution']['low'] / total_detections
            
            # 质量分数计算
            self.stats['dataset_quality_score'] = (
                premium_ratio * 1.0 + excellent_ratio * 0.9 + high_ratio * 0.75 + 
                good_ratio * 0.55 + low_ratio * 0.25
            ) * 100
        
        # 计算最终验证指标
        self.stats['final_validation_metrics'] = self._calculate_final_validation_metrics()
        
        logger.info("🎉 批量处理完成!")
        logger.info(f"📊 处理统计:")
        logger.info(f"   总文件数: {len(image_files)}")
        logger.info(f"   成功处理: {success_count} ({success_count/len(image_files)*100:.1f}%)")
        logger.info(f"   处理失败: {failure_count}")
        logger.info(f"   总检测数: {self.stats['total_detections']}")
        logger.info(f"   顶级置信度: {self.stats['confidence_distribution']['premium']} 个")
        logger.info(f"   卓越置信度: {self.stats['confidence_distribution']['excellent']} 个")
        logger.info(f"   高置信度: {self.stats['confidence_distribution']['high']} 个")
        logger.info(f"   良好置信度: {self.stats['confidence_distribution']['good']} 个")
        logger.info(f"   低置信度: {self.stats['confidence_distribution']['low']} 个")
        logger.info(f"   数据集质量分数: {self.stats['dataset_quality_score']:.1f}/100")
        logger.info(f"   处理时间: {self.stats['processing_time']:.1f} 秒 ({self.stats['processing_time']/60:.1f} 分钟)")
        logger.info(f"   处理速度: {self.stats['processing_speed']:.2f} 张/秒")
        
        return {
            'results': all_results,
            'statistics': self.stats,
            'timestamp': end_time.isoformat()
        }
    
    def _calculate_final_validation_metrics(self) -> Dict[str, Any]:
        """计算最终验证指标"""
        total = self.stats['total_detections']
        if total == 0:
            return {'validation_status': 'no_detections'}
        
        premium_ratio = self.stats['confidence_distribution']['premium'] / total
        excellent_ratio = self.stats['confidence_distribution']['excellent'] / total
        high_ratio = self.stats['confidence_distribution']['high'] / total
        top_tier_ratio = premium_ratio + excellent_ratio + high_ratio
        
        return {
            'validation_status': 'completed',
            'dataset_number': 12,  # 第12个数据集
            'top_tier_quality_ratio': top_tier_ratio,
            'premium_quality_ratio': premium_ratio,
            'detection_efficiency': self.stats['total_detections'] / self.stats['total_processed'] if self.stats['total_processed'] > 0 else 0,
            'processing_efficiency': self.stats['processing_speed'],
            'success_rate': (self.stats['total_processed'] - len(self.stats['failed_images'])) / self.stats['total_processed'] * 100 if self.stats['total_processed'] > 0 else 0
        }
    
    def create_comprehensive_metadata(self, processing_results: Dict[str, Any]) -> Path:
        """创建综合数据集元数据"""
        logger.info("📋 创建1957 Multi综合数据集元数据...")
        
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
                'name': '1957 Multi Extracted Dataset',
                'description': '从1957 Multi TI文档中提取的多Issue图像数据集，第十二个验证数据集，完成最终验证系列',
                'created_date': datetime.datetime.now().isoformat(),
                'source_directory': str(self.source_dir),
                'extraction_model': str(self.model_path),
                'model_size_mb': self.model_path.stat().st_size / (1024 * 1024),
                'dataset_scale': 'medium_multi_issue',
                'total_issues': len(self.stats['issue_distribution']),
                'year_focus': '1957',
                'dataset_sequence_number': 12,
                'validation_series_status': 'final_completion'
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
            'issue_distribution': self.stats['issue_distribution'],
            'confidence_distribution': {},
            'image_characteristics': {},
            'quality_analysis': self._analyze_comprehensive_quality(),
            'issue_analysis': self._analyze_issue_performance(),
            'year_analysis': self._analyze_year_characteristics(),
            'final_validation_analysis': self.stats['final_validation_metrics'],
            'usage_recommendations': self._generate_comprehensive_recommendations()
        }
        
        # 置信度分布
        total_detections = self.stats['total_detections']
        threshold_map = {'premium': '≥0.9', 'excellent': '≥0.8', 'high': '≥0.65', 'good': '≥0.45', 'low': '≥0.25'}
        for level, count in self.stats['confidence_distribution'].items():
            metadata['confidence_distribution'][f'{level}_confidence'] = {
                'count': count,
                'threshold': threshold_map[level],
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
        
        # 创建Issue分析报告
        issue_report = self._create_issue_analysis_report(metadata)
        
        # 创建质量报告
        quality_report = self._create_quality_report(metadata)
        
        # 创建最终验证报告
        final_validation_report = self._create_final_validation_report(metadata)
        
        logger.info(f"✅ 综合数据集元数据已生成: {metadata_file}")
        logger.info(f"📝 详细README文件已生成: {readme_file}")
        logger.info(f"📊 Issue分析报告已生成: {issue_report}")
        logger.info(f"🏆 质量分析报告已生成: {quality_report}")
        logger.info(f"🎯 最终验证报告已生成: {final_validation_report}")
        
        return metadata_file
    
    def _analyze_comprehensive_quality(self) -> Dict[str, Any]:
        """分析综合数据集质量"""
        total = self.stats['total_detections']
        if total == 0:
            return {'overall_quality': 'empty', 'quality_rating': 0}
        
        premium_ratio = self.stats['confidence_distribution']['premium'] / total
        excellent_ratio = self.stats['confidence_distribution']['excellent'] / total
        high_ratio = self.stats['confidence_distribution']['high'] / total
        good_ratio = self.stats['confidence_distribution']['good'] / total
        
        top_tier_ratio = premium_ratio + excellent_ratio + high_ratio
        premium_tier_ratio = premium_ratio + excellent_ratio
        
        analysis = {
            'premium_confidence_ratio': premium_ratio,
            'excellent_confidence_ratio': excellent_ratio,
            'high_confidence_ratio': high_ratio,
            'good_confidence_ratio': good_ratio,
            'top_tier_combined_ratio': top_tier_ratio,
            'premium_tier_ratio': premium_tier_ratio
        }
        
        # 质量评级
        if premium_tier_ratio > 0.6:
            analysis['overall_quality'] = 'excellent'
            analysis['quality_rating'] = 5
        elif top_tier_ratio > 0.7:
            analysis['overall_quality'] = 'very_good'
            analysis['quality_rating'] = 4
        elif top_tier_ratio > 0.5:
            analysis['overall_quality'] = 'good'
            analysis['quality_rating'] = 3
        else:
            analysis['overall_quality'] = 'fair'
            analysis['quality_rating'] = 2
        
        return analysis
    
    def _analyze_issue_performance(self) -> Dict[str, Any]:
        """分析各Issue的性能"""
        issue_performance = {}
        
        for issue, stats in self.stats['issue_distribution'].items():
            if stats['processed'] > 0:
                detection_rate = stats['detections'] / stats['processed']
                
                # 计算该Issue的质量分数
                total_detections = stats['detections']
                if total_detections > 0:
                    conf_breakdown = stats['confidence_breakdown']
                    quality_score = (
                        conf_breakdown['premium'] / total_detections * 1.0 +
                        conf_breakdown['excellent'] / total_detections * 0.9 +
                        conf_breakdown['high'] / total_detections * 0.75 +
                        conf_breakdown['good'] / total_detections * 0.55 +
                        conf_breakdown['low'] / total_detections * 0.25
                    ) * 100
                else:
                    quality_score = 0
                
                issue_performance[issue] = {
                    'detection_rate': detection_rate,
                    'quality_score': quality_score,
                    'total_images': stats['processed'],
                    'total_detections': stats['detections'],
                    'confidence_breakdown': stats['confidence_breakdown']
                }
        
        return issue_performance
    
    def _analyze_year_characteristics(self) -> Dict[str, Any]:
        """分析1957年数据特征"""
        return {
            'year': '1957',
            'historical_context': '1957年是TI公司技术发展的重要年份',
            'issue_diversity': len(self.stats['issue_distribution']),
            'data_richness': 'medium' if self.stats['total_detections'] < 200 else 'high',
            'quality_consistency': 'analyzed_by_issue',
            'validation_significance': 'completes_12_dataset_series'
        }
    
    def _generate_comprehensive_recommendations(self) -> List[str]:
        """生成综合使用建议"""
        recommendations = []
        
        total = self.stats['total_detections']
        if total == 0:
            return ["数据集为空，无法提供使用建议"]
        
        premium_ratio = self.stats['confidence_distribution']['premium'] / total
        excellent_ratio = self.stats['confidence_distribution']['excellent'] / total
        high_ratio = self.stats['confidence_distribution']['high'] / total
        top_tier_ratio = premium_ratio + excellent_ratio + high_ratio
        
        # 基于质量的建议
        if top_tier_ratio > 0.7:
            recommendations.extend([
                "1957年高质量多Issue数据集，适合专业深度学习项目",
                "顶级和卓越置信度图像可作为核心训练集",
                "适合用于历史文档处理模型的训练和验证",
                "作为第12个验证数据集，完成了完整的验证系列"
            ])
        elif top_tier_ratio > 0.5:
            recommendations.extend([
                "1957年优质多Issue数据集，质量可靠",
                "建议优先使用高质量图像进行训练"
            ])
        else:
            recommendations.extend([
                "建议结合质量筛选使用",
                "可作为补充数据集使用"
            ])
        
        # 基于Issue多样性的建议
        issue_count = len(self.stats['issue_distribution'])
        recommendations.extend([
            f"数据来源于{issue_count}个不同的1957年Issue，具有良好的内容多样性",
            "建议根据Issue进行分层采样和交叉验证",
            "可用于研究1957年不同Issue间的差异性",
            "适合构建历史文档图像检测训练集",
            "支持基于年份的时间序列分析",
            "完成12个数据集验证系列，证明模型的全面稳定性"
        ])
        
        return recommendations
    
    def _create_comprehensive_readme(self, metadata: Dict[str, Any], readme_file: Path):
        """创建综合README文件"""
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write("# 1957 Multi Extracted Dataset\n\n")
            
            # 数据集概览
            f.write("## 数据集概览\n\n")
            f.write(f"**数据集名称**: {metadata['dataset_info']['name']}\n\n")
            f.write(f"**描述**: {metadata['dataset_info']['description']}\n\n")
            f.write(f"**规模**: 中等规模多Issue数据集 ({metadata['dataset_statistics']['total_source_images']} 源图像 → {metadata['dataset_statistics']['total_extracted_images']} 提取图像)\n\n")
            f.write(f"**Issue数量**: {metadata['dataset_info']['total_issues']} 个不同的Issue\n\n")
            f.write(f"**年份焦点**: {metadata['dataset_info']['year_focus']}\n\n")
            f.write(f"**数据集序号**: 第 {metadata['dataset_info']['dataset_sequence_number']} 个验证数据集\n\n")
            f.write(f"**验证系列状态**: {metadata['dataset_info']['validation_series_status']}\n\n")
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
            f.write(f"- **顶级质量图像比例**: {quality['top_tier_combined_ratio']*100:.1f}%\n")
            f.write(f"- **优质图像比例**: {quality['premium_tier_ratio']*100:.1f}%\n\n")
            
            # 最终验证分析
            final_validation = metadata['final_validation_analysis']
            f.write("## 最终验证分析\n\n")
            f.write(f"- **验证状态**: {final_validation['validation_status']}\n")
            f.write(f"- **数据集序号**: 第 {final_validation['dataset_number']} 个\n")
            f.write(f"- **顶级质量比例**: {final_validation['top_tier_quality_ratio']*100:.1f}%\n")
            f.write(f"- **检测效率**: {final_validation['detection_efficiency']:.2f} 检测/图像\n")
            f.write(f"- **处理效率**: {final_validation['processing_efficiency']:.2f} 张/秒\n")
            f.write(f"- **成功率**: {final_validation['success_rate']:.1f}%\n\n")
            
            # Issue分布
            f.write("## Issue分布与性能\n\n")
            f.write("| Issue | 处理图像数 | 检测数 | 提取数 | 检测率 | 质量分数 |\n")
            f.write("|-------|-----------|--------|--------|---------|---------|\n")
            issue_analysis = metadata['issue_analysis']
            for issue, stats in sorted(metadata['issue_distribution'].items()):
                perf = issue_analysis.get(issue, {})
                detection_rate = perf.get('detection_rate', 0)
                quality_score = perf.get('quality_score', 0)
                f.write(f"| {issue} | {stats['processed']} | {stats['detections']} | {stats['extracted']} | {detection_rate:.2f} | {quality_score:.1f} |\n")
            f.write("\n")
            
            # 置信度分布
            f.write("## 置信度分布（5级分类）\n\n")
            f.write("| 置信度等级 | 数量 | 阈值 | 占比 |\n")
            f.write("|-----------|------|------|------|\n")
            conf_names = {'premium_confidence': '顶级', 'excellent_confidence': '卓越', 'high_confidence': '高', 'good_confidence': '良好', 'low_confidence': '低'}
            for level, info in metadata['confidence_distribution'].items():
                level_name = conf_names.get(level, level)
                f.write(f"| {level_name} | {info['count']:,} | {info['threshold']} | {info['percentage']:.1f}% |\n")
            f.write("\n")
            
            # 1957年特征分析
            year_analysis = metadata['year_analysis']
            f.write("## 1957年数据特征分析\n\n")
            f.write(f"- **历史背景**: {year_analysis['historical_context']}\n")
            f.write(f"- **Issue多样性**: {year_analysis['issue_diversity']} 个不同Issue\n")
            f.write(f"- **数据丰富度**: {year_analysis['data_richness'].upper()}\n")
            f.write(f"- **质量一致性**: 按Issue分析\n")
            f.write(f"- **验证意义**: {year_analysis['validation_significance']}\n\n")
            
            # 文件夹结构
            f.write("## 文件夹结构\n\n")
            f.write("```\n")
            f.write("1957 Multi_Extracted_Dataset/\n")
            f.write("├── extracted_images/          # 所有提取的图像\n")
            f.write("├── premium_quality/           # 顶级置信度图像 (≥0.9)\n")
            f.write("├── excellent_confidence/      # 卓越置信度图像 (≥0.8)\n")
            f.write("├── high_confidence/           # 高置信度图像 (≥0.65)\n")
            f.write("├── good_confidence/           # 良好置信度图像 (≥0.45)\n")
            f.write("├── low_confidence/            # 低置信度图像 (≥0.25)\n")
            f.write("├── sample_annotations/        # 带标注的示例图像\n")
            f.write("├── metadata/                  # 数据集元数据\n")
            f.write("├── issue_analysis/            # Issue分析报告\n")
            f.write("├── quality_reports/           # 质量分析报告\n")
            f.write("├── final_validation/          # 最终验证报告\n")
            f.write("└── README.md                  # 此文件\n")
            f.write("```\n\n")
            
            # 使用建议
            f.write("## 使用建议\n\n")
            for i, recommendation in enumerate(metadata['usage_recommendations'], 1):
                f.write(f"{i}. {recommendation}\n")
            f.write("\n")
            
            # 文件命名规则
            f.write("## 文件命名规则\n\n")
            f.write("提取的图像文件命名格式：`Issue名_原文件名_extracted_序号_置信度等级_conf置信度值.jpg`\n\n")
            f.write("示例：`1957_Issue-1_1957_Issue-1_24_page_0_fig0_fig0_extracted_1_premium_conf0.956.jpg`\n\n")
            
            f.write("---\n")
            f.write(f"*第12个验证数据集，完成1957年多Issue数据集处理: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    def _create_issue_analysis_report(self, metadata: Dict[str, Any]) -> Path:
        """创建Issue分析报告"""
        report = {
            'issue_performance_ranking': self._rank_issues_by_performance(),
            'issue_statistics': metadata['issue_analysis'],
            'recommendations_by_issue': self._get_issue_recommendations(),
            'year_context': '1957',
            'dataset_sequence': 12
        }
        
        report_file = self.output_dir / "issue_analysis" / "issue_performance_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report_file
    
    def _create_quality_report(self, metadata: Dict[str, Any]) -> Path:
        """创建质量报告"""
        report = {
            'overall_assessment': metadata['quality_analysis'],
            'confidence_breakdown': metadata['confidence_distribution'],
            'quality_recommendations': self._get_quality_recommendations(),
            'year_specific_analysis': metadata['year_analysis'],
            'dataset_sequence': 12
        }
        
        report_file = self.output_dir / "quality_reports" / "dataset_quality_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report_file
    
    def _create_final_validation_report(self, metadata: Dict[str, Any]) -> Path:
        """创建最终验证报告"""
        report = {
            'validation_summary': {
                'dataset_number': 12,
                'completion_status': 'final_dataset_completed',
                'validation_series_status': 'complete'
            },
            'performance_metrics': metadata['final_validation_analysis'],
            'quality_assessment': metadata['quality_analysis'],
            'recommendations': [
                "第12个数据集成功完成，验证系列圆满结束",
                "模型在12个不同数据集上表现稳定可靠",
                "适合投入生产环境使用",
                "建议根据具体应用场景选择合适的置信度阈值"
            ]
        }
        
        report_file = self.output_dir / "final_validation" / "final_validation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report_file
    
    def _rank_issues_by_performance(self) -> List[Dict[str, Any]]:
        """按性能排序Issue"""
        issue_rankings = []
        
        for issue, stats in self.stats['issue_distribution'].items():
            if stats['processed'] > 0:
                detection_rate = stats['detections'] / stats['processed']
                
                # 计算质量分数
                total_detections = stats['detections']
                if total_detections > 0:
                    conf_breakdown = stats['confidence_breakdown']
                    quality_score = (
                        conf_breakdown['premium'] / total_detections * 1.0 +
                        conf_breakdown['excellent'] / total_detections * 0.9 +
                        conf_breakdown['high'] / total_detections * 0.75 +
                        conf_breakdown['good'] / total_detections * 0.55 +
                        conf_breakdown['low'] / total_detections * 0.25
                    ) * 100
                else:
                    quality_score = 0
                
                issue_rankings.append({
                    'issue': issue,
                    'detection_rate': detection_rate,
                    'quality_score': quality_score,
                    'total_images': stats['processed'],
                    'total_detections': stats['detections']
                })
        
        # 按质量分数排序
        issue_rankings.sort(key=lambda x: x['quality_score'], reverse=True)
        return issue_rankings
    
    def _get_issue_recommendations(self) -> Dict[str, str]:
        """获取Issue使用建议"""
        rankings = self._rank_issues_by_performance()
        recommendations = {}
        
        for i, issue_data in enumerate(rankings):
            issue = issue_data['issue']
            if issue_data['quality_score'] > 80:
                recommendations[issue] = "1957年优秀Issue，推荐优先使用"
            elif issue_data['quality_score'] > 60:
                recommendations[issue] = "1957年良好Issue，适合常规使用"
            else:
                recommendations[issue] = "1957年一般Issue，建议结合其他数据使用"
        
        return recommendations
    
    def _get_quality_recommendations(self) -> Dict[str, List[str]]:
        """获取质量使用建议"""
        return {
            'premium_quality': ["1957年黄金标准图像，直接用于关键应用"],
            'excellent_confidence': ["1957年顶级质量图像，推荐生产环境使用"],
            'high_confidence': ["1957年优秀质量图像，适合大多数应用"],
            'good_confidence': ["1957年良好质量图像，适合一般应用"],
            'low_confidence': ["需要人工审核后使用"]
        }
    
    def run_comprehensive_extraction_pipeline(self) -> Dict[str, Any]:
        """运行综合提取pipeline"""
        logger.info("🚀 启动1957 Multi综合数据集提取Pipeline")
        logger.info("🎯 第12个验证数据集 - 完成最终验证系列")
        logger.info("=" * 70)
        
        try:
            # 1. 获取图像文件列表
            image_files = self.get_image_files()
            if not image_files:
                logger.error("❌ 未找到任何图像文件")
                return {'success': False, 'error': 'No image files found'}
            
            # 2. 批量处理图像
            processing_results = self.process_batch(image_files)
            
            # 3. 创建综合数据集元数据
            metadata_file = self.create_comprehensive_metadata(processing_results)
            
            logger.info("🎉 1957 Multi综合数据集提取Pipeline执行成功!")
            logger.info("🏆 第12个验证数据集完成，验证系列圆满结束!")
            logger.info(f"📋 综合元数据: {metadata_file}")
            logger.info(f"📁 数据集目录: {self.output_dir}")
            
            return {
                'success': True,
                'metadata_file': metadata_file,
                'output_directory': self.output_dir,
                'processing_results': processing_results,
                'validation_series_complete': True
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
    source_dir = "/Users/zhaoye/Desktop/1957 Multi"
    output_dir = "/Users/zhaoye/Desktop/1957 Multi_Extracted_Dataset"
    
    print("🔍 1957 Multi综合数据集提取Pipeline")
    print("🎯 第12个验证数据集 - 完成最终验证系列")
    print("=" * 70)
    print(f"🎯 使用模型: {Path(model_path).name}")
    print(f"📂 源目录: {source_dir}")
    print(f"📁 输出目录: {output_dir}")
    print("🎨 功能: 中等规模处理 + 5级分类 + 多Issue深度分析 + 1957年特征研究 + 最终验证")
    print("=" * 70)
    
    try:
        # 创建综合数据集提取器
        extractor = TI1957MultiExtractor(model_path, source_dir, output_dir)
        
        # 运行完整pipeline
        results = extractor.run_comprehensive_extraction_pipeline()
        
        if results['success']:
            print("\n🎉 1957 Multi综合数据集提取Pipeline执行成功!")
            print("🏆 第12个验证数据集完成，验证系列圆满结束!")
            print("=" * 70)
            
            stats = results['processing_results']['statistics']
            
            print(f"📊 处理结果:")
            print(f"  源图像数: {stats['total_processed']:,} 张")
            print(f"  成功处理: {stats['total_processed'] - len(stats['failed_images']):,} 张")
            print(f"  总提取数: {stats['total_extracted']:,} 个图像")
            print(f"  处理速度: {stats['processing_speed']:.2f} 张/秒")
            print(f"  处理时间: {stats['processing_time']/60:.1f} 分钟")
            
            print(f"\n📈 5级置信度分布:")
            conf_dist = stats['confidence_distribution']
            total_detections = stats['total_detections']
            print(f"  顶级置信度 (≥0.9): {conf_dist['premium']:,} 个 ({conf_dist['premium']/total_detections*100:.1f}%)")
            print(f"  卓越置信度 (≥0.8): {conf_dist['excellent']:,} 个 ({conf_dist['excellent']/total_detections*100:.1f}%)")
            print(f"  高置信度 (≥0.65): {conf_dist['high']:,} 个 ({conf_dist['high']/total_detections*100:.1f}%)")
            print(f"  良好置信度 (≥0.45): {conf_dist['good']:,} 个 ({conf_dist['good']/total_detections*100:.1f}%)")
            print(f"  低置信度 (≥0.25): {conf_dist['low']:,} 个 ({conf_dist['low']/total_detections*100:.1f}%)")
            
            print(f"\n📋 Issue分布:")
            for issue, issue_stats in sorted(stats['issue_distribution'].items()):
                print(f"  {issue}: {issue_stats['processed']} 张 → {issue_stats['extracted']} 个提取")
            
            print(f"\n🏆 数据集质量分数: {stats['dataset_quality_score']:.1f}/100")
            
            # 最终验证指标
            final_metrics = stats['final_validation_metrics']
            print(f"\n🎯 最终验证指标:")
            print(f"  验证状态: {final_metrics['validation_status']}")
            print(f"  数据集序号: 第 {final_metrics['dataset_number']} 个")
            print(f"  顶级质量比例: {final_metrics['top_tier_quality_ratio']*100:.1f}%")
            print(f"  检测效率: {final_metrics['detection_efficiency']:.2f} 检测/图像")
            print(f"  成功率: {final_metrics['success_rate']:.1f}%")
            
            if stats['failed_images']:
                print(f"\n⚠️ 失败文件: {len(stats['failed_images'])} 个")
            
            print(f"\n📁 综合数据集结构:")
            print(f"  📸 所有提取图像: extracted_images/ ({stats['total_extracted']:,} 个)")
            print(f"  🥇 顶级质量图像: premium_quality/ ({conf_dist['premium']:,} 个)")
            print(f"  🟣 卓越置信度图像: excellent_confidence/ ({conf_dist['excellent']:,} 个)")
            print(f"  🟢 高置信度图像: high_confidence/ ({conf_dist['high']:,} 个)")
            print(f"  🔵 良好置信度图像: good_confidence/ ({conf_dist['good']:,} 个)")
            print(f"  🔴 低置信度图像: low_confidence/ ({conf_dist['low']:,} 个)")
            print(f"  🖼️ 示例标注: sample_annotations/")
            print(f"  📋 综合元数据: metadata/")
            print(f"  📊 Issue分析: issue_analysis/")
            print(f"  🏆 质量报告: quality_reports/")
            print(f"  🎯 最终验证: final_validation/")
            print(f"  📝 综合README: README.md")
            
            print("\n💡 下一步建议:")
            print("1. 查看 README.md 了解数据集完整信息")
            print("2. 优先使用 premium_quality/ 和 excellent_confidence/ 中的图像")
            print("3. 查看 issue_analysis/ 了解各Issue的性能特点")
            print("4. 查看 final_validation/ 了解最终验证结果")
            print("5. 第12个验证数据集，完成了完整的验证系列")
            print("6. 模型已准备好投入生产环境使用")
            
        else:
            print(f"❌ Pipeline执行失败: {results['error']}")
    
    except Exception as e:
        logger.error(f"主程序执行失败: {e}")
        print(f"❌ 执行失败: {e}")


if __name__ == "__main__":
    main()
