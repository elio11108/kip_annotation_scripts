#!/usr/bin/env python3
"""
1959 TI综合数据集图像提取器
递归处理1959 TI文件夹及其所有子文件夹中的图像
确保不遗漏任何图像，创建最大规模的结构化数据集
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
        logging.FileHandler('/Users/zhaoye/Desktop/1956_TI_Dataset/ti_1959_comprehensive_extraction.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class TI1959ComprehensiveExtractor:
    def __init__(self, model_path: str, source_dir: str, output_dir: str):
        self.model_path = Path(model_path)
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # 验证输入
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        if not self.source_dir.exists():
            raise FileNotFoundError(f"源目录不存在: {self.source_dir}")
        
        # 创建输出目录结构（超大规模数据集结构）
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "extracted_images").mkdir(exist_ok=True)
        (self.output_dir / "excellent_confidence").mkdir(exist_ok=True)   # ≥0.9
        (self.output_dir / "high_confidence").mkdir(exist_ok=True)       # ≥0.7
        (self.output_dir / "good_confidence").mkdir(exist_ok=True)       # ≥0.5
        (self.output_dir / "medium_confidence").mkdir(exist_ok=True)     # ≥0.3
        (self.output_dir / "low_confidence").mkdir(exist_ok=True)        # ≥0.25
        (self.output_dir / "sample_annotations").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        (self.output_dir / "statistics").mkdir(exist_ok=True)
        (self.output_dir / "source_mapping").mkdir(exist_ok=True)
        
        # 加载模型
        self.model = None
        self.load_model()
        
        # 置信度阈值设置（5级细化分类）
        self.confidence_thresholds = {
            'excellent': 0.9,    # 卓越置信度
            'high': 0.7,         # 高置信度
            'good': 0.5,         # 良好置信度
            'medium': 0.3,       # 中等置信度
            'low': 0.25          # 低置信度（检测阈值）
        }
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'total_detections': 0,
            'confidence_distribution': {'excellent': 0, 'high': 0, 'good': 0, 'medium': 0, 'low': 0},
            'total_extracted': 0,
            'processing_time': 0,
            'failed_images': [],
            'dataset_quality_score': 0,
            'source_distribution': {},  # 记录每个源文件夹的统计
            'processing_speed': 0
        }
        
        # 处理进度跟踪
        self.progress = {
            'current_batch': 0,
            'total_batches': 0,
            'processed_images': 0,
            'start_time': None,
            'current_subfolder': None
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
    
    def get_all_image_files(self) -> Dict[str, List[Path]]:
        """递归获取所有子目录中的图像文件"""
        logger.info(f"📂 递归扫描源目录: {self.source_dir}")
        
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        all_image_files = {}
        
        # 递归搜索所有子目录
        for root, dirs, files in os.walk(self.source_dir):
            root_path = Path(root)
            relative_path = root_path.relative_to(self.source_dir)
            
            image_files = []
            for file in files:
                file_path = root_path / file
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(file_path)
            
            if image_files:
                folder_key = str(relative_path) if str(relative_path) != '.' else 'root'
                all_image_files[folder_key] = sorted(image_files)
                logger.info(f"   📁 {folder_key}: {len(image_files)} 个图像文件")
        
        # 计算总数
        total_images = sum(len(files) for files in all_image_files.values())
        logger.info(f"📊 总计找到 {total_images} 个图像文件，分布在 {len(all_image_files)} 个文件夹中")
        
        return all_image_files
    
    def categorize_by_confidence(self, confidence: float) -> str:
        """根据置信度分类（5级细化分级）"""
        if confidence >= self.confidence_thresholds['excellent']:
            return 'excellent'
        elif confidence >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence >= self.confidence_thresholds['good']:
            return 'good'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def process_single_image(self, image_path: Path, source_folder: str) -> Dict[str, Any]:
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
            confidence_counts = {'excellent': 0, 'high': 0, 'good': 0, 'medium': 0, 'low': 0}
            
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
                    
                    # 生成提取图像的文件名（包含源文件夹信息）
                    base_name = image_path.stem
                    source_prefix = source_folder.replace('/', '_').replace(' ', '_')
                    extracted_filename = f"{source_prefix}_{base_name}_extracted_{i+1}_{confidence_category}_conf{conf:.3f}.jpg"
                    
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
                        'area': area,
                        'extracted_file': extracted_filename,
                        'source_folder': source_folder,
                        'width': x2 - x1,
                        'height': y2 - y1
                    }
                    detections.append(detection_info)
                    extracted_images.append(extracted_path)
                    
                    # 在原图上绘制边界框（颜色根据置信度）
                    color_map = {
                        'excellent': (255, 0, 255),  # 紫色
                        'high': (0, 255, 0),         # 绿色
                        'good': (0, 255, 255),       # 黄色
                        'medium': (0, 165, 255),     # 橙色
                        'low': (0, 0, 255)           # 红色
                    }
                    color = color_map[confidence_category]
                    
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    # 添加置信度标签
                    label = f"{confidence_category.upper()}: {conf:.3f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 8), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_image, label, (x1, y1 - 4), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # 保存带标注的原图（限制数量以节省空间）
                if detections and self.stats['total_processed'] < 150:
                    annotated_filename = f"{source_prefix}_{image_path.stem}_annotated.jpg"
                    annotated_path = self.output_dir / "sample_annotations" / annotated_filename
                    cv2.imwrite(str(annotated_path), annotated_image, 
                               [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # 更新统计
            for category, count in confidence_counts.items():
                self.stats['confidence_distribution'][category] += count
            
            # 更新源文件夹统计
            if source_folder not in self.stats['source_distribution']:
                self.stats['source_distribution'][source_folder] = {
                    'processed': 0, 'detections': 0, 'extracted': 0
                }
            
            self.stats['source_distribution'][source_folder]['processed'] += 1
            self.stats['source_distribution'][source_folder]['detections'] += len(detections)
            self.stats['source_distribution'][source_folder]['extracted'] += len(detections)
            
            # 返回处理结果
            return {
                'success': True,
                'original_file': image_path.name,
                'source_folder': source_folder,
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
                'source_folder': source_folder,
                'error': str(e)
            }
    
    def process_all_subfolders(self, all_image_files: Dict[str, List[Path]], batch_size: int = 50) -> Dict[str, Any]:
        """处理所有子文件夹中的图像"""
        logger.info(f"🚀 开始综合处理所有子文件夹")
        logger.info(f"📦 批次大小: {batch_size}")
        logger.info(f"🎯 置信度阈值: 卓越≥{self.confidence_thresholds['excellent']}, 高≥{self.confidence_thresholds['high']}, 良好≥{self.confidence_thresholds['good']}, 中≥{self.confidence_thresholds['medium']}, 低≥{self.confidence_thresholds['low']}")
        
        start_time = datetime.datetime.now()
        self.progress['start_time'] = start_time
        all_results = []
        
        # 计算总图像数
        total_images = sum(len(files) for files in all_image_files.values())
        
        # 按子文件夹处理
        processed_count = 0
        for folder_name, image_files in all_image_files.items():
            logger.info(f"\n📁 处理文件夹: {folder_name} ({len(image_files)} 个图像)")
            self.progress['current_subfolder'] = folder_name
            
            # 分批处理当前文件夹的图像
            for i in range(0, len(image_files), batch_size):
                batch_files = image_files[i:i+batch_size]
                batch_num = (i // batch_size) + 1
                folder_batches = (len(image_files) + batch_size - 1) // batch_size
                
                batch_start_time = time.time()
                
                logger.info(f"   📦 {folder_name} 批次 {batch_num}/{folder_batches} ({len(batch_files)} 个文件)")
                
                for j, image_file in enumerate(batch_files):
                    result = self.process_single_image(image_file, folder_name)
                    all_results.append(result)
                    
                    # 更新统计
                    self.stats['total_processed'] += 1
                    processed_count += 1
                    
                    if result['success']:
                        self.stats['total_detections'] += result['detections_count']
                        self.stats['total_extracted'] += result['detections_count']
                    else:
                        self.stats['failed_images'].append(f"{folder_name}/{result['original_file']}")
                    
                    # 显示进度（每25个文件）
                    if (j + 1) % 25 == 0 or (j + 1) == len(batch_files):
                        overall_progress = (processed_count / total_images) * 100
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        avg_speed = processed_count / elapsed_time if elapsed_time > 0 else 0
                        
                        logger.info(f"      进度: {j+1}/{len(batch_files)} "
                                  f"(总进度: {overall_progress:.1f}%, 速度: {avg_speed:.2f}张/秒)")
                
                # 批次完成统计
                batch_detections = sum(r['detections_count'] for r in all_results[-len(batch_files):] if r['success'])
                batch_failures = sum(1 for r in all_results[-len(batch_files):] if not r['success'])
                batch_time = time.time() - batch_start_time
                
                logger.info(f"   ✅ 批次完成: {len(batch_files) - batch_failures} 成功, "
                           f"{batch_failures} 失败, {batch_detections} 个检测, 用时: {batch_time:.1f}秒")
            
            # 文件夹完成统计
            folder_results = [r for r in all_results if r.get('source_folder') == folder_name]
            folder_detections = sum(r['detections_count'] for r in folder_results if r['success'])
            folder_successes = sum(1 for r in folder_results if r['success'])
            
            logger.info(f"✅ 文件夹 {folder_name} 完成: {folder_successes}/{len(image_files)} 成功, {folder_detections} 个检测")
        
        # 计算总体统计
        end_time = datetime.datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        self.stats['processing_speed'] = total_images / self.stats['processing_time'] if self.stats['processing_time'] > 0 else 0
        
        success_count = sum(1 for r in all_results if r['success'])
        failure_count = len(all_results) - success_count
        
        # 计算数据集质量分数（5级分类）
        total_detections = self.stats['total_detections']
        if total_detections > 0:
            excellent_ratio = self.stats['confidence_distribution']['excellent'] / total_detections
            high_ratio = self.stats['confidence_distribution']['high'] / total_detections
            good_ratio = self.stats['confidence_distribution']['good'] / total_detections
            medium_ratio = self.stats['confidence_distribution']['medium'] / total_detections
            low_ratio = self.stats['confidence_distribution']['low'] / total_detections
            
            # 质量分数计算（卓越和高置信度权重最大）
            self.stats['dataset_quality_score'] = (
                excellent_ratio * 1.0 + high_ratio * 0.9 + good_ratio * 0.7 + 
                medium_ratio * 0.4 + low_ratio * 0.2
            ) * 100
        
        logger.info("\n🎉 综合处理完成!")
        logger.info(f"📊 最终统计:")
        logger.info(f"   总文件数: {total_images}")
        logger.info(f"   成功处理: {success_count} ({success_count/total_images*100:.1f}%)")
        logger.info(f"   处理失败: {failure_count}")
        logger.info(f"   总检测数: {self.stats['total_detections']}")
        logger.info(f"   卓越置信度: {self.stats['confidence_distribution']['excellent']} 个")
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
    
    def create_comprehensive_metadata(self, processing_results: Dict[str, Any]) -> Path:
        """创建综合数据集元数据"""
        logger.info("📋 创建超大规模数据集综合元数据...")
        
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
                'name': '1959 TI Comprehensive Extracted Dataset',
                'description': '从1959 TI文档中提取的超大规模综合图像数据集（包含1959 1和1959 2子文件夹）',
                'created_date': datetime.datetime.now().isoformat(),
                'source_directory': str(self.source_dir),
                'source_subfolders': list(self.stats['source_distribution'].keys()),
                'extraction_model': str(self.model_path),
                'model_size_mb': self.model_path.stat().st_size / (1024 * 1024),
                'dataset_scale': 'extra_large'
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
            'source_folder_distribution': self.stats['source_distribution'],
            'confidence_distribution': {},
            'image_characteristics': {},
            'quality_analysis': self._analyze_comprehensive_quality(),
            'usage_recommendations': self._generate_comprehensive_recommendations()
        }
        
        # 置信度分布
        total_detections = self.stats['total_detections']
        threshold_map = {'excellent': '≥0.9', 'high': '≥0.7', 'good': '≥0.5', 'medium': '≥0.3', 'low': '≥0.25'}
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
                    'median': float(np.median(all_areas)),
                    'percentile_25': float(np.percentile(all_areas, 25)),
                    'percentile_75': float(np.percentile(all_areas, 75))
                },
                'confidence_stats': {
                    'mean': float(np.mean(all_confidences)),
                    'std': float(np.std(all_confidences)),
                    'min': float(np.min(all_confidences)),
                    'max': float(np.max(all_confidences)),
                    'median': float(np.median(all_confidences)),
                    'percentile_25': float(np.percentile(all_confidences, 25)),
                    'percentile_75': float(np.percentile(all_confidences, 75))
                },
                'dimension_stats': {
                    'width_mean': float(np.mean(all_widths)),
                    'height_mean': float(np.mean(all_heights)),
                    'width_range': [float(np.min(all_widths)), float(np.max(all_widths))],
                    'height_range': [float(np.min(all_heights)), float(np.max(all_heights))],
                    'aspect_ratios': [w/h for w, h in zip(all_widths, all_heights) if h > 0]
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
        
        # 创建源文件夹映射文件
        mapping_file = self._create_source_mapping(processing_results)
        
        logger.info(f"✅ 综合数据集元数据已生成: {metadata_file}")
        logger.info(f"📝 详细README文件已生成: {readme_file}")
        logger.info(f"🗂️ 源文件夹映射已生成: {mapping_file}")
        
        return metadata_file
    
    def _create_source_mapping(self, processing_results: Dict[str, Any]) -> Path:
        """创建源文件夹到提取文件的映射"""
        mapping = {
            'source_to_extracted_mapping': {},
            'extraction_summary_by_folder': self.stats['source_distribution']
        }
        
        # 创建详细映射
        for result in processing_results['results']:
            if result['success'] and result['detections']:
                source_key = f"{result['source_folder']}/{result['original_file']}"
                mapping['source_to_extracted_mapping'][source_key] = {
                    'detections_count': result['detections_count'],
                    'extracted_files': result['extracted_files'],
                    'confidence_breakdown': result['confidence_distribution']
                }
        
        mapping_file = self.output_dir / "source_mapping" / "source_to_extracted_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        
        return mapping_file
    
    def _analyze_comprehensive_quality(self) -> Dict[str, Any]:
        """分析综合数据集质量"""
        total = self.stats['total_detections']
        if total == 0:
            return {'overall_quality': 'empty', 'quality_rating': 0}
        
        excellent_ratio = self.stats['confidence_distribution']['excellent'] / total
        high_ratio = self.stats['confidence_distribution']['high'] / total
        good_ratio = self.stats['confidence_distribution']['good'] / total
        top_tier_ratio = excellent_ratio + high_ratio + good_ratio
        
        analysis = {
            'excellent_confidence_ratio': excellent_ratio,
            'high_confidence_ratio': high_ratio,
            'good_confidence_ratio': good_ratio,
            'top_tier_combined_ratio': top_tier_ratio,
            'premium_quality_ratio': excellent_ratio + high_ratio
        }
        
        # 质量评级（基于5级分类）
        if top_tier_ratio > 0.8:
            analysis['overall_quality'] = 'exceptional'
            analysis['quality_rating'] = 5
        elif top_tier_ratio > 0.6:
            analysis['overall_quality'] = 'excellent'
            analysis['quality_rating'] = 4
        elif top_tier_ratio > 0.4:
            analysis['overall_quality'] = 'good'
            analysis['quality_rating'] = 3
        elif top_tier_ratio > 0.2:
            analysis['overall_quality'] = 'fair'
            analysis['quality_rating'] = 2
        else:
            analysis['overall_quality'] = 'poor'
            analysis['quality_rating'] = 1
        
        return analysis
    
    def _generate_comprehensive_recommendations(self) -> List[str]:
        """生成综合使用建议"""
        recommendations = []
        
        total = self.stats['total_detections']
        if total == 0:
            return ["数据集为空，无法提供使用建议"]
        
        excellent_ratio = self.stats['confidence_distribution']['excellent'] / total
        high_ratio = self.stats['confidence_distribution']['high'] / total
        good_ratio = self.stats['confidence_distribution']['good'] / total
        top_tier_ratio = excellent_ratio + high_ratio + good_ratio
        
        # 基于质量的建议
        if top_tier_ratio > 0.8:
            recommendations.extend([
                "超大规模高质量数据集，适合大型深度学习项目",
                "卓越和高置信度图像可作为黄金标准数据集",
                "适合用于SOTA模型训练和基准测试"
            ])
        elif top_tier_ratio > 0.6:
            recommendations.extend([
                "大规模优质数据集，适合工业级应用",
                "高质量图像比例优秀，可直接投入生产使用"
            ])
        else:
            recommendations.extend([
                "大规模数据集，建议结合质量筛选使用",
                "优先使用卓越和高置信度图像"
            ])
        
        # 基于数据量的建议
        if total > 1500:
            recommendations.append("超大规模数据集，适合大型transformer模型训练")
        elif total > 1000:
            recommendations.append("大规模数据集，适合深度学习模型训练")
        else:
            recommendations.append("中大规模数据集，适合迁移学习和微调")
        
        # 特殊建议
        recommendations.extend([
            "数据来源于多个子文件夹，具有良好的多样性",
            "建议根据源文件夹进行分层采样",
            "可用于跨域适应性研究",
            "适合构建大规模图像检测基准数据集"
        ])
        
        return recommendations
    
    def _create_comprehensive_readme(self, metadata: Dict[str, Any], readme_file: Path):
        """创建综合README文件"""
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write("# 1959 TI Comprehensive Extracted Dataset\n\n")
            
            # 数据集概览
            f.write("## 数据集概览\n\n")
            f.write(f"**数据集名称**: {metadata['dataset_info']['name']}\n\n")
            f.write(f"**描述**: {metadata['dataset_info']['description']}\n\n")
            f.write(f"**规模**: 超大规模数据集 ({metadata['dataset_statistics']['total_source_images']} 源图像 → {metadata['dataset_statistics']['total_extracted_images']} 提取图像)\n\n")
            f.write(f"**源文件夹**: {', '.join(metadata['dataset_info']['source_subfolders'])}\n\n")
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
            f.write(f"- **优质图像比例**: {quality['premium_quality_ratio']*100:.1f}%\n\n")
            
            # 源文件夹分布
            f.write("## 源文件夹分布\n\n")
            f.write("| 源文件夹 | 处理图像数 | 检测数 | 提取数 |\n")
            f.write("|---------|-----------|--------|--------|\n")
            for folder, stats in metadata['source_folder_distribution'].items():
                f.write(f"| {folder} | {stats['processed']:,} | {stats['detections']:,} | {stats['extracted']:,} |\n")
            f.write("\n")
            
            # 置信度分布
            f.write("## 置信度分布（5级分类）\n\n")
            f.write("| 置信度等级 | 数量 | 阈值 | 占比 |\n")
            f.write("|-----------|------|------|------|\n")
            conf_names = {'excellent_confidence': '卓越', 'high_confidence': '高', 'good_confidence': '良好', 'medium_confidence': '中等', 'low_confidence': '低'}
            for level, info in metadata['confidence_distribution'].items():
                level_name = conf_names.get(level, level)
                f.write(f"| {level_name} | {info['count']:,} | {info['threshold']} | {info['percentage']:.1f}% |\n")
            f.write("\n")
            
            # 文件夹结构
            f.write("## 文件夹结构\n\n")
            f.write("```\n")
            f.write("1959 TI_Extracted_Dataset/\n")
            f.write("├── extracted_images/          # 所有提取的图像\n")
            f.write("├── excellent_confidence/      # 卓越置信度图像 (≥0.9)\n")
            f.write("├── high_confidence/           # 高置信度图像 (≥0.7)\n")
            f.write("├── good_confidence/           # 良好置信度图像 (≥0.5)\n")
            f.write("├── medium_confidence/         # 中等置信度图像 (≥0.3)\n")
            f.write("├── low_confidence/            # 低置信度图像 (≥0.25)\n")
            f.write("├── sample_annotations/        # 前150个带标注的示例图像\n")
            f.write("├── metadata/                  # 数据集元数据和统计信息\n")
            f.write("├── statistics/                # 处理过程中的统计数据\n")
            f.write("├── source_mapping/            # 源文件到提取文件的映射\n")
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
                f.write(f"- **中位数**: {area_stats['median']:,.0f} 像素²\n")
                f.write(f"- **四分位数**: {area_stats['percentile_25']:,.0f} - {area_stats['percentile_75']:,.0f} 像素²\n\n")
                
                f.write("### 置信度统计\n")
                f.write(f"- **平均置信度**: {conf_stats['mean']:.3f}\n")
                f.write(f"- **置信度范围**: {conf_stats['min']:.3f} - {conf_stats['max']:.3f}\n")
                f.write(f"- **中位数**: {conf_stats['median']:.3f}\n")
                f.write(f"- **四分位数**: {conf_stats['percentile_25']:.3f} - {conf_stats['percentile_75']:.3f}\n\n")
                
                f.write("### 尺寸统计\n")
                f.write(f"- **平均宽度**: {dim_stats['width_mean']:.0f} 像素\n")
                f.write(f"- **平均高度**: {dim_stats['height_mean']:.0f} 像素\n")
                f.write(f"- **宽度范围**: {dim_stats['width_range'][0]:.0f} - {dim_stats['width_range'][1]:.0f} 像素\n")
                f.write(f"- **高度范围**: {dim_stats['height_range'][0]:.0f} - {dim_stats['height_range'][1]:.0f} 像素\n\n")
            
            # 技术说明
            f.write("## 技术说明\n\n")
            f.write("### 5级置信度分类系统\n")
            f.write("- **卓越置信度 (≥0.9)**: 模型极度确信的检测结果，黄金标准\n")
            f.write("- **高置信度 (≥0.7)**: 模型非常确信的检测结果，推荐直接使用\n")
            f.write("- **良好置信度 (≥0.5)**: 模型较为确信的检测结果，质量可靠\n")
            f.write("- **中等置信度 (≥0.3)**: 模型中等确信的检测结果，建议审核\n")
            f.write("- **低置信度 (≥0.25)**: 模型不太确信的检测结果，需要仔细审核\n\n")
            
            f.write("### 文件命名规则\n")
            f.write("提取的图像文件命名格式：`源文件夹_原文件名_extracted_序号_置信度等级_conf置信度值.jpg`\n\n")
            f.write("示例：`1959_1_1959_Issue-3_1_page_0_fig0_fig0_extracted_1_high_conf0.876.jpg`\n\n")
            
            # 注意事项
            f.write("## 注意事项\n\n")
            f.write("1. 这是一个超大规模数据集，包含来自多个子文件夹的图像\n")
            f.write("2. source_mapping文件夹包含源文件到提取文件的完整映射\n")
            f.write("3. 建议优先使用卓越和高置信度图像\n")
            f.write("4. sample_annotations包含前150个带标注的原图用于质量评估\n")
            f.write("5. 文件名包含源文件夹信息，便于追溯\n\n")
            
            f.write("---\n")
            f.write(f"*超大规模数据集生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    def run_comprehensive_extraction_pipeline(self) -> Dict[str, Any]:
        """运行综合提取pipeline"""
        logger.info("🚀 启动1959 TI超大规模综合数据集提取Pipeline")
        logger.info("=" * 80)
        
        try:
            # 1. 递归获取所有图像文件
            all_image_files = self.get_all_image_files()
            if not all_image_files:
                logger.error("❌ 未找到任何图像文件")
                return {'success': False, 'error': 'No image files found'}
            
            # 2. 处理所有子文件夹
            processing_results = self.process_all_subfolders(all_image_files)
            
            # 3. 创建综合数据集元数据
            metadata_file = self.create_comprehensive_metadata(processing_results)
            
            logger.info("🎉 超大规模综合数据集提取Pipeline执行成功!")
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
    source_dir = "/Users/zhaoye/Desktop/1959 TI"
    output_dir = "/Users/zhaoye/Desktop/1959 TI_Extracted_Dataset"
    
    print("🔍 1959 TI超大规模综合数据集提取Pipeline")
    print("=" * 80)
    print(f"🎯 使用模型: {Path(model_path).name}")
    print(f"📂 源目录: {source_dir}")
    print(f"📁 输出目录: {output_dir}")
    print("🎨 功能: 递归处理 + 5级分类 + 源映射 + 超大规模数据集构建")
    print("=" * 80)
    
    try:
        # 创建综合数据集提取器
        extractor = TI1959ComprehensiveExtractor(model_path, source_dir, output_dir)
        
        # 运行完整pipeline
        results = extractor.run_comprehensive_extraction_pipeline()
        
        if results['success']:
            print("\n🎉 超大规模综合数据集提取Pipeline执行成功!")
            print("=" * 80)
            
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
            print(f"  卓越置信度 (≥0.9): {conf_dist['excellent']:,} 个 ({conf_dist['excellent']/total_detections*100:.1f}%)")
            print(f"  高置信度 (≥0.7): {conf_dist['high']:,} 个 ({conf_dist['high']/total_detections*100:.1f}%)")
            print(f"  良好置信度 (≥0.5): {conf_dist['good']:,} 个 ({conf_dist['good']/total_detections*100:.1f}%)")
            print(f"  中等置信度 (≥0.3): {conf_dist['medium']:,} 个 ({conf_dist['medium']/total_detections*100:.1f}%)")
            print(f"  低置信度 (≥0.25): {conf_dist['low']:,} 个 ({conf_dist['low']/total_detections*100:.1f}%)")
            
            print(f"\n📁 源文件夹分布:")
            for folder, folder_stats in stats['source_distribution'].items():
                print(f"  {folder}: {folder_stats['processed']:,} 张 → {folder_stats['extracted']:,} 个提取")
            
            print(f"\n🏆 数据集质量分数: {stats['dataset_quality_score']:.1f}/100")
            
            if stats['failed_images']:
                print(f"\n⚠️ 失败文件: {len(stats['failed_images'])} 个")
            
            print(f"\n📁 超大规模数据集结构:")
            print(f"  📸 所有提取图像: extracted_images/ ({stats['total_extracted']:,} 个)")
            print(f"  🟣 卓越置信度图像: excellent_confidence/ ({conf_dist['excellent']:,} 个)")
            print(f"  🟢 高置信度图像: high_confidence/ ({conf_dist['high']:,} 个)")
            print(f"  🔵 良好置信度图像: good_confidence/ ({conf_dist['good']:,} 个)")
            print(f"  🟡 中等置信度图像: medium_confidence/ ({conf_dist['medium']:,} 个)")
            print(f"  🔴 低置信度图像: low_confidence/ ({conf_dist['low']:,} 个)")
            print(f"  🖼️ 示例标注: sample_annotations/ (前150个)")
            print(f"  📋 综合元数据: metadata/")
            print(f"  📊 处理统计: statistics/")
            print(f"  🗂️ 源映射: source_mapping/")
            print(f"  📝 详细README: README.md")
            
            print("\n💡 下一步建议:")
            print("1. 查看 README.md 了解数据集完整信息")
            print("2. 优先使用 excellent_confidence/ 和 high_confidence/ 中的图像")
            print("3. 使用 source_mapping/ 追溯图像来源")
            print("4. 这是超大规模数据集，适合大型深度学习项目")
            print("5. 可用于构建行业标准的图像检测基准")
            
        else:
            print(f"❌ Pipeline执行失败: {results['error']}")
    
    except Exception as e:
        logger.error(f"主程序执行失败: {e}")
        print(f"❌ 执行失败: {e}")


if __name__ == "__main__":
    main()
