#!/usr/bin/env python3
"""
1962 Multi数据集图像提取器
使用训练好的模型对1962 Multi文件夹中的图像进行处理
提取检测到的图像区域并保存到"1962 Multi_Extracted_Dataset"文件夹
专为小规模数据集优化
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

# 设置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/Users/zhaoye/Desktop/1956_TI_Dataset/ti_1962_multi_extraction.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class TI1962MultiExtractor:
    def __init__(self, model_path: str, source_dir: str, output_dir: str):
        self.model_path = Path(model_path)
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # 验证输入
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        if not self.source_dir.exists():
            raise FileNotFoundError(f"源目录不存在: {self.source_dir}")
        
        # 创建输出目录结构（小规模数据集结构）
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "extracted_images").mkdir(exist_ok=True)
        (self.output_dir / "high_confidence").mkdir(exist_ok=True)    # ≥0.7
        (self.output_dir / "medium_confidence").mkdir(exist_ok=True)  # ≥0.4
        (self.output_dir / "low_confidence").mkdir(exist_ok=True)     # ≥0.25
        (self.output_dir / "annotated_originals").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        # 加载模型
        self.model = None
        self.load_model()
        
        # 置信度阈值设置
        self.confidence_thresholds = {
            'high': 0.7,      # 高置信度
            'medium': 0.4,    # 中等置信度
            'low': 0.25       # 低置信度（检测阈值）
        }
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'total_detections': 0,
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'total_extracted': 0,
            'processing_time': 0,
            'failed_images': [],
            'dataset_quality_score': 0,
            'processing_speed': 0
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
        """根据置信度分类"""
        if confidence >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def process_single_image(self, image_path: Path) -> Dict[str, Any]:
        """处理单个图像"""
        try:
            logger.info(f"🔍 处理图像: {image_path.name}")
            
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
            confidence_counts = {'high': 0, 'medium': 0, 'low': 0}
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                logger.info(f"   检测到 {len(boxes)} 个目标")
                
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
                    
                    logger.info(f"   提取图像 {i+1}: {confidence_category} 置信度 {conf:.3f}, 区域大小 {x2-x1}x{y2-y1}")
                    
                    # 记录检测信息
                    detection_info = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'confidence_category': confidence_category,
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
                        'medium': (0, 165, 255),  # 橙色
                        'low': (0, 0, 255)        # 红色
                    }
                    color = color_map[confidence_category]
                    
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
                    
                    # 添加置信度标签
                    label = f"{confidence_category.upper()}: {conf:.3f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 保存带标注的原图
                if detections:
                    annotated_filename = f"{image_path.stem}_annotated.jpg"
                    annotated_path = self.output_dir / "annotated_originals" / annotated_filename
                    cv2.imwrite(str(annotated_path), annotated_image, 
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
                    logger.info(f"   保存标注图像: {annotated_filename}")
            else:
                logger.info(f"   未检测到任何目标")
            
            # 更新统计
            for category, count in confidence_counts.items():
                self.stats['confidence_distribution'][category] += count
            
            # 返回处理结果
            return {
                'success': True,
                'original_file': image_path.name,
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
    
    def process_all_images(self, image_files: List[Path]) -> Dict[str, Any]:
        """处理所有图像（小规模数据集）"""
        logger.info(f"🚀 开始处理 {len(image_files)} 个图像文件")
        
        start_time = datetime.datetime.now()
        all_results = []
        
        for i, image_file in enumerate(image_files):
            logger.info(f"📦 处理进度: {i+1}/{len(image_files)} ({(i+1)/len(image_files)*100:.1f}%)")
            
            result = self.process_single_image(image_file)
            all_results.append(result)
            
            # 更新统计
            self.stats['total_processed'] += 1
            if result['success']:
                self.stats['total_detections'] += result['detections_count']
                self.stats['total_extracted'] += result['detections_count']
            else:
                self.stats['failed_images'].append(result['original_file'])
        
        # 计算总体统计
        end_time = datetime.datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        self.stats['processing_speed'] = len(image_files) / self.stats['processing_time'] if self.stats['processing_time'] > 0 else 0
        
        success_count = sum(1 for r in all_results if r['success'])
        failure_count = len(all_results) - success_count
        
        # 计算数据集质量分数
        total_detections = self.stats['total_detections']
        if total_detections > 0:
            high_ratio = self.stats['confidence_distribution']['high'] / total_detections
            medium_ratio = self.stats['confidence_distribution']['medium'] / total_detections
            low_ratio = self.stats['confidence_distribution']['low'] / total_detections
            
            # 质量分数计算
            self.stats['dataset_quality_score'] = (high_ratio * 1.0 + medium_ratio * 0.6 + low_ratio * 0.3) * 100
        
        logger.info("🎉 处理完成!")
        logger.info(f"📊 处理统计:")
        logger.info(f"   总文件数: {len(image_files)}")
        logger.info(f"   成功处理: {success_count}")
        logger.info(f"   处理失败: {failure_count}")
        logger.info(f"   总检测数: {self.stats['total_detections']}")
        logger.info(f"   高置信度: {self.stats['confidence_distribution']['high']} 个")
        logger.info(f"   中置信度: {self.stats['confidence_distribution']['medium']} 个")
        logger.info(f"   低置信度: {self.stats['confidence_distribution']['low']} 个")
        logger.info(f"   数据集质量分数: {self.stats['dataset_quality_score']:.1f}/100")
        logger.info(f"   处理时间: {self.stats['processing_time']:.1f} 秒")
        logger.info(f"   处理速度: {self.stats['processing_speed']:.2f} 张/秒")
        
        return {
            'results': all_results,
            'statistics': self.stats,
            'timestamp': end_time.isoformat()
        }
    
    def create_dataset_metadata(self, processing_results: Dict[str, Any]) -> Path:
        """创建数据集元数据"""
        logger.info("📋 创建数据集元数据...")
        
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
        
        # 创建元数据
        metadata = {
            'dataset_info': {
                'name': '1962 Multi Extracted Dataset',
                'description': '从1962 Multi TI文档中提取的小规模精品图像数据集',
                'created_date': datetime.datetime.now().isoformat(),
                'source_directory': str(self.source_dir),
                'extraction_model': str(self.model_path),
                'model_size_mb': self.model_path.stat().st_size / (1024 * 1024),
                'dataset_scale': 'small'
            },
            'processing_performance': {
                'total_processing_time_seconds': self.stats['processing_time'],
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
            'image_characteristics': {},
            'detailed_results': processing_results['results'],
            'usage_recommendations': self._generate_usage_recommendations()
        }
        
        # 置信度分布
        total_detections = self.stats['total_detections']
        for level, count in self.stats['confidence_distribution'].items():
            threshold_map = {'high': '≥0.7', 'medium': '≥0.4', 'low': '≥0.25'}
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
        metadata_file = self.output_dir / "metadata" / f"dataset_metadata_{timestamp}.json"
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 创建README文件
        readme_file = self.output_dir / "README.md"
        self._create_detailed_readme(metadata, readme_file)
        
        logger.info(f"✅ 数据集元数据已生成: {metadata_file}")
        logger.info(f"📝 详细README文件已生成: {readme_file}")
        
        return metadata_file
    
    def _generate_usage_recommendations(self) -> List[str]:
        """生成使用建议"""
        recommendations = []
        
        total = self.stats['total_detections']
        if total == 0:
            return ["数据集为空，无法提供使用建议"]
        
        high_ratio = self.stats['confidence_distribution']['high'] / total
        medium_ratio = self.stats['confidence_distribution']['medium'] / total
        
        # 基于质量的建议
        if high_ratio > 0.6:
            recommendations.extend([
                "小规模高质量数据集，适合精细化应用",
                "高置信度图像可作为黄金标准样本",
                "适合用于模型验证和基准测试"
            ])
        elif high_ratio > 0.3:
            recommendations.extend([
                "质量良好的小规模数据集",
                "建议结合人工筛选使用"
            ])
        else:
            recommendations.extend([
                "建议对所有图像进行人工审核",
                "可作为补充数据集使用"
            ])
        
        # 基于数据量的建议
        if total < 10:
            recommendations.append("小规模数据集，建议用于特定场景测试或验证")
        else:
            recommendations.append("适中规模数据集，可用于小规模训练或验证")
        
        # 通用建议
        recommendations.extend([
            "每个图像都有详细的检测记录，便于质量控制",
            "所有原图都有对应的标注可视化",
            "适合用于精细化的图像分析任务"
        ])
        
        return recommendations
    
    def _create_detailed_readme(self, metadata: Dict[str, Any], readme_file: Path):
        """创建详细README文件"""
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write("# 1962 Multi Extracted Dataset\n\n")
            
            # 数据集概览
            f.write("## 数据集概览\n\n")
            f.write(f"**数据集名称**: {metadata['dataset_info']['name']}\n\n")
            f.write(f"**描述**: {metadata['dataset_info']['description']}\n\n")
            f.write(f"**规模**: 小规模精品数据集 ({metadata['dataset_statistics']['total_source_images']} 源图像 → {metadata['dataset_statistics']['total_extracted_images']} 提取图像)\n\n")
            f.write(f"**创建时间**: {metadata['dataset_info']['created_date']}\n\n")
            f.write(f"**提取模型**: {Path(metadata['dataset_info']['extraction_model']).name} ({metadata['dataset_info']['model_size_mb']:.2f} MB)\n\n")
            
            # 处理性能
            f.write("## 处理性能\n\n")
            perf = metadata['processing_performance']
            f.write(f"- **处理时间**: {perf['total_processing_time_seconds']:.1f} 秒\n")
            f.write(f"- **处理速度**: {perf['processing_speed_images_per_second']:.2f} 张/秒\n")
            f.write(f"- **成功率**: {perf['success_rate_percentage']:.1f}%\n\n")
            
            # 数据集统计
            f.write("## 数据集统计\n\n")
            stats = metadata['dataset_statistics']
            f.write(f"- **源图像数量**: {stats['total_source_images']} 张\n")
            f.write(f"- **提取图像数量**: {stats['total_extracted_images']} 张\n")
            f.write(f"- **平均检测数/源图像**: {stats['average_detections_per_source']:.2f}\n")
            f.write(f"- **数据集质量分数**: {stats['quality_score']:.1f}/100\n\n")
            
            # 置信度分布
            f.write("## 置信度分布\n\n")
            f.write("| 置信度等级 | 数量 | 阈值 | 占比 |\n")
            f.write("|-----------|------|------|------|\n")
            conf_names = {'high_confidence': '高', 'medium_confidence': '中等', 'low_confidence': '低'}
            for level, info in metadata['confidence_distribution'].items():
                level_name = conf_names.get(level, level)
                f.write(f"| {level_name} | {info['count']} | {info['threshold']} | {info['percentage']:.1f}% |\n")
            f.write("\n")
            
            # 详细处理结果
            f.write("## 详细处理结果\n\n")
            for i, result in enumerate(metadata['detailed_results'], 1):
                if result['success']:
                    f.write(f"### 图像 {i}: {result['original_file']}\n\n")
                    f.write(f"- **检测数量**: {result['detections_count']} 个\n")
                    if result['detections']:
                        f.write("- **检测详情**:\n")
                        for j, detection in enumerate(result['detections'], 1):
                            f.write(f"  {j}. 置信度: {detection['confidence']:.3f} ({detection['confidence_category']})\n")
                            f.write(f"     尺寸: {detection['width']}×{detection['height']} 像素\n")
                            f.write(f"     文件: {detection['extracted_file']}\n")
                    f.write("\n")
                else:
                    f.write(f"### 图像 {i}: {result['original_file']} (处理失败)\n\n")
                    f.write(f"- **错误**: {result.get('error', '未知错误')}\n\n")
            
            # 文件夹结构
            f.write("## 文件夹结构\n\n")
            f.write("```\n")
            f.write("1962 Multi_Extracted_Dataset/\n")
            f.write("├── extracted_images/          # 所有提取的图像\n")
            f.write("├── high_confidence/           # 高置信度图像 (≥0.7)\n")
            f.write("├── medium_confidence/         # 中等置信度图像 (≥0.4)\n")
            f.write("├── low_confidence/            # 低置信度图像 (≥0.25)\n")
            f.write("├── annotated_originals/       # 带标注的原图\n")
            f.write("├── metadata/                  # 数据集元数据\n")
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
            
            # 注意事项
            f.write("## 注意事项\n\n")
            f.write("1. 这是一个小规模精品数据集，每个样本都有详细记录\n")
            f.write("2. 所有原图都有对应的标注可视化文件\n")
            f.write("3. 提取的图像文件名包含详细的置信度信息\n")
            f.write("4. 建议在使用前查看annotated_originals中的标注图像\n")
            f.write("5. 适合用于精细化的图像分析和验证任务\n\n")
            
            f.write("---\n")
            f.write(f"*数据集生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    def run_small_scale_extraction_pipeline(self) -> Dict[str, Any]:
        """运行小规模数据集提取pipeline"""
        logger.info("🚀 启动1962 Multi小规模数据集提取Pipeline")
        logger.info("=" * 70)
        
        try:
            # 1. 获取图像文件列表
            image_files = self.get_image_files()
            if not image_files:
                logger.error("❌ 未找到任何图像文件")
                return {'success': False, 'error': 'No image files found'}
            
            # 2. 处理所有图像
            processing_results = self.process_all_images(image_files)
            
            # 3. 创建数据集元数据
            metadata_file = self.create_dataset_metadata(processing_results)
            
            logger.info("🎉 小规模数据集提取Pipeline执行成功!")
            logger.info(f"📋 数据集元数据: {metadata_file}")
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
    source_dir = "/Users/zhaoye/Desktop/1962 Multi"
    output_dir = "/Users/zhaoye/Desktop/1962 Multi_Extracted_Dataset"
    
    print("🔍 1962 Multi小规模数据集提取Pipeline")
    print("=" * 60)
    print(f"🎯 使用模型: {Path(model_path).name}")
    print(f"📂 源目录: {source_dir}")
    print(f"📁 输出目录: {output_dir}")
    print("🎨 功能: 小规模精品处理 + 详细记录 + 质量分析")
    print("=" * 60)
    
    try:
        # 创建小规模数据集提取器
        extractor = TI1962MultiExtractor(model_path, source_dir, output_dir)
        
        # 运行完整pipeline
        results = extractor.run_small_scale_extraction_pipeline()
        
        if results['success']:
            print("\n🎉 小规模数据集提取Pipeline执行成功!")
            print("=" * 60)
            
            stats = results['processing_results']['statistics']
            
            print(f"📊 处理结果:")
            print(f"  源图像数: {stats['total_processed']} 张")
            print(f"  成功处理: {stats['total_processed'] - len(stats['failed_images'])} 张")
            print(f"  总提取数: {stats['total_extracted']} 个图像")
            print(f"  处理速度: {stats['processing_speed']:.2f} 张/秒")
            print(f"  处理时间: {stats['processing_time']:.1f} 秒")
            
            print(f"\n📈 置信度分布:")
            conf_dist = stats['confidence_distribution']
            total_detections = stats['total_detections']
            if total_detections > 0:
                print(f"  高置信度 (≥0.7): {conf_dist['high']} 个 ({conf_dist['high']/total_detections*100:.1f}%)")
                print(f"  中置信度 (≥0.4): {conf_dist['medium']} 个 ({conf_dist['medium']/total_detections*100:.1f}%)")
                print(f"  低置信度 (≥0.25): {conf_dist['low']} 个 ({conf_dist['low']/total_detections*100:.1f}%)")
            
            print(f"\n🏆 数据集质量分数: {stats['dataset_quality_score']:.1f}/100")
            
            if stats['failed_images']:
                print(f"\n⚠️ 失败文件: {len(stats['failed_images'])} 个")
            
            print(f"\n📁 精品数据集结构:")
            print(f"  📸 所有提取图像: extracted_images/ ({stats['total_extracted']} 个)")
            print(f"  🟢 高置信度图像: high_confidence/ ({conf_dist['high']} 个)")
            print(f"  🟡 中置信度图像: medium_confidence/ ({conf_dist['medium']} 个)")
            print(f"  🔴 低置信度图像: low_confidence/ ({conf_dist['low']} 个)")
            print(f"  🖼️ 标注原图: annotated_originals/")
            print(f"  📋 详细元数据: metadata/")
            print(f"  📝 详细README: README.md")
            
            print("\n💡 下一步建议:")
            print("1. 查看 README.md 了解每个图像的详细处理结果")
            print("2. 检查 annotated_originals/ 中的标注可视化")
            print("3. 小规模数据集，建议逐个检查提取质量")
            print("4. 适合用于精细化验证和基准测试")
            
        else:
            print(f"❌ Pipeline执行失败: {results['error']}")
    
    except Exception as e:
        logger.error(f"主程序执行失败: {e}")
        print(f"❌ 执行失败: {e}")


if __name__ == "__main__":
    main()
