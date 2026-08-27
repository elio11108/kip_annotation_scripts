#!/usr/bin/env python3
"""
真实世界图像提取器
使用训练好的模型对1961 TI文件夹中的所有图像进行处理
提取检测到的图像区域并保存到指定文件夹
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
        logging.FileHandler('/Users/zhaoye/Desktop/1956_TI_Dataset/real_world_extraction.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class RealWorldImageExtractor:
    def __init__(self, model_path: str, source_dir: str, output_dir: str):
        self.model_path = Path(model_path)
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # 验证输入
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        if not self.source_dir.exists():
            raise FileNotFoundError(f"源目录不存在: {self.source_dir}")
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "extracted_images").mkdir(exist_ok=True)
        (self.output_dir / "annotated_originals").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # 加载模型
        self.model = None
        self.load_model()
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'total_detections': 0,
            'total_extracted': 0,
            'processing_time': 0,
            'failed_images': []
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
    
    def process_single_image(self, image_path: Path) -> Dict[str, Any]:
        """处理单个图像"""
        try:
            # 运行模型推理
            results = self.model(str(image_path), conf=0.25, iou=0.45, verbose=False)
            result = results[0]
            
            # 加载原图像
            original_image = cv2.imread(str(image_path))
            if original_image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            original_pil = Image.open(image_path)
            
            detections = []
            extracted_images = []
            
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
                    
                    # 提取图像区域
                    extracted_region = original_pil.crop((x1, y1, x2, y2))
                    
                    # 生成提取图像的文件名
                    extracted_filename = f"{image_path.stem}_extracted_{i+1}_conf{conf:.3f}.jpg"
                    extracted_path = self.output_dir / "extracted_images" / extracted_filename
                    
                    # 保存提取的图像
                    extracted_region.save(extracted_path, 'JPEG', quality=95)
                    
                    # 记录检测信息
                    detection_info = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'area': (x2 - x1) * (y2 - y1),
                        'extracted_file': extracted_filename
                    }
                    detections.append(detection_info)
                    extracted_images.append(extracted_path)
                    
                    # 在原图上绘制边界框
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # 添加置信度标签
                    label = f"Image: {conf:.3f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(annotated_image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                # 保存带标注的原图
                annotated_filename = f"{image_path.stem}_annotated.jpg"
                annotated_path = self.output_dir / "annotated_originals" / annotated_filename
                cv2.imwrite(str(annotated_path), annotated_image, 
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # 返回处理结果
            return {
                'success': True,
                'original_file': image_path.name,
                'detections_count': len(detections),
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
        """批量处理图像"""
        logger.info(f"🚀 开始批量处理 {len(image_files)} 个图像文件")
        logger.info(f"📦 批次大小: {batch_size}")
        
        start_time = datetime.datetime.now()
        all_results = []
        
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(image_files) + batch_size - 1) // batch_size
            
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
                
                # 显示进度
                overall_progress = ((i + j + 1) / len(image_files)) * 100
                if (j + 1) % 10 == 0 or (j + 1) == len(batch_files):
                    logger.info(f"   批次进度: {j+1}/{len(batch_files)} "
                              f"(总进度: {overall_progress:.1f}%)")
            
            # 批次完成统计
            batch_detections = sum(r['detections_count'] for r in batch_results if r['success'])
            batch_failures = sum(1 for r in batch_results if not r['success'])
            
            logger.info(f"✅ 批次 {batch_num} 完成: "
                       f"{len(batch_files) - batch_failures} 成功, "
                       f"{batch_failures} 失败, "
                       f"{batch_detections} 个检测")
        
        # 计算总体统计
        end_time = datetime.datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        
        success_count = sum(1 for r in all_results if r['success'])
        failure_count = len(all_results) - success_count
        
        logger.info("🎉 批量处理完成!")
        logger.info(f"📊 处理统计:")
        logger.info(f"   总文件数: {len(image_files)}")
        logger.info(f"   成功处理: {success_count}")
        logger.info(f"   处理失败: {failure_count}")
        logger.info(f"   总检测数: {self.stats['total_detections']}")
        logger.info(f"   提取图像数: {self.stats['total_extracted']}")
        logger.info(f"   处理时间: {self.stats['processing_time']:.1f} 秒")
        logger.info(f"   处理速度: {len(image_files) / self.stats['processing_time']:.2f} 张/秒")
        
        return {
            'results': all_results,
            'statistics': self.stats,
            'timestamp': end_time.isoformat()
        }
    
    def generate_extraction_report(self, processing_results: Dict[str, Any]) -> Path:
        """生成提取报告"""
        logger.info("📋 生成提取报告...")
        
        # 创建详细报告
        report = {
            'extraction_session': {
                'timestamp': datetime.datetime.now().isoformat(),
                'model_path': str(self.model_path),
                'source_directory': str(self.source_dir),
                'output_directory': str(self.output_dir),
                'model_size_mb': self.model_path.stat().st_size / (1024 * 1024)
            },
            'processing_results': processing_results,
            'summary': self._create_extraction_summary(processing_results),
            'file_structure': self._analyze_output_structure()
        }
        
        # 保存JSON报告
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / "reports" / f"extraction_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 创建文本摘要
        text_report_file = self.output_dir / "reports" / f"extraction_summary_{timestamp}.txt"
        self._create_text_extraction_report(report, text_report_file)
        
        logger.info(f"✅ 提取报告已生成: {report_file}")
        logger.info(f"📝 文本摘要已生成: {text_report_file}")
        
        return report_file
    
    def _create_extraction_summary(self, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建提取摘要"""
        stats = processing_results['statistics']
        results = processing_results['results']
        
        # 分析置信度分布
        all_confidences = []
        for result in results:
            if result['success']:
                for detection in result['detections']:
                    all_confidences.append(detection['confidence'])
        
        # 分析检测区域大小
        all_areas = []
        for result in results:
            if result['success']:
                for detection in result['detections']:
                    all_areas.append(detection['area'])
        
        summary = {
            'total_source_images': stats['total_processed'],
            'successful_processing': stats['total_processed'] - len(stats['failed_images']),
            'failed_processing': len(stats['failed_images']),
            'total_detections': stats['total_detections'],
            'total_extracted_images': stats['total_extracted'],
            'processing_speed_images_per_second': stats['total_processed'] / stats['processing_time'] if stats['processing_time'] > 0 else 0,
            'average_detections_per_image': stats['total_detections'] / stats['total_processed'] if stats['total_processed'] > 0 else 0,
        }
        
        if all_confidences:
            summary.update({
                'confidence_stats': {
                    'mean': float(np.mean(all_confidences)),
                    'std': float(np.std(all_confidences)),
                    'min': float(np.min(all_confidences)),
                    'max': float(np.max(all_confidences)),
                    'median': float(np.median(all_confidences))
                }
            })
        
        if all_areas:
            summary.update({
                'area_stats': {
                    'mean': float(np.mean(all_areas)),
                    'std': float(np.std(all_areas)),
                    'min': float(np.min(all_areas)),
                    'max': float(np.max(all_areas)),
                    'median': float(np.median(all_areas))
                }
            })
        
        return summary
    
    def _analyze_output_structure(self) -> Dict[str, Any]:
        """分析输出目录结构"""
        structure = {}
        
        for subdir in ['extracted_images', 'annotated_originals', 'reports']:
            subdir_path = self.output_dir / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob('*'))
                structure[subdir] = {
                    'file_count': len(files),
                    'total_size_mb': sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
                }
        
        return structure
    
    def _create_text_extraction_report(self, report: Dict[str, Any], output_file: Path):
        """创建文本提取报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("真实世界图像提取报告\n")
            f.write("=" * 70 + "\n\n")
            
            # 基本信息
            session = report['extraction_session']
            f.write("📋 提取会话信息\n")
            f.write("-" * 40 + "\n")
            f.write(f"时间戳: {session['timestamp']}\n")
            f.write(f"模型路径: {session['model_path']}\n")
            f.write(f"模型大小: {session['model_size_mb']:.2f} MB\n")
            f.write(f"源目录: {session['source_directory']}\n")
            f.write(f"输出目录: {session['output_directory']}\n\n")
            
            # 处理结果摘要
            summary = report['summary']
            f.write("📊 处理结果摘要\n")
            f.write("-" * 40 + "\n")
            f.write(f"源图像总数: {summary['total_source_images']} 张\n")
            f.write(f"成功处理: {summary['successful_processing']} 张\n")
            f.write(f"处理失败: {summary['failed_processing']} 张\n")
            f.write(f"总检测数: {summary['total_detections']} 个\n")
            f.write(f"提取图像数: {summary['total_extracted_images']} 个\n")
            f.write(f"处理速度: {summary['processing_speed_images_per_second']:.2f} 张/秒\n")
            f.write(f"平均检测数/图像: {summary['average_detections_per_image']:.2f}\n\n")
            
            # 置信度统计
            if 'confidence_stats' in summary:
                conf_stats = summary['confidence_stats']
                f.write("📈 置信度统计\n")
                f.write("-" * 40 + "\n")
                f.write(f"平均置信度: {conf_stats['mean']:.3f}\n")
                f.write(f"置信度标准差: {conf_stats['std']:.3f}\n")
                f.write(f"最低置信度: {conf_stats['min']:.3f}\n")
                f.write(f"最高置信度: {conf_stats['max']:.3f}\n")
                f.write(f"置信度中位数: {conf_stats['median']:.3f}\n\n")
            
            # 区域大小统计
            if 'area_stats' in summary:
                area_stats = summary['area_stats']
                f.write("📐 检测区域统计\n")
                f.write("-" * 40 + "\n")
                f.write(f"平均区域大小: {area_stats['mean']:.0f} 像素²\n")
                f.write(f"区域大小标准差: {area_stats['std']:.0f} 像素²\n")
                f.write(f"最小区域: {area_stats['min']:.0f} 像素²\n")
                f.write(f"最大区域: {area_stats['max']:.0f} 像素²\n")
                f.write(f"区域大小中位数: {area_stats['median']:.0f} 像素²\n\n")
            
            # 输出文件结构
            f.write("📁 输出文件结构\n")
            f.write("-" * 40 + "\n")
            structure = report['file_structure']
            for folder, info in structure.items():
                f.write(f"{folder}:\n")
                f.write(f"  文件数量: {info['file_count']}\n")
                f.write(f"  总大小: {info['total_size_mb']:.2f} MB\n")
            
            # 失败文件列表
            failed_files = report['processing_results']['statistics']['failed_images']
            if failed_files:
                f.write(f"\n❌ 处理失败的文件 ({len(failed_files)} 个)\n")
                f.write("-" * 40 + "\n")
                for i, failed_file in enumerate(failed_files[:10], 1):  # 只显示前10个
                    f.write(f"{i}. {failed_file}\n")
                if len(failed_files) > 10:
                    f.write(f"... 还有 {len(failed_files) - 10} 个失败文件\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("报告生成完成\n")
            f.write("=" * 70 + "\n")
    
    def run_extraction_pipeline(self) -> Dict[str, Any]:
        """运行完整的提取pipeline"""
        logger.info("🚀 启动真实世界图像提取Pipeline")
        logger.info("=" * 70)
        
        try:
            # 1. 获取图像文件列表
            image_files = self.get_image_files()
            if not image_files:
                logger.error("❌ 未找到任何图像文件")
                return {'success': False, 'error': 'No image files found'}
            
            # 2. 批量处理图像
            processing_results = self.process_batch(image_files)
            
            # 3. 生成报告
            report_file = self.generate_extraction_report(processing_results)
            
            logger.info("🎉 图像提取Pipeline执行成功!")
            logger.info(f"📋 详细报告: {report_file}")
            logger.info(f"📁 输出目录: {self.output_dir}")
            
            return {
                'success': True,
                'report_file': report_file,
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
    source_dir = "/Users/zhaoye/Desktop/1961 TI"
    output_dir = "/Users/zhaoye/Desktop/1961 TI_Extracted"
    
    print("🔍 真实世界图像提取Pipeline")
    print("=" * 60)
    print(f"🎯 使用模型: {Path(model_path).name}")
    print(f"📂 源目录: {source_dir}")
    print(f"📁 输出目录: {output_dir}")
    print("🎨 功能: 批量检测 + 图像提取 + 标注可视化 + 详细报告")
    print("=" * 60)
    
    try:
        # 创建图像提取器
        extractor = RealWorldImageExtractor(model_path, source_dir, output_dir)
        
        # 运行完整pipeline
        results = extractor.run_extraction_pipeline()
        
        if results['success']:
            print("\n🎉 图像提取Pipeline执行成功!")
            print("=" * 60)
            
            stats = results['processing_results']['statistics']
            summary = results['processing_results']['statistics']
            
            print(f"📊 提取结果:")
            print(f"  源图像数: {stats['total_processed']} 张")
            print(f"  成功处理: {stats['total_processed'] - len(stats['failed_images'])} 张")
            print(f"  总检测数: {stats['total_detections']} 个")
            print(f"  提取图像数: {stats['total_extracted']} 个")
            print(f"  处理速度: {stats['total_processed'] / stats['processing_time']:.2f} 张/秒")
            print(f"  处理时间: {stats['processing_time']:.1f} 秒")
            
            if stats['failed_images']:
                print(f"  失败文件: {len(stats['failed_images'])} 个")
            
            print(f"\n📁 输出文件:")
            print(f"  📸 提取的图像: {output_dir}/extracted_images/")
            print(f"  🖼️ 标注的原图: {output_dir}/annotated_originals/")
            print(f"  📋 详细报告: {results['report_file']}")
            
            print("\n💡 下一步建议:")
            print("1. 查看 extracted_images/ 文件夹中的提取结果")
            print("2. 检查 annotated_originals/ 中的检测可视化")
            print("3. 阅读详细报告了解处理统计")
            print("4. 根据结果评估模型在真实数据上的表现")
            
        else:
            print(f"❌ Pipeline执行失败: {results['error']}")
    
    except Exception as e:
        logger.error(f"主程序执行失败: {e}")
        print(f"❌ 执行失败: {e}")


if __name__ == "__main__":
    main()
