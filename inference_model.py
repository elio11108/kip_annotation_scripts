#!/usr/bin/env python3
"""
1956 TI 数据集推理脚本
使用训练好的模型对新图像进行图像区域检测和提取
"""

import os
import sys
from pathlib import Path
import json
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageInference:
    def __init__(self, model_path, dataset_dir):
        self.model_path = Path(model_path)
        self.dataset_dir = Path(dataset_dir)
        self.results_dir = self.dataset_dir / "inference_results"
        self.extracted_dir = self.results_dir / "extracted_images"
        
        # 创建结果目录
        self.results_dir.mkdir(exist_ok=True)
        self.extracted_dir.mkdir(exist_ok=True)
        
        # 加载模型
        self.model = None
        self.load_model()
        
        # 推理参数
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.max_det = 300
        
    def load_model(self):
        """加载训练好的模型"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        try:
            self.model = YOLO(str(self.model_path))
            logger.info(f"成功加载模型: {self.model_path}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def predict_single_image(self, image_path, save_visualization=True):
        """对单张图像进行预测"""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        logger.info(f"正在处理图像: {image_path.name}")
        
        # 运行推理
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            save=False,
            save_txt=False,
            save_conf=True,
            save_crop=False,
            show=False,
            verbose=False
        )
        
        if not results:
            logger.warning(f"未在图像 {image_path.name} 中检测到任何对象")
            return None
        
        result = results[0]  # 获取第一个结果
        
        # 解析检测结果
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标
            confidences = result.boxes.conf.cpu().numpy()  # 置信度
            classes = result.boxes.cls.cpu().numpy()  # 类别
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = box.astype(int)
                detections.append({
                    'id': i,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class': int(cls),
                    'class_name': 'image'  # 我们只有一个类别
                })
        
        # 保存可视化结果
        if save_visualization and detections:
            self.visualize_detections(image_path, detections)
        
        return detections
    
    def visualize_detections(self, image_path, detections):
        """可视化检测结果"""
        # 加载图像
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # 设置字体
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # 绘制检测框
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            # 绘制标签
            label = f"Image: {confidence:.2f}"
            text_bbox = draw.textbbox((x1, y1-25), label, font=font)
            draw.rectangle(text_bbox, fill='red')
            draw.text((x1, y1-25), label, fill='white', font=font)
        
        # 保存可视化结果
        vis_path = self.results_dir / f"{image_path.stem}_visualization.jpg"
        image.save(vis_path, 'JPEG', quality=95)
        logger.info(f"可视化结果已保存到: {vis_path}")
    
    def extract_image_regions(self, image_path, detections, min_area=1000):
        """提取检测到的图像区域"""
        if not detections:
            return []
        
        # 加载原图
        original_image = Image.open(image_path)
        extracted_images = []
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # 检查区域大小
            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                logger.warning(f"跳过小区域: 面积 {area} < {min_area}")
                continue
            
            # 裁剪图像
            cropped_image = original_image.crop((x1, y1, x2, y2))
            
            # 保存提取的图像
            filename = f"{Path(image_path).stem}_extracted_{i:03d}_conf{confidence:.2f}.png"
            save_path = self.extracted_dir / filename
            cropped_image.save(save_path, 'PNG')
            
            extracted_info = {
                'filename': filename,
                'save_path': str(save_path),
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'area': area,
                'size': [x2-x1, y2-y1]
            }
            extracted_images.append(extracted_info)
            
            logger.info(f"提取图像: {filename} (大小: {x2-x1}x{y2-y1}, 置信度: {confidence:.2f})")
        
        return extracted_images
    
    def process_folder(self, input_folder, output_json=True):
        """处理整个文件夹的图像"""
        input_folder = Path(input_folder)
        if not input_folder.exists():
            raise FileNotFoundError(f"输入文件夹不存在: {input_folder}")
        
        # 获取所有图像文件
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(input_folder.glob(f'*{ext}')))
            image_files.extend(list(input_folder.glob(f'*{ext.upper()}')))
        
        image_files.sort()
        logger.info(f"找到 {len(image_files)} 个图像文件")
        
        if not image_files:
            logger.warning("未找到任何图像文件")
            return
        
        # 处理结果
        all_results = {
            'model_path': str(self.model_path),
            'input_folder': str(input_folder),
            'total_images': len(image_files),
            'total_detections': 0,
            'total_extracted': 0,
            'results': []
        }
        
        # 处理每个图像
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"进度: {i}/{len(image_files)} - {image_file.name}")
            
            try:
                # 预测
                detections = self.predict_single_image(image_file)
                
                if detections:
                    # 提取图像区域
                    extracted = self.extract_image_regions(image_file, detections)
                    
                    # 记录结果
                    result_info = {
                        'image_name': image_file.name,
                        'image_path': str(image_file),
                        'num_detections': len(detections),
                        'num_extracted': len(extracted),
                        'detections': detections,
                        'extracted_images': extracted
                    }
                    
                    all_results['total_detections'] += len(detections)
                    all_results['total_extracted'] += len(extracted)
                else:
                    result_info = {
                        'image_name': image_file.name,
                        'image_path': str(image_file),
                        'num_detections': 0,
                        'num_extracted': 0,
                        'detections': [],
                        'extracted_images': []
                    }
                
                all_results['results'].append(result_info)
                
            except Exception as e:
                logger.error(f"处理 {image_file.name} 时出错: {e}")
                result_info = {
                    'image_name': image_file.name,
                    'image_path': str(image_file),
                    'error': str(e),
                    'num_detections': 0,
                    'num_extracted': 0
                }
                all_results['results'].append(result_info)
        
        # 保存结果
        if output_json:
            results_file = self.results_dir / 'inference_results.json'
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            logger.info(f"推理结果已保存到: {results_file}")
        
        # 生成摘要报告
        self.generate_summary_report(all_results)
        
        return all_results
    
    def generate_summary_report(self, results):
        """生成摘要报告"""
        report = f"""
1956 TI 图像检测推理报告
========================

模型信息:
- 模型路径: {results['model_path']}
- 置信度阈值: {self.conf_threshold}
- IoU阈值: {self.iou_threshold}

处理统计:
- 总图像数: {results['total_images']}
- 总检测数: {results['total_detections']}
- 总提取数: {results['total_extracted']}
- 平均每张图检测数: {results['total_detections']/results['total_images']:.2f}
- 平均每张图提取数: {results['total_extracted']/results['total_images']:.2f}

详细结果:
"""
        
        for result in results['results']:
            if 'error' not in result:
                report += f"- {result['image_name']}: {result['num_detections']} 检测, {result['num_extracted']} 提取\n"
            else:
                report += f"- {result['image_name']}: 处理失败 - {result['error']}\n"
        
        # 保存报告
        report_file = self.results_dir / 'summary_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"摘要报告已保存到: {report_file}")
        print(report)

def main():
    """主函数"""
    dataset_dir = "/Users/zhaoye/Desktop/1956_TI_Dataset"
    
    # 查找最佳模型
    models_dir = Path(dataset_dir) / "models"
    model_files = list(models_dir.glob("best_*.pt"))
    
    if not model_files:
        logger.error("未找到训练好的模型文件，请先运行 train_model.py")
        return
    
    # 使用第一个找到的模型
    model_path = model_files[0]
    logger.info(f"使用模型: {model_path}")
    
    # 创建推理器
    inferencer = ImageInference(model_path, dataset_dir)
    
    # 设置推理参数
    inferencer.conf_threshold = 0.3  # 可以调整置信度阈值
    inferencer.iou_threshold = 0.45
    
    # 选择要处理的图像文件夹
    input_folder = "/Users/zhaoye/Desktop/1956 TI"  # 原始图像文件夹
    
    if not Path(input_folder).exists():
        logger.error(f"输入文件夹不存在: {input_folder}")
        return
    
    try:
        logger.info("开始推理...")
        results = inferencer.process_folder(input_folder)
        
        print("\n=== 推理完成 ===")
        print(f"处理了 {results['total_images']} 张图像")
        print(f"检测到 {results['total_detections']} 个图像区域")
        print(f"提取了 {results['total_extracted']} 个图像")
        print(f"结果保存在: {inferencer.results_dir}")
        
    except Exception as e:
        logger.error(f"推理失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

