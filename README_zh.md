# 1956 TI 图像分割数据集项目

这是一个完整的计算机视觉Pipeline，用于从1956 TI文档图像中检测和提取纯图像区域。

## 项目结构

```
1956_TI_Dataset/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖列表
├── setup.py                     # 安装脚本
├── pipeline_manager.py          # 主要Pipeline管理脚本
├── image_annotation_tool.py     # 图像标注工具
├── dataset_processor.py         # 数据集处理脚本
├── train_model.py              # 模型训练脚本
├── inference_model.py          # 推理脚本
├── raw_images/                 # 原始图像
├── annotations/                # 标注文件
├── processed_data/             # 处理后的数据集
│   ├── train/                  # 训练集
│   ├── val/                    # 验证集
│   └── test/                   # 测试集
├── models/                     # 训练好的模型
├── results/                    # 训练结果
└── inference_results/          # 推理结果
```

## 快速开始

### 1. 安装依赖

```bash
cd /Users/zhaoye/Desktop/1956_TI_Dataset
pip install -r requirements.txt
```

### 2. 运行完整Pipeline

```bash
python pipeline_manager.py --step full --epochs 50 --batch-size 16
```

### 3. 分步执行

#### 步骤1: 图像标注
```bash
python image_annotation_tool.py
```
- 导入原始图像
- 手动框选图像区域
- 保存标注数据

#### 步骤2: 处理数据集
```bash
python pipeline_manager.py --step process
```
- 将标注转换为YOLO格式
- 分割训练/验证/测试集
- 生成配置文件

#### 步骤3: 训练模型
```bash
python pipeline_manager.py --step train --model-size n --epochs 50
```
- 使用YOLOv8训练检测模型
- 保存最佳模型

#### 步骤4: 推理
```bash
python pipeline_manager.py --step inference
```
- 使用训练好的模型检测图像
- 提取图像区域
- 生成结果报告

## 详细说明

### 图像标注工具

`image_annotation_tool.py` 提供了一个基于tkinter的图形界面：

- **导入图像**: 将原始图像复制到数据集目录
- **标注框选**: 左键拖拽框选图像区域
- **删除标注**: 右键点击删除标注框
- **保存标注**: 将标注保存为JSON格式

### 数据集处理

`dataset_processor.py` 负责：

- 加载所有标注文件
- 转换为YOLO格式 (归一化坐标)
- 按7:2:1比例分割数据集
- 生成训练配置文件
- 计算数据集统计信息

### 模型训练

`train_model.py` 使用YOLOv8进行训练：

- 支持多种模型大小 (n, s, m, l, x)
- 自动保存最佳模型
- 生成训练报告和可视化
- 支持GPU加速

### 推理和提取

`inference_model.py` 进行图像检测和提取：

- 加载训练好的模型
- 批量处理图像
- 可视化检测结果
- 提取并保存图像区域
- 生成详细报告

### Pipeline管理

`pipeline_manager.py` 统一管理整个流程：

- 检查依赖和状态
- 按步骤执行或一键运行
- 跟踪进度和结果
- 生成摘要报告

## 参数配置

### 训练参数
- `--model-size`: YOLOv8模型大小 (n/s/m/l/x)
- `--epochs`: 训练轮数 (默认50)
- `--batch-size`: 批次大小 (默认16)
- `--img-size`: 输入图像大小 (默认640)

### 推理参数
- 置信度阈值: 0.25 (可在代码中调整)
- IoU阈值: 0.45
- 最小区域面积: 1000像素

## 输出结果

### 训练输出
- `models/best_yolov8n.pt`: 最佳模型文件
- `results/`: 训练日志和可视化
- `training_report_*.json`: 训练报告

### 推理输出
- `inference_results/extracted_images/`: 提取的图像
- `inference_results/*_visualization.jpg`: 可视化结果
- `inference_results/inference_results.json`: 详细结果
- `inference_results/summary_report.txt`: 摘要报告

## 性能优化

### GPU加速
确保安装了CUDA版本的PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 内存优化
- 减小批次大小
- 使用较小的模型 (yolov8n)
- 启用混合精度训练

### 数据增强
模型训练时会自动应用数据增强:
- 随机缩放和裁剪
- 颜色变换
- 几何变换

## 故障排除

### 常见问题

1. **导入错误**: 确保所有依赖都已正确安装
2. **内存不足**: 减小批次大小或使用更小的模型
3. **标注质量**: 确保标注框准确且完整
4. **训练不收敛**: 增加训练轮数或调整学习率

### 日志查看
所有脚本都会输出详细日志，可以通过日志信息定位问题。

## 扩展功能

### 添加新类别
1. 修改标注工具以支持多类别
2. 更新数据集处理脚本
3. 调整训练配置

### 自定义模型
可以替换YOLOv8为其他检测模型:
- Faster R-CNN
- SSD
- RetinaNet

### 后处理优化
- 非极大值抑制 (NMS)
- 置信度过滤
- 区域合并

## 许可证

本项目仅供学术研究使用。

## 联系信息

如有问题或建议，请联系项目维护者。

