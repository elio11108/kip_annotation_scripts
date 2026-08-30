#!/usr/bin/env python3
"""
1956 TI 数据集图像标注工具
使用tkinter创建简单的图像标注界面，用于框选图像中的纯图片部分
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
import os
from pathlib import Path
import shutil

class ImageAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("1956 TI 图像标注工具")
        self.root.geometry("1200x800")
        
        # 数据路径
        self.dataset_dir = Path("/Users/zhaoye/Desktop/1956_TI_Dataset")
        self.raw_images_dir = self.dataset_dir / "raw_images"
        self.annotations_dir = self.dataset_dir / "annotations"
        
        # 确保目录存在
        self.raw_images_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)
        
        # 当前状态
        self.current_image = None
        self.current_image_path = None
        self.image_list = []
        self.current_index = 0
        self.annotations = {}
        
        # 标注状态
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_bbox = None
        self.bboxes = []
        
        # 缩放因子
        self.scale_factor = 1.0
        self.canvas_width = 800
        self.canvas_height = 600
        
        self.setup_ui()
        self.load_images()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 文件操作按钮
        ttk.Button(control_frame, text="导入图像", command=self.import_images).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="保存标注", command=self.save_annotations).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="加载标注", command=self.load_annotations).pack(side=tk.LEFT, padx=(0, 5))
        
        # 导航按钮
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(side=tk.RIGHT)
        
        ttk.Button(nav_frame, text="上一张", command=self.prev_image).pack(side=tk.LEFT, padx=(0, 5))
        self.image_info_label = ttk.Label(nav_frame, text="0/0")
        self.image_info_label.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="下一张", command=self.next_image).pack(side=tk.LEFT)
        
        # 图像显示区域
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # 画布
        self.canvas = tk.Canvas(image_frame, bg='white', width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 滚动条
        v_scrollbar = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(fill=tk.X)
        self.canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_bbox)
        self.canvas.bind("<ButtonRelease-1>", self.end_drawing)
        self.canvas.bind("<Button-3>", self.delete_bbox)  # 右键删除
        self.canvas.bind("<Control-Button-1>", self.delete_bbox)  # Control+左键删除
        
        # 绑定键盘事件
        self.root.bind("<Key>", self.key_pressed)
        self.canvas.focus_set()  # 设置焦点以接收键盘事件
        
        # 信息面板
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 标注信息
        self.info_text = tk.Text(info_frame, height=4, width=50)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 操作说明
        instructions = """
操作说明（修复版）：
1. 左键拖拽：创建标注框
2. 右键点击红框：删除标注框
3. Control+左键：删除标注框
4. D键：删除最后一个标注框
5. C键：清除所有标注框
6. S键：保存当前标注
7. 退格键/Delete键：删除最后一个标注框

提示：如果右键删除不工作，
请使用键盘快捷键！
        """
        
        instruction_label = ttk.Label(info_frame, text=instructions, justify=tk.LEFT)
        instruction_label.pack(side=tk.RIGHT, padx=(10, 0))
        
    def import_images(self):
        """导入图像到数据集"""
        source_dir = filedialog.askdirectory(title="选择包含图像的文件夹")
        if not source_dir:
            return
            
        # 复制图像文件到raw_images目录
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        copied_count = 0
        
        for file_path in Path(source_dir).rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                dest_path = self.raw_images_dir / file_path.name
                if not dest_path.exists():
                    shutil.copy2(file_path, dest_path)
                    copied_count += 1
        
        messagebox.showinfo("导入完成", f"成功导入 {copied_count} 张图像")
        self.load_images()
        
    def load_images(self):
        """加载图像列表"""
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        self.image_list = []
        
        for ext in image_extensions:
            self.image_list.extend(list(self.raw_images_dir.glob(f'*{ext}')))
            self.image_list.extend(list(self.raw_images_dir.glob(f'*{ext.upper()}')))
        
        self.image_list.sort()
        self.current_index = 0
        
        if self.image_list:
            self.load_current_image()
            
        self.update_info_label()
        
    def load_current_image(self):
        """加载当前图像"""
        if not self.image_list:
            return
            
        self.current_image_path = self.image_list[self.current_index]
        self.current_image = Image.open(self.current_image_path)
        
        # 计算缩放因子
        img_width, img_height = self.current_image.size
        scale_x = self.canvas_width / img_width
        scale_y = self.canvas_height / img_height
        self.scale_factor = min(scale_x, scale_y, 1.0)  # 不放大，只缩小
        
        # 缩放图像
        new_width = int(img_width * self.scale_factor)
        new_height = int(img_height * self.scale_factor)
        display_image = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 转换为tkinter格式
        self.photo = ImageTk.PhotoImage(display_image)
        
        # 清除画布并显示图像
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # 重置标注
        self.bboxes = []
        self.load_existing_annotations()
        
    def load_existing_annotations(self):
        """加载已有的标注"""
        if not self.current_image_path:
            return
            
        annotation_file = self.annotations_dir / f"{self.current_image_path.stem}.json"
        if annotation_file.exists():
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.bboxes = data.get('bboxes', [])
                self.draw_all_bboxes()
                
    def start_drawing(self, event):
        """开始绘制标注框"""
        # 未载入图像时不允许画框（此时 current_image 为 None）
        if self.current_image is None:
            return
        self.drawing = True
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        
    def draw_bbox(self, event):
        """绘制标注框"""
        if not self.drawing:
            return
            
        # 删除当前绘制的临时框
        if self.current_bbox:
            self.canvas.delete(self.current_bbox)
            
        # 绘制新的临时框
        current_x = self.canvas.canvasx(event.x)
        current_y = self.canvas.canvasy(event.y)
        
        self.current_bbox = self.canvas.create_rectangle(
            self.start_x, self.start_y, current_x, current_y,
            outline='red', width=2, tags="temp_bbox"
        )
        
    def end_drawing(self, event):
        """结束绘制标注框"""
        if not self.drawing:
            return
            
        self.drawing = False
        
        # 获取最终坐标
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        # 确保坐标顺序正确
        x1, x2 = sorted([self.start_x, end_x])
        y1, y2 = sorted([self.start_y, end_y])
        
        # 检查框的大小
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            if self.current_bbox:
                self.canvas.delete(self.current_bbox)
            self.current_bbox = None
            return
            
        # 转换为原图坐标（每个坐标双边裁剪到图像边界内，防止画布坐标换算产生负值或越界；
        # 历史标注中有 16/197 个框因缺少此裁剪而略微越界，见 README 的 Annotation quality note）
        img_width, img_height = self.current_image.size
        orig_x1 = min(max(0, int(x1 / self.scale_factor)), img_width)
        orig_y1 = min(max(0, int(y1 / self.scale_factor)), img_height)
        orig_x2 = min(max(0, int(x2 / self.scale_factor)), img_width)
        orig_y2 = min(max(0, int(y2 / self.scale_factor)), img_height)

        # 裁剪后再检查一次退化框（框完全画在图像外的画布空白处时会被裁成零宽/零高），丢弃
        if orig_x2 - orig_x1 < 1 or orig_y2 - orig_y1 < 1:
            if self.current_bbox:
                self.canvas.delete(self.current_bbox)
            self.current_bbox = None
            return

        # 把画布上的临时框校正为裁剪后的坐标，使显示与保存一致
        self.canvas.coords(self.current_bbox,
                           orig_x1 * self.scale_factor, orig_y1 * self.scale_factor,
                           orig_x2 * self.scale_factor, orig_y2 * self.scale_factor)
        
        # 保存标注框
        bbox_info = {
            'x1': orig_x1, 'y1': orig_y1,
            'x2': orig_x2, 'y2': orig_y2,
            'canvas_id': self.current_bbox
        }
        self.bboxes.append(bbox_info)
        
        # 更新显示
        self.canvas.itemconfig(self.current_bbox, tags="bbox")
        self.current_bbox = None
        self.update_info_display()
        
    def delete_bbox(self, event):
        """删除标注框 - 修复版本"""
        print(f"右键点击位置: ({event.x}, {event.y})")
        
        # 获取点击位置的画布坐标
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # 查找点击位置附近的所有对象（扩大搜索范围）
        overlapping_items = self.canvas.find_overlapping(
            canvas_x - 10, canvas_y - 10, 
            canvas_x + 10, canvas_y + 10
        )
        
        print(f"找到重叠对象: {overlapping_items}")
        
        # 查找标注框对象
        deleted = False
        for item in overlapping_items:
            tags = self.canvas.gettags(item)
            print(f"对象 {item} 的标签: {tags}")
            
            if "bbox" in tags:
                # 找到对应的标注框并删除
                for i, bbox in enumerate(self.bboxes):
                    if bbox.get('canvas_id') == item:
                        self.canvas.delete(item)
                        del self.bboxes[i]
                        self.update_info_display()
                        print(f"成功删除标注框: {item}")
                        deleted = True
                        break
                if deleted:
                    break
        
        # 如果没有找到标注框，提供备用删除方式
        if not deleted and self.bboxes:
            from tkinter import messagebox
            choice = messagebox.askyesno("删除确认", 
                f"点击位置没有找到标注框。\n"
                f"当前有 {len(self.bboxes)} 个标注框。\n"
                f"是否删除最后一个标注框？")
            if choice:
                self.delete_last_bbox()
    
    def delete_last_bbox(self):
        """删除最后一个标注框"""
        if self.bboxes:
            last_bbox = self.bboxes[-1]
            canvas_id = last_bbox.get('canvas_id')
            if canvas_id:
                self.canvas.delete(canvas_id)
            self.bboxes.pop()
            self.update_info_display()
            print("删除了最后一个标注框")
    
    def key_pressed(self, event):
        """处理键盘事件"""
        print(f"按键: {event.char}, 键码: {event.keysym}")
        
        if event.char.lower() == 'd':
            # D键删除最后一个标注框
            self.delete_last_bbox()
        elif event.char.lower() == 'c':
            # C键清除所有标注框
            self.clear_all_annotations()
        elif event.char.lower() == 's':
            # S键保存标注
            self.save_annotations()
        elif event.keysym == 'BackSpace' or event.keysym == 'Delete':
            # 退格键或Delete键删除最后一个标注框
            self.delete_last_bbox()
    
    def clear_all_annotations(self):
        """清除所有标注框"""
        if self.bboxes:
            from tkinter import messagebox
            choice = messagebox.askyesno("确认", "确定要清除所有标注框吗？")
            if choice:
                # 删除所有画布对象
                for bbox in self.bboxes:
                    canvas_id = bbox.get('canvas_id')
                    if canvas_id:
                        self.canvas.delete(canvas_id)
                
                # 清除数据
                self.bboxes = []
                self.update_info_display()
                print("清除了所有标注框")
                    
    def draw_all_bboxes(self):
        """绘制所有标注框"""
        for bbox in self.bboxes:
            # 转换为画布坐标
            x1 = bbox['x1'] * self.scale_factor
            y1 = bbox['y1'] * self.scale_factor
            x2 = bbox['x2'] * self.scale_factor
            y2 = bbox['y2'] * self.scale_factor
            
            canvas_id = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline='red', width=2, tags="bbox"
            )
            bbox['canvas_id'] = canvas_id
            
    def prev_image(self):
        """上一张图像"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
            self.update_info_label()
            
    def next_image(self):
        """下一张图像"""
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.load_current_image()
            self.update_info_label()
            
    def update_info_label(self):
        """更新图像信息标签"""
        if self.image_list:
            self.image_info_label.config(text=f"{self.current_index + 1}/{len(self.image_list)}")
        else:
            self.image_info_label.config(text="0/0")
            
    def update_info_display(self):
        """更新信息显示"""
        if not self.current_image_path:
            return
            
        info = f"当前图像: {self.current_image_path.name}\n"
        info += f"图像尺寸: {self.current_image.size}\n"
        info += f"标注框数量: {len(self.bboxes)}\n"
        
        if self.bboxes:
            info += "标注框坐标:\n"
            for i, bbox in enumerate(self.bboxes):
                info += f"  {i+1}: ({bbox['x1']}, {bbox['y1']}) -> ({bbox['x2']}, {bbox['y2']})\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)
        
    def save_annotations(self):
        """保存当前图像的标注"""
        if not self.current_image_path or not self.bboxes:
            messagebox.showwarning("警告", "没有标注数据可保存")
            return
            
        # 准备标注数据
        annotation_data = {
            'image_path': str(self.current_image_path),
            'image_name': self.current_image_path.name,
            'image_size': {
                'width': self.current_image.size[0],
                'height': self.current_image.size[1]
            },
            'bboxes': [
                {
                    'x1': bbox['x1'], 'y1': bbox['y1'],
                    'x2': bbox['x2'], 'y2': bbox['y2'],
                    'label': 'image',  # 标签为图像
                    'area': (bbox['x2'] - bbox['x1']) * (bbox['y2'] - bbox['y1'])
                }
                for bbox in self.bboxes
            ],
            'num_annotations': len(self.bboxes)
        }
        
        # 保存到JSON文件
        annotation_file = self.annotations_dir / f"{self.current_image_path.stem}.json"
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, ensure_ascii=False, indent=2)
            
        messagebox.showinfo("成功", f"标注已保存到: {annotation_file}")
        
    def load_annotations(self):
        """加载所有标注文件"""
        annotation_files = list(self.annotations_dir.glob("*.json"))
        messagebox.showinfo("信息", f"找到 {len(annotation_files)} 个标注文件")

def main():
    root = tk.Tk()
    app = ImageAnnotationTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()
