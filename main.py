import tkinter as tk

from frames.frame1 import create_frame_1
from frames.frame2 import create_frame_2
from frames.frame3 import create_frame_3
from frames.frame4 import create_frame_4


import ttkbootstrap as ttk
from ttkbootstrap.constants import *

root = ttk.Window()
style = ttk.Style()
theme_names = style.theme_names()  # 以列表的形式返回多个主题名
theme_selection = ttk.Frame(root, padding=(1, 2, 2, 0))
theme_selection.pack(fill=X, expand=NO)
lbl = ttk.Label(theme_selection, text="选择主题:")
theme_cbo = ttk.Combobox(
    master=theme_selection,
    text=style.theme.name,
    values=theme_names,
)
theme_cbo.pack(padx=2, side=RIGHT)
theme_cbo.current(theme_names.index(style.theme.name))
lbl.pack(side=RIGHT)


def change_theme(event):
    theme_cbo_value = theme_cbo.get()
    style.theme_use(theme_cbo_value)
    theme_selected.configure(text=theme_cbo_value)
    theme_cbo.selection_clear()


theme_cbo.bind('<<ComboboxSelected>>', change_theme)
theme_selected = ttk.Label(
    master=theme_selection,
    text="litera",
    font="-size 24 -weight bold"
)
theme_selected.pack(side=LEFT)

# 创建主窗口
# root = tk.Tk()
root.title("甲骨文世界初探")
root.state('zoomed')  # 设置窗口为全屏
# root.geometry('650x600')

# 创建一个水平的 PanedWindow
paned_window = tk.PanedWindow(root, orient=tk.HORIZONTAL)
paned_window.pack(fill=tk.BOTH, expand=1)

# 左侧按钮区域
button_frame = tk.Frame(paned_window, width=200, bg='lightgray')
paned_window.add(button_frame)

# 右侧内容区域
content_frame = tk.Frame(paned_window)
paned_window.add(content_frame)

# 切换函数
def switch_frame(frame_func):
    for widget in content_frame.winfo_children():
        widget.destroy()
    frame = frame_func(content_frame)
    frame.pack(fill=tk.BOTH, expand=1)
# 创建按钮
buttons = [
    ("主页", create_frame_1),
    ("形态学图像分割", create_frame_2),
    ("YOLOV8图像分割识别", create_frame_3),
    ("模型的训练与测试", create_frame_4),
]

for (text, frame_func) in buttons:
    button = ttk.Button(button_frame, text=text, command=lambda f=frame_func: switch_frame(f))
    button.pack(fill=tk.X, pady=10)

# 默认显示第一个界面
switch_frame(create_frame_1)

# 运行主循环
root.mainloop()
