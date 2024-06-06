import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
from CNN_model import Net
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来设置字体样式以正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False


def bfs(grid, row, col, visited, llt_list, cnt):
    rows = len(grid)
    cols = len(grid[0])
    queue = deque([(row, col)])

    movements = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (1, -1), (-1, 1)]

    while queue:
        curr_row, curr_col = queue.popleft()

        if curr_row < 0 or curr_col < 0 or curr_row >= rows or curr_col >= cols or visited[curr_row][curr_col] or \
                grid[curr_row][curr_col] == 0:
            continue

        visited[curr_row][curr_col] = True

        llt_list[cnt].append(np.array([curr_row, curr_col]))

        for movement in movements:
            next_row = curr_row + movement[0]
            next_col = curr_col + movement[1]
            queue.append((next_row, next_col))

    return len(llt_list[cnt])


def calculate_connected_areas(grid):
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    areas = []
    cnt = 0
    llt_list = {}

    for row in range(rows):
        for col in range(cols):
            if not visited[row][col] and grid[row][col] == 255:
                llt_list[cnt] = []
                area = bfs(grid, row, col, visited, llt_list, cnt)
                cnt += 1
                areas.append(area)

    return areas, llt_list


def plot_image(image, I_bw, I_bfs, image_color):
    fig, axs = plt.subplots(1, 4)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    axs[0].imshow(image)
    axs[0].set_title('原图')
    axs[0].axis('off')

    I_bw = cv2.cvtColor(I_bw, cv2.COLOR_GRAY2BGR)
    axs[1].imshow(I_bw)
    axs[1].set_title('形态学处理后')
    axs[1].axis('off')

    I_bfs = cv2.cvtColor(I_bfs, cv2.COLOR_GRAY2BGR)
    axs[2].imshow(I_bfs)
    axs[2].set_title('区域生长算法去噪后')
    axs[2].axis('off')

    axs[3].imshow(image_color)
    axs[3].set_title('标记后')
    axs[3].axis('off')

    plt.tight_layout()
    plt.show()


# def split_image(image, s_name, r, cnt):
#     x, y, w, h = r
#     I_split = image[y:y + h, x:x + w]
#     cv2.imwrite(f'data/3_Test/Split/{s_name}_{cnt}.jpg', I_split)

def morphological_processing(image):
    # 第一步
    # Convert the image to grayscale
    if len(image.shape) >= 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 第一步
    kernel_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    I_b = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    I_f = cv2.subtract(image, I_b)

    # 第二步
    _, I_bw = cv2.threshold(I_f, 0, 255, cv2.THRESH_OTSU)
    return I_bw



contours = []

def image_labeling(image, rec_pic=False):
    I_bw = image
    # 统计连通图面积 和 对应坐标
    areas, llt_list = calculate_connected_areas(I_bw)
    choose_areas = []
    for a in areas:
        if a > 200:
            choose_areas.append(a)
    avg, s = np.mean(choose_areas), np.std(choose_areas)
    choose_areas = []
    for a in areas:
        if avg - s <= a <= avg + s and a > 200:
            choose_areas.append(a)

    # 根据长宽比进行
    end_areas = []
    for i, area in enumerate(areas):
        if area in choose_areas:
            first_column = [row[0] for row in llt_list[i]]
            second_column = [row[1] for row in llt_list[i]]
            row_max, row_min = max(first_column), min(first_column)
            col_max, col_min = max(second_column), min(second_column)
            p = (row_max - row_min) / (col_max - col_min)
            if 0.1 < p < 10:
                end_areas.append(area)

    I_bfs = I_bw.copy()
    for i, area in enumerate(areas):
        if area not in end_areas:
            for l in llt_list[i]:
                row, col = l[0], l[1]
                I_bfs[row][col] = 0

    if rec_pic:
        # 框出来剩余连通图的面积
        contours.clear()
        for i, area in enumerate(areas):
            if area in end_areas:
                first_column = [row[0] for row in llt_list[i]]
                second_column = [row[1] for row in llt_list[i]]
                row_max, row_min = max(first_column), min(first_column)
                col_max, col_min = max(second_column), min(second_column)
                x, y, w, h = row_min, col_min, row_max - row_min, col_max - col_min
                contours.append((y, x, h, w))  # cv2

        image_color = cv2.cvtColor(I_bfs, cv2.COLOR_GRAY2BGR)
        for i, contour in enumerate(contours):
            # split_image(image, I_name[:-4], contour, i)
            x, y, w, h = contour
            cv2.rectangle(image_color, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(image_color, str(i + 1), (x-1, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220,20,60), 1)

        return image_color
    else:
        return I_bfs


# 生成原始图片
def ret_yuanshi(I_bfs):
    if len(I_bfs.shape) >= 3 and I_bfs.shape[2] == 3:
        I_bfs = cv2.cvtColor(I_bfs, cv2.COLOR_BGR2GRAY)
    image_color = cv2.cvtColor(I_bfs, cv2.COLOR_GRAY2BGR)
    for i, contour in enumerate(contours):
        x, y, w, h = contour
        print(x, y, w, h)
        cv2.rectangle(image_color, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(image_color, str(i + 1), (x - 1, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 20, 60), 1)

    return image_color, contours


CNN_model = Net()

CNN_model.load_state_dict(torch.load('result/CNN_model_29.pth'), strict=False)

CNN_model.eval()

def Pre_predict(input_image):
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 将图像转换为灰度图像
        transforms.ToTensor(),  # 将PIL图像转换为张量
        transforms.Resize((64, 64)),  # 尺寸调整
        # 可以添加其他转换，如归一化等
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # 创建一个batch
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CNN_model.to(device)
    input_batch = input_batch.to(device)

    # 进行预测
    with torch.no_grad():
        output = CNN_model(input_batch)
        _, predicted = torch.max(output, 1)
    predicted_class_index = predicted.item()
    return predicted_class_index
