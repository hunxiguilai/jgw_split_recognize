import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os

# 定义图像大小和设备
img_size = (224, 224)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

new_mapping_dict={'万': 0,'丘': 1,'丙': 2,'丧': 3,'乘': 4,'亦': 5,'人': 6,'今': 7,'介': 8,'从': 9,'令': 10,'以': 11,'伊': 12,'何': 13,'余': 14,'允': 15,'元': 16,'兄': 17,'光': 18,'兔': 19,
 '入': 20,'凤': 21,'化': 22,'北': 23,'印': 24,'及': 25,'取': 26,'口': 27,'吉': 28,'囚': 29,'夫': 30,'央': 31,'宗': 32,'宾': 33,'尞': 34,'巳': 35,'帽': 36,'并': 37,'彘': 38,'往': 39,
 '御': 40,'微': 41,'旨': 42,'昃': 43,'木': 44,'朿': 45,'涎': 46,'灾': 47,'焦': 48,'爽': 49,'牝': 50,'牡': 51,'牧': 52,'生': 53,'田': 54,'疑': 55,'祝': 56,'福': 57,'立': 58,
 '羊': 59,'羌': 60,'翌': 61,'翼': 62,'老': 63,'艰': 64,'艺': 65,'若': 66,'莫': 67,'获': 68,'衣': 69,'逆': 70,'门': 71,'降': 72,'陟': 73,'雍': 74,'鹿': 75}

# 预处理输入图像
def preprocess_image(image_path):
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # 创建一个批次
    return input_batch

# 加载模型
def load_model(model_path):
    # 先创建一个空的模型对象
    model = None
    if 'mobilenet_v3_model' in model_path:
        model = models.mobilenet_v3_small(pretrained=False, num_classes=76)
    elif 'resnet18_model' in model_path:
        model = models.resnet18(pretrained=False, num_classes=76)
    # 添加更多的模型架构判断
    elif 'vgg16_model' in model_path:
        model = models.vgg16(pretrained=False, num_classes=76)
    # 添加更多的模型架构判断
    elif 'AlexNet_model' in model_path:
        model = models.alexnet(pretrained=False, num_classes=76)
    
    if model is not None:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        model.eval()
        model.to(device)
    else:
        raise ValueError('Unsupported model architecture.')
    
    return model


# 执行预测
def predict_image(model, image_path):
    input_batch = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(input_batch)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# 创建图形界面
class App:
    def __init__(self, root):
        self.root = root
        self.root.title('图像分类应用程序')
        # 模型选择下拉框
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(root, textvariable=self.model_var)
        self.model_dropdown['values'] = self.get_model_list()
        self.model_dropdown.grid(row=0, column=0, padx=10, pady=10)
        
        # 上传按钮
        self.upload_button = tk.Button(root, text='上传图片', command=self.upload_image)
        self.upload_button.grid(row=0, column=1, padx=10, pady=10)
        
        # 显示图像的标签
        self.image_label = tk.Label(root)
        self.image_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        
        # 显示预测结果的标签
        self.result_label = tk.Label(root, text='预测结果: ')
        self.result_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
    
    def get_model_list(self):
        model_dir = 'C:\\Users\\92756\\Desktop\\jiaguwen'  # 包含模型文件的目录
        return [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    def upload_image(self):
        image_path = filedialog.askopenfilename(filetypes=[('图像文件', '*jpg *jpeg *png')])
        # print(image_path)
        if image_path:
            # 显示图像
            image = Image.open(image_path)
            image.thumbnail((500, 500))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # 保持对图像的引用
            
            # 执行预测
            model_name = self.model_var.get()
            if model_name:
                model_path = os.path.join('C:\\Users\\92756\\Desktop\\jiaguwen', model_name)
                try:
                    model = load_model(model_path)
                    predicted_class = predict_image(model, image_path)
                    for key, value in new_mapping_dict.items():
                        if value == predicted_class:
                            self.result_label.config(text=f'预测结果: {key}')
                            break
                    else:
                        self.result_label.config(text=f'Value {predicted_class} not found in the dictionary.')
                except ValueError as e:
                    self.result_label.config(text=str(e))
            else:
                self.result_label.config(text='请选择一个模型。')


# 运行应用程序
if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
