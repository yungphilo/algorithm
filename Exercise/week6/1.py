import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np

# 配置设置
class Config:
    image_dir = "images"  # 图片目录
    batch_size = 4              # 批处理大小
    num_epochs = 15             # 训练轮数
    lr = 0.001                  # 学习率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备
    seed = 42                   # 随机种子

# 确保结果可复现
torch.manual_seed(Config.seed)
np.random.seed(Config.seed)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),          # 调整大小
    transforms.CenterCrop(224),      # 中心裁剪
    transforms.ToTensor(),           # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 增强的数据集类 - 处理无效文件和隐藏文件
class RobustDogCatDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 收集有效图像文件
        for img_name in os.listdir(image_dir):
            # 跳过隐藏文件（以 ._ 开头的文件）
            if img_name.startswith('._'):
                continue
                
            img_path = os.path.join(image_dir, img_name)
            
            # 检查是否为有效图像文件
            if not os.path.isfile(img_path):
                continue
                
            # 检查文件扩展名
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif','webp')):
                continue
                
            # 尝试打开文件以验证是否为有效图像
            try:
                with Image.open(img_path) as img:
                    img.verify()  # 验证文件完整性
                    
                # 根据文件名判断类别（狗为目标类别1，猫为0）
                if 'dog' in img_name.lower():
                    self.labels.append(1)  # 狗为目标类别
                    self.image_paths.append(img_path)
                elif 'cat' in img_name.lower():
                    self.labels.append(0)  # 猫为非目标类别
                    self.image_paths.append(img_path)
                    
            except (IOError, OSError, Image.UnidentifiedImageError):
                # 跳过无效图像
                continue
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, torch.tensor(label, dtype=torch.long)
            
        except Exception:
            # 返回一个空图像作为回退
            dummy_image = torch.zeros(3, 224, 224)  # 创建一个黑色图像
            return dummy_image, torch.tensor(label, dtype=torch.long)

# 创建数据集和数据加载器
dataset = RobustDogCatDataset(Config.image_dir, transform=transform)
print(f"找到 {len(dataset)} 张有效图像")
print(f"- 狗: {sum(dataset.labels)} 张")
print(f"- 猫: {len(dataset.labels) - sum(dataset.labels)} 张")

# 确保有足够的图像
if len(dataset) == 0:
    raise ValueError("未找到有效图像！请检查目录路径和文件格式")

dataloader = DataLoader(
    dataset, 
    batch_size=Config.batch_size, 
    shuffle=True,
    num_workers=0  # 为避免多进程问题，设为0
)

# 加载预训练模型
def create_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # 冻结所有主干网络参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 替换最后一层（狗为目标类别）
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),  # 特征维度
        nn.ReLU(),                     # 激活函数
        nn.Dropout(0.3),               # 防止过拟合
        nn.Linear(256, 2)               # 二分类输出
    )
    return model.to(Config.device)

# 初始化模型、损失函数和优化器
model = create_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=Config.lr)

# 训练循环
def train_model():
    model.train()
    
    for epoch in range(Config.num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(Config.device)
            labels = labels.to(Config.device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计信息
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        print(f'Epoch [{epoch+1}/{Config.num_epochs}] | '
              f'Loss: {epoch_loss:.4f} | '
              f'Accuracy: {epoch_acc:.2f}%')
    
    print("训练完成!")
    return model

# 执行训练
print("\n开始训练（狗为目标类别）...")
trained_model = train_model()

# 保存模型
torch.save({
    'model_state_dict': trained_model.state_dict(),
    'class_mapping': {0: '猫', 1: '狗'}  # 保存类别映射
}, 'dog_classifier.pth')

print("模型已保存为 'dog_classifier.pth'")

# 预测函数（返回类别和置信度）
def predict(image_path):
    model.eval()
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(Config.device)
        
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, pred = output.max(1)
            confidence = probabilities[0, pred.item()].item()
            class_label = "狗" if pred.item() == 1 else "猫"
            return class_label, confidence
            
    except Exception as e:
        print(f"预测时出错: {str(e)}")
        return "错误", 0.0

# 测试模型
print("\n测试模型:")
for img_path in dataset.image_paths[:5]:  # 测试前5个图像
    prediction, confidence = predict(img_path)
    filename = os.path.basename(img_path)
    print(f"{filename}: {prediction} (置信度: {confidence:.2%})")

def predict(image_path):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(Config.device)
    
    with torch.no_grad():
        output = model(image_tensor)
        _, pred = output.max(1)
        return "狗" if pred.item() == 1 else "猫"

# 示例预测
print(predict("images/test-caigou.jpg"))  