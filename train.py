import os
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
import matplotlib.pyplot as plt


class MyDataset(Dataset):  # 定义数据集类 继承Dataset
    def __init__(self, txt_path, transform=None):  # 创建ToTensor对象
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.imgs = []
        self.labels = []
        for line in lines:
            line = line.strip("\n").rstrip().split("\t")  # 删除头尾/n 删除末尾空格 以/t分隔
            # line = line.split()
            self.imgs.append(line[0])
            self.labels.append(int(line[1]))
        self.transform = transform

    def __getitem__(self, mapping):
        img_path = self.imgs[mapping]
        label = self.labels[mapping]
        img = cv2.resize(cv2.imread(img_path), (96, 192))
        if self.transform is not None:
            img = self.transform(img)  # 将图像转换为tensor格式 归一化   将 HWC 的图像格式转为 CHW 的 tensor 格式
        return img, label

    def __len__(self):
        return len(self.imgs)


ResNet = models.resnet18(num_classes=100)

epochs = 100
batch_size = 32
# 较小的batch size可能会导致模型收敛速度慢，而太大的batch size可能会导致模型在训练过程中发散或者过拟合
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not os.path.exists("output"):
    os.makedirs("output")

# 数据预处理和加载
train_dataset = MyDataset("txt_data/train.txt", transform=transforms.ToTensor())
val_dataset = MyDataset("txt_data/val.txt", transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# shuffle参数控制是否在每个epoch开始时对数据进行随机排序。在训练过程中，通常推荐将其设置为True以提高模型的泛化能力

optimizer = torch.optim.AdamW(ResNet.parameters(), lr=0.001, weight_decay=1e-3)
# lr学习率决定了模型参数更新的步长大小1e-6至1.0   weight_decay权重衰减是一种正则化技术，用于防止模型过拟合 通常取1e-3  1e-2至1e-4
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 80], 0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 80], 0.1)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80, 150, 250], 0.2)
# 学习率调度器，用于在训练过程中调整学习率。     在每个milestone时，将此前学习率乘以gamma。
loss_func = nn.CrossEntropyLoss()
# 交叉熵损失函数   主要用于多分类项目中运用    nn: 深度集成神经网络库

ResNet.to(device)

best_acc = 0
min_loss = 100
loss_list = []
acc_list = []

for epoch in range(epochs):
    # train     训练模式
    ResNet.train()
    for i, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)
        output = ResNet(img)                # 将输入图像通过模型得到输出。    输出通常是一个概率分布  数据归一化后每个类别的概率之和为1
        loss = loss_func(output, label)     # 计算输出和真实标签之间的损失。   为标量
        optimizer.zero_grad()               # 清除之前的梯度信息，为新的梯度计算做准备。
        loss.backward()                     # 根据损失函数反向传播梯度。
        optimizer.step()                    # 根据梯度更新模型的权重。
    scheduler.step()                        # 根据学习率调度器更新学习率。

    # val       评估模式
    ResNet.eval()
    with torch.no_grad():                   # 更进一步加速和节省gpu空间（因为不用计算和存储梯度）
        total = 0
        correct = 0
        for i, (img, label) in enumerate(val_loader):
            img = img.to(device)
            label = label.to(device)
            output = ResNet(img)            # 输出的大小是 [batch_size, num_classes]
            _, predicted = torch.max(output.data, 1)  # 最大值被赋值给变量_  最大值的索引predicted
            total += label.size(0)          # 将label张量的第一个维度的大小（即元素数量）加到total变量上。
            correct += (predicted == label).sum().item()    # item()函数取出的元素值的精度更高，所以在求损失函数等时一般用item()
        print(f"Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {correct / total}")  # 打印每个epoch的损失和准确率。
        loss_list.append(loss.item())       # 将损失和准确率添加到列表中，用于后续分析。
        acc_list.append(correct / total)
        # if correct / total >= best_acc and loss.item() <= min_loss:
        #     best_acc = correct / total
        #     min_loss = loss.item()
        if correct / total >= best_acc:
            best_acc = correct / total
        torch.save(ResNet.state_dict(), f"output/{epoch}model_{correct / total:.4f}.pt")
        # 保存当前模型的权重。文件名中包含了epoch和准确率信息，方便后续查找最佳模型。

print(f"best_acc: {best_acc}, min_loss: {min_loss}")


fig, ax1 = plt.subplots()
# fig: 这是创建的整个图形的引用。你可以通过fig来访问和修改图形的各种属性，比如标题、布局等。
# ax1: 这是创建的第一个坐标轴的引用。坐标轴是图形中的一个子图，用于绘制数据。
# 你可以通过ax1来访问和修改坐标轴的各种属性，比如轴标签、刻度、图例等。

# ax1.set_title('My Plot')
ax1.plot(loss_list, label="loss", color='#ff8213')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.tick_params(axis='y', labelcolor='#ff8213')

ax2 = ax1.twinx()
ax2.plot(acc_list, label="acc", color='#1f77b4')
ax2.set_ylabel('Accuracy')
ax2.tick_params(axis='y', labelcolor='#1f77b4')

fig.tight_layout()
plt.show()



'''
由于没有设计很复杂的baseline，目前基于baseline训练出来的模型acc最高仅有0.71（验证集）。
想要提高模型的准确度，不仅可以增加模型的层数，加入各式各样的数据增强，还可以对原数据集进一步预处理，比如标准化亮度、清洗脏数据等等。

提高模型的准确度:
    增加模型的层数
    数据增强
    数据预处理
    更换更强大的模型
    ......

从根源上解决数据集存在的问题，着手问题优化模型
目标：在测试集（test）上达到0.90+的水平
'''
