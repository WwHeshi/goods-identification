# 基于pytorch与renet模型实现小商品检测
数据为100种饮料  
以下是流程介绍  

## Preprocessing.py
预处理程序  
创建新的训练用数据集data，从0开始排序  
在txt_data下创建total.txt与mapping.json  
total.txt格式：  
相对路径＋种类序号  
mapping.json格式（字典）：  
种类序号：种类中文名称  

## train_test_split.py
划分训练集程序  
使用sklearn.model_selection中的分离器函数train_test_split()  
实验采用2 8 分  

## train.py
训练代码，包括Dataset类方法  
采用resnet18  
包括eval与制图代码  

