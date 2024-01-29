import os
import shutil
from tqdm import tqdm
import json

data_path = r"./data_before"
new_data_path = r"./data"
categories = os.listdir(data_path)
mapping = {}

if not os.path.exists(new_data_path):
    os.mkdir(new_data_path)
if not os.path.exists("txt_data"):
    os.mkdir("txt_data")

with open('txt_data/total.txt', 'w', encoding="utf-8") as total_txt:
    for i, category in enumerate(tqdm(categories)):
        if not os.path.exists(os.path.join(new_data_path, str(i))):
            os.mkdir(os.path.join(new_data_path, str(i)))
        mapping[i] = category
        category_path = os.path.join(data_path, category)
        new_category_path = os.path.join(new_data_path, str(i))
        files = os.listdir(category_path)
        for j, file in enumerate(tqdm(files, leave=False)):
            file_path = os.path.join(category_path, file)
            new_file_path = os.path.join(new_category_path, str(i) + '_' + str(j) + '.jpg')
            shutil.copyfile(file_path, new_file_path)
            total_txt.write(new_file_path + '\t' + str(i) + '\n')

with open('txt_data/mapping.json', 'w', encoding="utf-8") as f:
    json.dump(mapping, f)   # 将python中的对象转化成json储存到文件中

'''
程序作用：
创建新的数据集data，从0开始排序
在txt_data下创建total.txt与mapping.json
total.txt格式：
相对路径＋种类序号
mapping.json格式（字典）：
种类序号：种类中文名称
'''
