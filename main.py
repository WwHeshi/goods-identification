import os
import json
import torch

with open('./txt_data/mapping.json', encoding="utf-8") as f:
    mapping = json.load(f)
    print(mapping)

valid_path = r"./valid_new"
categories = os.listdir(valid_path)
print(categories)

ant=0
for category in categories :
    for key in mapping:
        if mapping[key] == category:
            print(key)
            ant += 1
print(ant)

imgs=[]
labels=[]
with open('txt_data/total.txt', "r", encoding="utf-8") as f:
    lines = f.readlines()
for line in lines:
    print(line)
    # line = line.strip("\n").rstrip().split("\t")
    line = line.split()
    imgs.append(line[0])
    labels.append(int(line[1]))
    print(line)

print(torch.cuda.is_available())