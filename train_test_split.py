from sklearn.model_selection import train_test_split

with open("txt_data/total.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
# print(lines)
train, val = train_test_split(lines, test_size=0.2, random_state=42)

with open("txt_data/train.txt", "w", encoding="utf-8") as f_train:
    f_train.writelines(train)
with open("txt_data/val.txt", "w", encoding="utf-8") as f_val:
    f_val.writelines(val)
