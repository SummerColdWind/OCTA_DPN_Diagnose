# 该代码用于划分数据集，将数据集按照比例划分为训练集和测试集，在训练集和测试集中分别有对应的类别文件夹如0和1。
# 运行该代码前，请确保将数据集整体复制到data文件夹中，并将对应的 CSV 文件也放入其中，CSV 文件中包含每个文件夹的类别标签。
# 运行该代码后，会在data文件夹下创建 train 和 val 两个文件夹，并按照比例分配每个类别的图像文件到对应的文件夹中。

# 导入必要的库
import os
import shutil
import random
import pandas as pd

# 指定数据集路径和CSV文件路径
dataset_path = 'data'
csv_file_path = os.path.join(dataset_path, 'label.csv')
# 读取CSV文件
df = pd.read_csv(csv_file_path)
# 获取类别的数量和类别名称
classes = df['label'].unique()
print('类别数量:', len(classes))
print('类别名称:', classes)

valid_frac = 0.1  # 测试集比例
# 按类别划分和移动文件
for cls in classes:
    train_path = os.path.join(dataset_path, 'train', str(cls))
    valid_path = os.path.join(dataset_path, 'val', str(cls))
    # 创建训练集文件夹
    if os.path.exists(train_path):
        print(f"训练集文件夹 {train_path} 已存在!")
    else:
        os.makedirs(train_path)
        print(f"已创建训练集文件夹: {train_path}")
    # 创建验证集文件夹
    if os.path.exists(valid_path):
        print(f"验证集文件夹 {valid_path} 已存在!")
    else:
        os.makedirs(valid_path)
        print(f"已创建验证集文件夹: {valid_path}")
    # 从DataFrame中获取该类别的所有文件名
    class_files = df[df['label'] == cls]
    # print(class_files)
    # print(f"类别 {cls} 的文件数量为 {len(class_files)}")
    # 随机打乱文件顺序
    class_files = class_files.sample(frac=1, random_state=123).reset_index(drop=True)
    # print(class_files)
    # print(f"类别 {cls} 的文件顺序为 {class_files['name']}")
    # print(f"类别 {cls} 的文件数量为 {len(class_files)}")

    # # 计算测试集数量
    test_set_numer = int(len(class_files) * valid_frac)
    # print(f"类别 {cls} 的测试集数量为 {test_set_numer}")
    # 测试集和训练集文件名
    test_set_files = class_files.iloc[:test_set_numer]
    # print(f"类别 {cls} 的测试集文件名为 {test_set_files['name']}")
    train_set_files = class_files.iloc[test_set_numer:]
    # print(f"类别 {cls} 的训练集文件名为 {train_set_files['name']}")

    # 移动测试集文件到对应的验证集文件夹
    for _, row in test_set_files.iterrows():
        old_img_path = os.path.join(dataset_path, row['name'])
        new_valid_path = os.path.join(valid_path, os.path.basename(row['name']))
        # print(f"测试集文件 {old_img_path} 移动至 {new_valid_path}")
        shutil.move(old_img_path, new_valid_path)
    #
    # 移动训练集文件到对应的训练集文件夹
    for _, row in train_set_files.iterrows():
        old_img_path = os.path.join(dataset_path, row['name'])
        new_train_path = os.path.join(train_path, os.path.basename(row['name']))
        # print(f"测试集文件 {old_img_path} 移动至 {new_train_path}")
        shutil.move(old_img_path, new_train_path)
print("文件移动完成！数据集准备完成！")
