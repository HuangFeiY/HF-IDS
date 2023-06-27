import numpy as np
import os 
import time 
import pandas as pd 
import pickle

start = time.time()

data_dir = '/root/dataset/mydataset/NSL-KDD'
file_train = 'KDDTrain+.txt'
file_test = 'KDDTest+.txt'
file_Path_train = os.path.join(data_dir,file_train)
file_Path_test = os.path.join(data_dir,file_test)


# 读取训练集和测试集文件
train_data = pd.read_csv(file_Path_train, delimiter=',', header=None)
test_data = pd.read_csv(file_Path_test, delimiter=',', header=None)

# 合并训练集和测试集
merged_data = pd.concat([train_data, test_data], axis=0)

# 处理非实数类型的列，对其进行编号
value_dicts = {}  # 用于保存每列的唯一值和对应的编号字典

for column in merged_data.columns:
    if not np.issubdtype(merged_data[column].dtype, np.number):  # 检查列的数据类型是否为非实数类型
        unique_values = merged_data[column].unique()  # 获取列的唯一值
        value_dict = {value: idx for idx, value in enumerate(unique_values)}  # 创建值到编号的字典
        merged_data[column] = merged_data[column].map(value_dict)  # 使用字典进行编号
        value_dicts[column] = value_dict  # 保存字典

# 将数据转换为适当的数据类型
merged_data = merged_data.astype(float)

# 将合并后的数据集分割为训练集和测试集
train_data = merged_data[:len(train_data)]
test_data = merged_data[len(train_data):]

# 输出每列的唯一值和编号字典
for column, value_dict in value_dicts.items():
    print(column)
    print(value_dict)

with open('./Data/value_dict.pkl','wb') as f:
    pickle.dump(value_dicts,f)

# 转换为 NumPy 数组
train_array = train_data.to_numpy()
test_array = test_data.to_numpy()
print(train_array.shape)
print(test_array.shape)

# 保存为 NumPy 数组文件
np.save('./Data/NSLKDD_train.npy', train_array)
np.save('./Data/NSLKDD_test.npy', test_array)



end = time.time()
print('elapsed time:',end - start)
