import numpy as np

# 读取原始数据
data = np.load('./Data/NSLKDD_train.npy')
print(data.shape)

# 筛选倒数第二列为0的样本
filtered_data = data[data[:, -2] == 0]
print(filtered_data.shape)

# 保存筛选后的数据
np.save('./Data/NSLKDD_train_benign.npy', filtered_data[:, :-2])
