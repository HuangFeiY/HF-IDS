import numpy as np

# 加载原始数据
train_data = np.load('./Data/NSLKDD_train.npy')
test_data = np.load('./Data/NSLKDD_test.npy')

# 合并数据
merged_data = np.concatenate((train_data, test_data), axis=0)

# 获取 Label 列
labels = merged_data[:, -2]

# 获取 Label 值的唯一集合
unique_labels = np.unique(labels)

# 计算测试集中的样本数量
test_ratio = 0.15
# test_samples = int(len(merged_data) * test_ratio)

# # 初始化训练集和测试集列表
# train_samples = len(merged_data) - test_samples
train_data_new = []
test_data_new = []

# 遍历每个 Label 值
for label in unique_labels:
    # 选择当前 Label 值的样本
    label_data = merged_data[labels == label]

    # 计算当前 Label 值在测试集中的样本数量
    label_test_samples = int(len(label_data) * test_ratio)

    # 将样本分配给训练集和测试集
    train_data_new.append(label_data[label_test_samples:])
    test_data_new.append(label_data[:label_test_samples])

# 合并并保存训练集和测试集
train_data_new = np.concatenate(train_data_new, axis=0)
test_data_new = np.concatenate(test_data_new, axis=0)

print(train_data_new.shape)
print(test_data_new.shape)

# 保存新的训练集和测试集
np.save('./Data/NSLKDD_train_new.npy', train_data_new)
np.save('./Data/NSLKDD_test_new.npy', test_data_new)
