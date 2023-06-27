import numpy as np
import pickle

# 读取.npy文件
data = np.load('./Data/NSLKDD_train_new.npy')
print(data.shape)

# 加载Label编号含义字典
with open('./Data/value_dict.pkl', 'rb') as f:
    value_dicts = pickle.load(f)

# 定义需要筛选的Label编号
# target_labels = ['normal', 'back', 'ipsweep', 'neptune', 'nmap', 'portsweep', 'satan', 'smurf', 'teardrop']

# setting2
# target_labels = ['normal','ipsweep', 'neptune', 'nmap', 'portsweep','smurf', 'teardrop','guess_passwd','warezclient']

# setting3
# target_labels = ['normal','nmap', 'neptune','satan','pod','back','ipsweep','teardrop']

# setting4
target_labels = ['normal','ipsweep','portsweep','teardrop','nmap','pod','smurf','warezclient']

# 筛选满足条件的样本
selected_samples = []
for sample in data:
    label_idx = int(sample[-2])
    label = list(value_dicts[41].keys())[list(value_dicts[41].values()).index(label_idx)]
    if label in target_labels:
        selected_samples.append(sample)

# 转换为ndarray对象
selected_samples = np.array(selected_samples)
print(selected_samples.shape)

# 保存为新的.npy文件
np.save('./Data/NSLKDD_train_setting4.npy', selected_samples)
