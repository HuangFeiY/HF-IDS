import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
import time 
import joblib
from sklearn.metrics import roc_auc_score

start = time.time()

# 加载OCSVM模型
ocsvm_model_path = "./model/ocsvm_best_NSLKDD.pkl"


# 加载测试数据集
test_data = np.load('./Data/NSLKDD_test.npy')

# 获取label列
label_column = test_data[:, -2]

# 将非零值的索引找出
nonzero_indices = label_column != 0

# 将非零值的位置改为 1
test_data[nonzero_indices, -2] = 1

# 获取输入数据和标签
X_test = test_data[:, :-2]  # 输入数据，不包括最后一列标签
y_test = test_data[:, -2]   # 标签，最后一列

datatrain = np.load('./Data/NSLKDD_train_setting1.npy')
xtrain = datatrain[:,:-2]  
ytrain = datatrain[:,-2]


# scaler = MinMaxScaler()
X_test = MinMaxScaler().fit(xtrain).transform(X_test)
# X_test = scaler.fit_transform(X_test)


# 加载模型
ocsvm_model = joblib.load(ocsvm_model_path)

y_test_modified = np.where(y_test == 0, 1, -1)
print(y_test)
print(y_test_modified)

# 使用OCSVM模型进行异常检测
y_pred = ocsvm_model.predict(X_test)
print(y_pred==1)

num_same_values = np.sum(y_test_modified == y_pred)
print(num_same_values)
print(len(y_test))
print(num_same_values/len(y_test))
# np.save('./y_test.npy',y_test_modified)
# np.save('./y_pred.npy',y_pred)




# 计算混淆矩阵
confusion = confusion_matrix(y_test_modified, y_pred)

# 提取混淆矩阵的各项指标
tn, fp, fn, tp = confusion.ravel()
detection_rate = tp / (tp + fn)
false_positive_rate = fp / (fp + tn)

# 打印混淆矩阵和指标
print("Confusion Matrix:")
print(confusion)
print("Detection Rate:", detection_rate)
print("False Positive Rate:", false_positive_rate)

auc = roc_auc_score(y_test_modified, y_pred)
print("AUC:", auc)

end = time.time()
print('elapsed time:',end - start)
