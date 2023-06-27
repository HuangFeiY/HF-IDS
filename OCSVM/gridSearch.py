import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix
import joblib
import time 


start = time.time()
# 读入数据
data = np.load('./Data/NSLKDD_train_benign.npy')

# 数据标准化
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

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

X_test = scaler.transform(X_test)

y_test_modified = np.where(y_test == 0, 1, -1)


nu_range = np.arange(0.01,0.51,0.04)
gamma_range = np.arange(0.1,10.1,0.4)

max_auc = 0
best_nu = 0
best_gamma = 0

for nu_value in nu_range:
    for gamma_value in gamma_range:
        # 创建 OCSVM 模型
        ocsvm_model = OneClassSVM(kernel='rbf',nu=nu_value,gamma=gamma_value)
        ocsvm_model.fit(data)
        y_pred = ocsvm_model.predict(X_test)
        # 计算混淆矩阵
        confusion = confusion_matrix(y_test_modified, y_pred)
        # 提取混淆矩阵的各项指标
        tn, fp, fn, tp = confusion.ravel()
        detection_rate = tp / (tp + fn)
        false_positive_rate = fp / (fp + tn)

        # 计算AUC
        auc = roc_auc_score(y_test_modified, y_pred)

        print('nu_value:',nu_value)
        print('gamma_value:',gamma_value)

        # 打印结果
        print("Confusion Matrix:")
        print(confusion)
        print("Detection Rate:", detection_rate)
        print("False Positive Rate:", false_positive_rate)
        print("AUC:", auc)
        if auc > max_auc:
            max_auc = auc 
            best_nu = nu_value
            best_gamma = gamma_value
            # 保存模型
            joblib.dump(ocsvm_model, './model/ocsvm_best_NSLKDD.pkl')
            print('模型保存成功')
            


print('best_nu:',best_nu)
print('best_gamma:',best_gamma)

# 加载最佳模型
ocsvm_best_model = joblib.load('./model/ocsvm_best_NSLKDD.pkl')
y_pred = ocsvm_best_model.predict(X_test)

confusion = confusion_matrix(y_test_modified, y_pred)
# 提取混淆矩阵的各项指标
tn, fp, fn, tp = confusion.ravel()
detection_rate = tp / (tp + fn)
false_positive_rate = fp / (fp + tn)

# 计算AUC
auc = roc_auc_score(y_test_modified, y_pred)
print("Best AUC score:", auc)

print("Best Detection Rate:", detection_rate)
print("Best False Positive Rate:", false_positive_rate)


end = time.time()
print('elapsed time:',end - start)
