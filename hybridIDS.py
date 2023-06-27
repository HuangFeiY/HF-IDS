import numpy as np 
import pandas as pd 
import time
import joblib 
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score,confusion_matrix
import torch
import torch.utils.data as Data
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import itertools 


start = time.time()

ocsvm_model_path = './model/ocsvm_best_NSLKDD.pkl'
DCVAE_model_path = './minmax_f3_CVAE1_setting4.pkl'
# setting1 = ['normal', 'back', 'ipsweep', 'neptune', 'nmap', 'portsweep', 'satan', 'smurf', 'teardrop']
# setting2
# setting1 = ['normal','ipsweep', 'neptune', 'nmap', 'portsweep','smurf', 'teardrop','guess_passwd','warezclient']
# setting3
# setting1 = ['normal','nmap', 'neptune','satan','pod','back','ipsweep','teardrop']
# setting4 
setting1 = ['normal','ipsweep','portsweep','teardrop','nmap','pod','smurf','warezclient']
value_dicts_path = './Data/value_dict.pkl'
train_data_path = './Data/NSLKDD_train_setting4.npy'
test_data_path = './Data/NSLKDD_test_new.npy'
result_save_path = './pic/ours_setting4.png'
BATCH_SIZE = 50

# =======================================
# 数据导入及预处理
print('data preprocessing ======')
datatrain = np.load(train_data_path)
xtrain = datatrain[:,:-2]  
ytrain = datatrain[:,-2]

test_data = np.load(test_data_path)
print(test_data.shape)

# 加载Label编号含义字典
with open(value_dicts_path, 'rb') as f:
    value_dicts = pickle.load(f)

unknown_label = len(value_dicts[41])  # 使用当前编号字典中的最大编号+1

    # 未知攻击标签处理
selected_samples = []
for sample in test_data:
    label_idx = int(sample[-2])
    if label_idx in value_dicts[41].values():
        label = list(value_dicts[41].keys())[list(value_dicts[41].values()).index(label_idx)]
        if label not in setting1:
            sample[-2] = unknown_label
            selected_samples.append(sample)
        else:
            selected_samples.append(sample)

# 转换为ndarray对象
test_data = np.array(selected_samples)
print(test_data.shape)

X_test = test_data[:, :-2]  # 输入数据，不包括最后一列标签
y_test = test_data[:, -2]   # 标签，最后一列

X_test = MinMaxScaler().fit(xtrain).transform(X_test)
print('data preprocess done')
# =======================================================================

# =======================================================================
# 模型导入，类代码
print('load model ======')
ocsvm_model = joblib.load(ocsvm_model_path)
class CVAE1(nn.Module):
    def __init__(self,num_classes):
        super(CVAE1, self).__init__()
        self.l_z_xy=nn.Sequential(nn.Linear(41+num_classes, 35), nn.Softplus(),nn.Linear(35, 20), nn.Softplus(), nn.Linear(20, 2*3))
        self.l_z_x=nn.Sequential(nn.Linear(41,36),nn.Softplus(),nn.Softplus(), nn.Linear(36,20),nn.Softplus(),nn.Linear(20, 2*3))
        self.l_y_xz=nn.Sequential(nn.Linear(41+3,35),nn.Softplus(), nn.Linear(35,20),nn.Softplus(),nn.Linear(20, num_classes),nn.Sigmoid())     
        self.lb = LabelBinarizer()
    """   
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu   
                        
    def to_categrical(self, y: torch.FloatTensor):
        y_n = y.numpy()
        self.lb.fit(list(range(0,9)))
        y_one_hot = self.lb.transform(y_n)
        floatTensor = torch.FloatTensor(y_one_hot).cuda()
        return floatTensor
    """   
    def z_xy(self,x,y):
        #y_c = self.to_categrical(y)
        xy =  torch.cat((x, y), 1)
        h=self.l_z_xy(xy)
        mu, logsigma = h.chunk(2, dim=-1)
        return D.Normal(mu, logsigma.exp())
        #return mu, logsigma
        
        
    def z_x(self,x):
        mu, logsigma = self.l_z_x(x).chunk(2, dim=-1)
        return D.Normal(mu, logsigma.exp())
        #return mu,logsigma
        
    def y_xz(self,x,z):
        xz=torch.cat((x, z), 1)
        #return D.Bernoulli(self.y_xz(xz))
        return self.l_y_xz(xz)
    
    def forward(self, x):
        mu, logsigma = self.l_z_x(x).chunk(2, dim=-1)
        return self.l_y_xz(torch.cat((x, mu), 1))

def test1(test_loader,device,cuda):
    CVAE = torch.load(DCVAE_model_path)
    CVAE = CVAE.to(device)
    CVAE.eval()  
    total=0
    correct=0
    i=0
    for step, (xtest, ytest) in enumerate(test_loader):
        xtest = Variable(xtest.float())
        if cuda:
            xtest = xtest.cuda()
        ytest = ytest.long()
        #ytest=torch.unsqueeze(ytest, 1)
        #ytest=torch.zeros(ytest.size()[0], 9).scatter_(1, ytest, 1).cuda()
        out = CVAE(xtest)
        _, predicted = torch.max(out.data, 1)
        total += xtest.size(0)
        correct += (predicted.cpu() == ytest).sum()
        if i==0:
                label=np.array(predicted.cpu().data)
        else:
                label = np.array(np.concatenate((label,np.array(predicted.cpu().data)),axis=0))
        i=i+1
    print('Test Accuracy of the model on the XXXX test flows: %4f %%' % (100.0 * correct / total))
    return label

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
 
    # print(cm)
    plt.switch_backend('agg')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    plt.savefig(result_save_path)

def calculate_unknown_detection_rate(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    true_positives_unknown = confusion_matrix[-1, -1]
    total = np.sum(confusion_matrix[-1, :])
    unknown_detection_rate = true_positives_unknown / total
    return unknown_detection_rate


# =========================================================================

# =========================================================================
# 开始混合异常检测
print('start hybrid detection ======')
ocsvm_pred = ocsvm_model.predict(X_test)
y_test_benign_filtered = y_test[ocsvm_pred == 1]
x_test_benign_filtered = X_test[ocsvm_pred == 1]
y_test_anomaly_filtered = y_test[ocsvm_pred == -1]
x_test_anomaly_filtered = X_test[ocsvm_pred == -1]

y_test_benign_filtered=torch.from_numpy(y_test_benign_filtered)
x_test_benign_filtered = torch.from_numpy(x_test_benign_filtered)
y_test_anomaly_filtered=torch.from_numpy(y_test_anomaly_filtered)
x_test_anomaly_filtered = torch.from_numpy(x_test_anomaly_filtered)

test_dataset_benign_filtered = Data.TensorDataset(x_test_benign_filtered, y_test_benign_filtered)
test_loader_benign_filtered = Data.DataLoader(dataset=test_dataset_benign_filtered, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=2)

test_dataset_anomaly_filtered = Data.TensorDataset(x_test_anomaly_filtered, y_test_anomaly_filtered)
test_loader_anomaly_filtered = Data.DataLoader(dataset=test_dataset_anomaly_filtered, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=2)

no_cuda = False
cuda_available = not no_cuda and torch.cuda.is_available()
SEED = 1234
torch.manual_seed(SEED)
device = torch.device("cuda" if cuda_available else "cpu")

predict_benign_filtered=test1(test_loader_benign_filtered,device,cuda_available)
predict_anomaly_filtered=test1(test_loader_anomaly_filtered,device,cuda_available)

predict_anomaly_filtered[predict_anomaly_filtered == 0] = unknown_label

final_predict = np.concatenate((predict_benign_filtered,predict_anomaly_filtered),axis=0)

y_test_benign_filtered = y_test_benign_filtered.numpy()
y_test_anomaly_filtered = y_test_anomaly_filtered.numpy()
final_true = np.concatenate((y_test_benign_filtered,y_test_anomaly_filtered),axis=0)

confusionmatrix=confusion_matrix(final_true,final_predict)
print(confusionmatrix)

target_names = sorted(setting1, key=lambda x: value_dicts[41].get(x, float('inf')))

    # 在列表最后添加"unknown"
target_names.append("unknown")
plot_confusion_matrix(confusionmatrix, classes=target_names, normalize=True, title='')
    # 计算加权精确率
weighted_precision = precision_score(final_true, final_predict, average='weighted')

    # 计算加权召回率
weighted_recall = recall_score(final_true, final_predict, average='weighted')

    # 计算加权F1分数
weighted_f1_score = f1_score(final_true, final_predict, average='weighted')
unknown_detection_rate = calculate_unknown_detection_rate(confusionmatrix)

print("Weighted Precision:", weighted_precision)
print("Weighted Recall:", weighted_recall)
print("Weighted F1 Score:", weighted_f1_score)
print('unknown_detection_rate:',unknown_detection_rate)



end = time.time()
print('elapsed time:',end - start)