import pandas as pd
import torch
from bert_serving.client import BertClient
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

bc = BertClient()


class LinearModel1(nn.Module):
    def __init__(self):
        super(LinearModel1, self).__init__()
        self.fc1 = nn.Linear(768, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class LinearModel2(nn.Module):
    def __init__(self):
        super(LinearModel2, self).__init__()
        self.fc1 = nn.Linear(768, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 9)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class LinearModel3(nn.Module):
    def __init__(self):
        super(LinearModel3, self).__init__()
        self.fc1 = nn.Linear(768, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 25)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class LinearModel4(nn.Module):
    def __init__(self):
        super(LinearModel4, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 87)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class NLP_Data(Dataset):
    def __init__(self, input_feature,label,is_train=True):
        super(NLP_Data, self).__init__()
        self.data = input_feature
        self.label = label
        self.length = (len(self.data))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        feature = torch.from_numpy(self.data[idx][0])
        label = torch.tensor(self.label[idx])
        return feature, label

def get_feature():
    feature_list = []
    df = pd.read_excel("德勤NLP笔试题数据.xlsx", engine='openpyxl')
    for index, row in df.iterrows():
        data = bc.encode([df.iloc[index,0]])
        feature_list.append(data)
    return feature_list

def get_label(type,map):
    df = pd.read_excel("德勤NLP笔试题数据.xlsx", engine='openpyxl')
    type_indice = 'CICS'+ str(type)
    type_list = list(set(df[type_indice]))
    label_list = []
    for i in range(len(type_list)):
        label_list.append(i)
    mapping = dict(zip(type_list, label_list))
    map[type_indice] = mapping
    label_list = []
    for index, row in df.iterrows():
        label = mapping[df.iloc[index, 5-type]]
        label_list.append(label)
    return label_list


def writeout(model, df_result,data_set,map,type):
    type_indice = 'CICS'+ str(type)
    df_tem = []
    result_dataloader = torch.utils.data.DataLoader(data_set, batch_size=16, shuffle=False)
    for counter, data in enumerate(result_dataloader):
        feature, lb = data
        feature = feature.type(torch.FloatTensor)
        lb = lb.type(torch.LongTensor)
        pred = model(feature)
        _, pred = torch.max(pred.data, 1)
        for pred_result in pred:
            for type, label in map[type_indice].items():
                if pred_result == label:
                    df_tem.append(type)
                    break

    df_result[type_indice] = df_tem

def fit(ephochs, lr, data_set, type, opt_func=torch.optim.Adam):
    dataset = data_set
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    results = {}

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16, sampler=test_subsampler)

        if type == 1:
            model = LinearModel1()
        elif type == 2:
            model = LinearModel2()
        elif type ==3 :
            model = LinearModel3()
        else:
            model = LinearModel4()

        optimizer = opt_func(model.parameters(), lr, betas=(0.9, 0.999))
        criterion = nn.CrossEntropyLoss()
        train_loss = []
        accuracy = []
        ##############
        #   train
        #############
        for epoch in range(ephochs):
            l = 0
            for counter, data in enumerate(trainloader):
                feature, lb = data
                feature = feature.type(torch.FloatTensor)
                lb = lb.type(torch.LongTensor)
                pred = model(feature)
                loss = criterion(pred, lb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                l += loss.item()
            train_loss.append(l / (counter + 1))

        ##############
        # validation
        #############
        total = 0
        correct = 0
        for counter, data in enumerate(testloader):
            feature, lb = data
            feature = feature.type(torch.FloatTensor)
            lb = lb.type(torch.LongTensor)
            pred = model(feature)
            _, pred = torch.max(pred.data, 1)
            total += lb.size(0)
            correct += pred.eq(lb.data).sum()
        accuracy.append(100 * correct / total)
        print('acc:%.3f%%' % (100 * correct / total))
        results[fold] = 100.0 * (correct / total)

        plt.figure(1)
        plt.plot(train_loss, '-x')
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs. No. of epochs');
        plt.show()

    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum / len(results.items())} %')
    average_acc = sum / len(results.items())
    return model,average_acc

if __name__ == "__main__":
    map = {}
    acc = []
    df = pd.read_excel("德勤NLP笔试题数据.xlsx", engine='openpyxl')
    df_result = df.iloc[:,0:1]
    input_feature = get_feature()
    for type in range(1,5):
        label = get_label(type,map)
        data_set = NLP_Data(input_feature=input_feature,label=label)
        model,average_acc = fit(100, 0.0003, data_set, type, opt_func=torch.optim.Adam)
        acc.append(average_acc)
        writeout(model, df_result, data_set,map,type)

    print('-------Result-------')
    for i in range(len(acc)):
        index = 'CICS'+str(i+1)+':' + str(acc[i].item()) +'%'
        print(index)

    order = ['业务名称','CICS4','CICS3','CICS2','CICS1']
    df_result = df_result[order]
    df_result.to_excel('Result.xlsx')
