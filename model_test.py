import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt
import pdb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import tree
import random
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader, ConcatDataset
import torch
from tqdm import tqdm




class model_DT(object):
    """docstring for model_DT"""
    def __init__(self, config):
        super(model_DT, self).__init__()
        self.config  = config
    def load_data(self):        
        train_data   = np.load('train_data.npy')
        train_labels = np.load('train_label.npy')
        return train_data,train_labels,train_data,train_labels

    def main(self):
        X,y,train_data,train_labels = self.load_data()
        num_train = self.config.num_train
        model_dt     = tree.DecisionTreeClassifier()

        model        = model_dt.fit(X[:num_train,:], y[:num_train])  

        val          = model.predict(train_data[:num_train,:])
        train_pre    = model.predict(X[:num_train,:])

        acc_train    = np.sum(train_pre[:num_train] == y[:num_train])/len(y[:num_train])
        acc          = np.sum(val[:num_train] == train_labels[:num_train])/len(train_labels[:num_train])

        print('Evaluation_DT:')
        print('Accuracy_val:',acc)
        print('Accuracy_train:',acc_train)
        end_epoch = len(X)
        re = []
        for i in tqdm(range(end_epoch)):
            tmp = model_dt.predict_proba(X[i].reshape(1,4),check_input=True)[0][1]

            re.append(tmp)
        print('Computation is done !')
        re = np.array(re)
        np.save('Results_DT',re)

class model_SVM(object):
    """docstring for model_DT"""
    def __init__(self, config):
        super(model_SVM, self).__init__()
        self.config  = config
    def load_data(self):        
        train_data   = np.load('train_data.npy')
        train_labels = np.load('train_label.npy')
        return train_data,train_labels,train_data,train_labels

    def main(self):
        X,y,train_data,train_labels = self.load_data()
        # print(X.shape)
        # exit()
        num_train = self.config.num_train
        model_svm    = svm.SVC(kernel='linear',gamma=0.8)
        model        = model_svm.fit(X[:num_train,:], y[:num_train])  
        ## get parameters from trained model ##
        w = model_svm.coef_[0]
        b = model_svm.intercept_[0]
        val          = model.predict(train_data)
        train_pre    = model.predict(X[:num_train,:])
        acc_train    = np.sum(train_pre == y[:num_train])/len(y[:num_train])
        acc          = np.sum(val == train_labels)/len(train_labels)

        print('Evaluation_SVM:')
        print('Accuracy_val:',acc)
        print('Accuracy_train:',acc_train)
        start_epoch = num_train
        end_epoch = len(X)
        ## compute distance ## 
        re = []
        for i in tqdm(range(end_epoch)):
            tmp = np.sum(w * X[i]) + b
            re.append(tmp)
        print('Computation is done !')
        re = np.array(re)
        np.save('Results_SVM',re)
        

class mydataset(torch.utils.data.Dataset):
    def __init__(self,x,y = None):
        self.x       = x
        self.y       = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        if self.y is None:
            return self.x[index]
        return self.x[index],self.y[index]


class model_ANN(object):
    """docstring for model_DT"""
    def __init__(self, config,device):
        super(model_ANN, self).__init__()
        self.config   = config
        self.epoch    = self.config.num_epochs
        self.device   = device

    def load_data(self):        
        train_data    = np.load('train_data.npy')
        train_labels  = np.load('train_label.npy')
        return train_data,train_labels,train_data,train_labels



    def generateANN(self):
        net           =      nn.Sequential(
                             nn.Linear(self.config.input_size,self.config.num_class),nn.Softmax())
        net.apply(self.init_weights)
        return net

    def generateMLP(self):
        net           =    nn.Sequential(
                             nn.Linear(self.config.input_size,self.config.hidden_layer*2),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(self.config.hidden_layer*2,self.config.hidden_layer),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(self.config.hidden_layer,self.config.num_class),
                             nn.ReLU(),
                             nn.Softmax())
        net.apply(self.init_weights)
        return net

    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)


    ####  build the train data loader ####
    def main(self):
        if self.config.model_name == 'ANN':
            net      = self.generateANN()
        else:
            net      = self.generateMLP()
        self.X,self.y,self.train_data,self.train_labels = self.load_data()
        train_set    = mydataset(self.X,self.y)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config.batch_size,
                                                     shuffle=True,drop_last = True)
        ####  build the train data loader ####
        train_set      = mydataset(self.train_data,self.train_labels)
        train_loader   = torch.utils.data.DataLoader(train_set, batch_size=self.config.batch_size,
                                                     shuffle=False,drop_last=True)

        net.train()

        criterion    = nn.CrossEntropyLoss()

        optimizer    = torch.optim.SGD(net.parameters(),lr = self.config.init_lr,
                                                       momentum=self.config.momentum,
                                                       weight_decay=self.config.weight_decay)

        if self.config.load:
            print("Restore from old params")
            net.load_state_dict(torch.load('oldParams.t7'))
        else:
            print("Start from default params")



        #### def the train function ######
            num     = 0
            cnt     = 0
            for i in range(self.epoch):
                print('epoch:',(i+1))
                cnt        += 1
                total_train = 0
                train_acc   = 0
                

                tra_loss    = []
                
                for batch_num, (train_data, train_label) in enumerate(train_loader):

                    optimizer.zero_grad()
                    train_data       = train_data.to(self.device).float()
                    train_label      = train_label.to(self.device).float()
                    train_output     = net(train_data)
                    train_loss       = criterion(train_output,train_label.long())
                    train_loss.backward()
                    optimizer.step()

                    train_prediction = train_output.cpu().detach().argmax(dim=1)
                    train_distribution = train_output.cpu().detach()


                    ## get the max prob ## 


                    train_acc       += torch.sum(torch.eq(train_prediction, train_label.long())).item()
                    total_train     += len(train_label)

                    tra_loss.extend([train_loss.item()])
                    mean_train_loss  = np.mean(tra_loss)
                    tra_accuracy     = train_acc / total_train
                    train_accuracy   = (train_prediction.numpy()==train_label.cpu().numpy()).mean()
                    if (num+1) % 500 == 0:
                        print('num:', (num+1))
                        print("Train loss :",mean_train_loss)
                        print("Train accuracy :",tra_accuracy)
                    num += 1
            # torch.save(net.state_dict(),'oldParams.t7')
        
        ############### for test -> prob score ##################
        re = []
        for batch_num, (train_data, train_label) in enumerate(train_loader):
            net.eval()
            optimizer.zero_grad()
            train_acc    = 0
            valid_loss = []
            total_val  = 0
            train_data   = train_data.to(self.device).float()
            train_label  = train_label.to(self.device).float()
            train_output = net(train_data)
            train_loss   = criterion(train_output, train_label.long())


            train_prediction = train_output.cpu().detach().argmax(dim=1)

            index = train_output.cpu().detach().argmax(dim=1)
            train_distribution = train_output.cpu().detach()

            ## get the max prob ## 
            index = train_prediction
            for num in range(self.config.batch_size):
                tmp = train_distribution[num,index[num]]
                re.append(tmp)
            #     print(train_distribution[num,index[num]])
            # exit()
            train_acc       += torch.sum(torch.eq(train_prediction, train_label.long())).item()
            total_val     += len(train_label)
            valid_loss.extend([train_loss.item()])
            mean_train_loss = np.mean(valid_loss)
            train_accuracy  = train_acc / total_val

            train_accuracy = (train_prediction.numpy() == train_label.cpu().numpy()).mean()
            # net.train()
        re = np.array(re)
        print('Evaluation_ANN:')
        print("Val loss :", mean_train_loss)
        print("Val accuracy :", train_accuracy)
        np.save('score_results',re)
        print('Computation is done !')


    
    

if __name__ == '__main__':
    DecisionTree_model = model_DT(8)
    DecisionTree_model.main()


    SVM_model          = model_SVM(8)
    SVM_model.main()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch = 5
    ANN_model = model_ANN(8,device)
    ANN_model.main()


