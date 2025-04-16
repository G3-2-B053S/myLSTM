import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import os
import random
from matplotlib import pyplot
import time
train_data = 'C:/Users/Administrator/PycharmProjects/pythonProject/files/train'
val_data = 'C:/Users/Administrator/PycharmProjects/pythonProject/files/val'
test_data = 'C:/Users/Administrator/PycharmProjects/pythonProject/files/test'

# 假设有一个简单的LSTM模型用于序列分类
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=0.01)
        self.fc1 = nn.Linear(hidden_size, num_classes)
        #self.fc2 = nn.Linear(num_classes, num_classes)
        #self.activate = nn.ReLU()

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 将x传递给LSTM
        out, _ = self.lstm(x, (h0, c0))
        #print(out.shape)
        # 将LSTM的最后一个时间步输出传递给全连接层
        out = self.fc1(out[:, -1, :])
        #out=self.activate(out)
        # 将LSTM的最后一个时间步输出传递给全连接层
        #out = self.fc2(out)
        return out


class SeqData(object):
    def __init__(self, dataType):
        self.dataType = dataType
        # 程序生成数据集
        self.x, self.y = self.TrainingData()
        self.x_v, self.y_v = self.getValData()
        self.x_t, self.y_t = self.readTestData()
        self.x=np.concatenate((self.x, self.x_v),axis=0)
        self.x_v=None
        self.x=np.concatenate((self.x, self.x_t),axis=0)
        self.x_t=None
        self.y=np.concatenate((self.y, self.y_v),axis=0)
        self.y_v=None
        self.y=np.concatenate((self.y, self.y_t),axis=0)
        self.y_t=None
        # 训练集样本数据索引表
        self.sample_range = [i for i in range(len(self.y))]
        #self.sample_range_v = [i for i in range(len(self.y_v))]
        # 测试集样本数据索引表
        #self.sample_range_t = [i for i in range(len(self.y_t))]

    def read_data(self,data_path,y):
        datas = os.listdir(data_path)
        curr_xArray = np.empty((len(datas), 200, 12))
        curr_yArray = np.empty((len(datas)))
        i=0
        for data in datas:
            file = open(data_path+'/'+data, "r")
            lines = file.readlines()
            file.close()
            j=0
            for line in lines:
                arr=line.split(' ')
                k=0
                for word in arr:
                    value = word.strip("[]")
                    if value == '\n':
                        continue
                    curr_xArray[i][j][k]=(np.float32(value))
                    k=k+1
                #curr_xArray[i][j]=eval(line)
                j=j+1
            curr_yArray[i]=y
            i=i+1
        fd=open(r"C:/Users/Administrator/PycharmProjects/pythonProject/files/file.txt","a",encoding="utf-8")
        print(data_path,np.shape(curr_xArray),file=fd)
        fd.close()
        return curr_xArray,curr_yArray

    # 求和结果分类，x1+x2>60
    def TrainingData(self):
        yArray = []
        xArray=[]
        paths = os.listdir(train_data)
        y=0
        for dir in paths:
            data_path=train_data+'/'+dir
            if y == 0:
                xArray,yArray=self.read_data(data_path,y)
            else:
                tempx, tempy = self.read_data(data_path, y)
                xArray = np.concatenate((xArray, tempx), axis=0)
                yArray = np.concatenate((yArray, tempy), axis=0)
            y=y+1
        # 监督学习数据 n*[X1, X2] -> n*[y]  <=> X.shape(sample, 1 , 2) -> Y.shape(sample, 1, 1)
        #trainX = np.array(xArray[:Params.TRAINING_EXAMPLES]).reshape(Params.TRAINING_EXAMPLES, 200, 12)
        #trainY = np.array(yArray[:Params.TRAINING_EXAMPLES])
        trainY=yArray
        trainX=xArray
        return trainX, trainY

    # 对训练样本序号按照miniBatchSize尺寸随机分成(sample_range%miniBatchSize)组
    # 剩余不足miniBatchSize的单独成一组
    # 功能类似shuffle = True
    def getTrainRanges(self, miniBatchSize):
        rangeAll = self.sample_range
        random.shuffle(rangeAll)
        rngs = [rangeAll[i:i + miniBatchSize] for i in range(0, len(rangeAll), miniBatchSize)]
        return rngs

    # 根据传入的训练样本序号，获取对应的输入x和输出y
    def getTrainDataByRng(self, rng):
        xs = np.array([self.x[sample] for sample in rng], self.dataType)
        values = np.array([self.y[sample] for sample in rng])
        return xs, values

    # 获取验证样本,不打乱，用于显示连续曲线
    def getValData(self):
        yArray = []
        xArray=[]
        paths = os.listdir(val_data)
        y=340
        for dir in paths:
            data_path=val_data+'/'+dir
            if y == 340:
                xArray,yArray=self.read_data(data_path,y)
            else:
                tempx, tempy = self.read_data(data_path, y)
                xArray = np.concatenate((xArray, tempx), axis=0)
                yArray = np.concatenate((yArray, tempy), axis=0)
            y=y+1
        # 监督学习数据 n*[X1, X2] -> n*[y]  <=> X.shape(sample, 1 , 2) -> Y.shape(sample, 1, 1)
        #valX = np.array(xArray[Params.TESTING_EXAMPLES:]).reshape(Params.TESTING_EXAMPLES, 200, 12)
        #valY = np.array(yArray[Params.TESTING_EXAMPLES:])
        valY=yArray
        valX=xArray
        return valX, valY

    #获取test样本,不打乱，用于显示连续曲线
    def readTestData(self):
        yArray = []
        xArray=[]
        paths = os.listdir(test_data)
        y=380
        for dir in paths:
            data_path=test_data+'/'+dir
            if y == 380:
                xArray,yArray=self.read_data(data_path,y)
            else:
                tempx, tempy = self.read_data(data_path, y)
                xArray = np.concatenate((xArray, tempx), axis=0)
                yArray = np.concatenate((yArray, tempy), axis=0)
            y=y+1
        # 监督学习数据 n*[X1, X2] -> n*[y]  <=> X.shape(sample, 1 , 2) -> Y.shape(sample, 1, 1)
        #testX = np.array(xArray[Params.TESTING_EXAMPLES:]).reshape(Params.TESTING_EXAMPLES, 200, 12)
        #testY = np.array(yArray[Params.TESTING_EXAMPLES:])
        testY=yArray
        testX=xArray
        return testX, testY
'''
    def getValRanges(self,miniBatchSize):
        rangeAll = self.sample_range_v
        rngs = [rangeAll[i:i + miniBatchSize] for i in range(0, len(rangeAll), miniBatchSize)]
        return rngs
    def getValDataByRng(self, rng):
        xs = np.array([self.x_v[sample] for sample in rng], self.dataType)
        values = np.array([self.y_v[sample] for sample in rng])
        return xs, values
'''
'''
    def getTestRanges(self,miniBatchSize):
        rangeAll = self.sample_range_t
        rngs = [rangeAll[i:i + miniBatchSize] for i in range(0, len(rangeAll), miniBatchSize)]
        return rngs
    def getTestDataByRng(self, rng):
        xs = np.array([self.x_t[sample] for sample in rng], self.dataType)
        values = np.array([self.y_t[sample] for sample in rng])
        return xs, values
        '''
# 示例数据
batch_size = 32
seq_len = 200
input_size = 12
hidden_size = 200
num_layers = 3
num_classes = 400
batch_first = True
lr = 0.001
start = time.time()
print("start time is :", start)
# 实例化模型并优化器
model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
seqData = SeqData(np.float32)
Triain_ACC=[]
Triain_loss=[]
VAL_ACC=[]
VAL_loss=[]
# 训练过程
epochs = 10
dataRngs = seqData.getTrainRanges(batch_size)
dataRngs_len=len(dataRngs)
train_data_len=int(dataRngs_len*0.85)
validate_data_len=train_data_len+int(dataRngs_len*0.1)

for epoch in range(epochs):
    count_acc=0
    for batch_idx in range(train_data_len):
        x, y_ = seqData.getTrainDataByRng(dataRngs[batch_idx])
        inpt = torch.tensor(x)
        target = torch.tensor(y_)
        optimizer.zero_grad()
        output = model(inpt)
        y_argmax = torch.argmax(output, axis=1)
        #print(output.dtype)
        #print(y_argmax)
        #print(target)
        loss = nn.CrossEntropyLoss()(output, target.to(torch.int64))
        #loss.requires_grad_()
        loss.backward()
        optimizer.step()
        correct = (y_argmax.to(target.dtype) == target).sum().float() / 32
        count_acc+=correct
        if (batch_idx % 5 == 0):  # 若干个batch show一次日志
            print("training epoch: {} batch: {} Accuracy: {} loss {}".format(epoch, batch_idx, correct, loss.item()))
    Triain_ACC.append(count_acc/(len(dataRngs)-1))
    Triain_loss.append(loss.item())
        # 使用随机验证样本验证结果
    #dataRngs_v = seqData.getValRanges(batch_size)
    count_acc=0
    for batch_idx in range(train_data_len,validate_data_len):
        x_v, y_v = seqData.getTrainDataByRng(dataRngs[batch_idx])
        # 多个出数值的序列，只需要比较0维
        inpt_v = torch.tensor(x_v)
        target_v = torch.tensor(y_v)
        output_v = model(inpt_v)
        y_argmax = torch.argmax(output_v, axis=1)
        loss = nn.CrossEntropyLoss()(output_v, target_v.to(torch.int64))
        correct = (y_argmax.to(target_v.dtype) == target_v).sum().float() / 32
        count_acc += correct
        if (batch_idx % 5 == 0):  # 若干个batch show一次日志
            print("validate module epoch: {} batch: {} Accuracy: {} loss {}".format(epoch, batch_idx, correct, loss.item()))
    VAL_ACC.append(count_acc/(len(dataRngs)-1))
    VAL_loss.append(loss.item())
# 训练完毕后，可以用模型进行预测
# ...
#dataRngs_t = seqData.getTestRanges(batch_size)
count_acc=0
test_acc=[]
test_loss=[]
for batch_idx in range(validate_data_len,dataRngs_len-1):
    x_t, y_t = seqData.getTrainDataByRng(dataRngs[batch_idx])
    # 多个出数值的序列，只需要比较0维
    inpt_t = torch.tensor(x_t)
    target_t = torch.tensor(y_t)
    output_t = model(inpt_t)
    y_argmax = torch.argmax(output_t, axis=1)
    loss = nn.CrossEntropyLoss()(output_t, target_t.to(torch.int64))
    correct = (y_argmax.to(target_t.dtype) == target_t).sum().float() / 32
    if (batch_idx % 5 == 0):  # 若干个batch show一次日志
        print("test module epoch: {} batch: {} Accuracy: {} loss {}".format(epoch, batch_idx, correct, loss.item()))
    test_acc.append(correct)
    test_loss.append(loss.item())
end = time.time()
print("end time is：",end)
#pyplot.plot(loss.detach().numpy(), label='loss')
pyplot.plot(Triain_ACC, label='train Accuracy')
pyplot.title('training accuracy')
pyplot.legend()
pyplot.show()
pyplot.plot(Triain_loss, label='train loss')
pyplot.title('training loss function')
pyplot.legend()
pyplot.show()
pyplot.plot(VAL_ACC, label='validate Accuracy')
pyplot.title('validate accuracy')
pyplot.legend()
pyplot.show()
pyplot.plot(VAL_loss, label='validate loss')
pyplot.title('validate loss function')
pyplot.legend()
pyplot.show()
pyplot.plot(test_acc, label='test Accuracy')
pyplot.title('test accuracy')
pyplot.legend()
pyplot.show()
pyplot.plot(test_loss, label='test loss')
pyplot.title('test loss function')
pyplot.legend()
pyplot.show()