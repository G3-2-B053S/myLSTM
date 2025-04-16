"""
Main Function
"""

#from unittest import result
#import numpy as np
import logging.config
import random, time
#import matplotlib.pyplot as plt
import data_deal
import sys
import os
import tensorflow as tf
import feature_extract
from matplotlib import pyplot

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(rootPath + '/xDLbase')
sys.path.append(rootPath + '/xutils')

from xDLbase.xview import *
from xDLbase.fc import *
from xDLbase.rnn import *
from xDLbase.optimizers import *
from xDLbase.activators import *
from xDLbase.session import *

# create logger
exec_abs = os.getcwd()
log_conf = exec_abs + '/config/logging.conf'
logging.config.fileConfig(log_conf)
logger = logging.getLogger('main')

# 持久化配置
trace_file_path = 'D:/0tmp/'
exec_name = os.path.basename(__file__)
trace_file = trace_file_path + exec_name + ".data"

train_data = 'C:/Users/Administrator/PycharmProjects/pythonProject/files/train'
val_data = 'C:/Users/Administrator/PycharmProjects/pythonProject/files/val'
test_data = 'C:/Users/Administrator/PycharmProjects/pythonProject/files/test'

# General params
class Params:
    EPOCH_NUM = 10  # EPOCH
    MINI_BATCH_SIZE = 8  # batch_size
    ITERATION = 1  # 每batch训练轮数
    # LEARNING_RATE = 0.005  # Vanilla E5:loss 0.0014, 好于AdaDelta的0.0021
    LEARNING_RATE = 0.001  # LSTM
    # LEARNING_RATE = 0.002  # GRU
    # LEARNING_RATE = 0.1  # BiLSTM
    # LEARNING_RATE = 0.1  # BiGRU
    # LEARNING_RATE = 0.05  # BiGRU+ReLU
    # VAL_FREQ = 30  # val per how many batches
    VAL_FREQ = 5  # val per how many batches
    # LOG_FREQ = 10  # log per how many batches
    LOG_FREQ = 1  # log per how many batches

    HIDDEN_SIZE = 12  # LSTM中隐藏节点的个数,每个时间节点上的隐藏节点的个数，是w的维度.
    # RNN/LSTM/GRU每个层次的的时间节点个数，有输入数据的元素个数确定。
    NUM_LAYERS = 2  # RNN/LSTM的层数。
    # 设置缺省数值类型
    DTYPE_DEFAULT = np.float32
    INIT_W = 0.01  # 权重矩阵初始化参数

    DROPOUT_R_RATE = 1  # dropout比率
    TIMESTEPS = 1  # 循环神经网络的训练序列长度。
    PRED_STEPS = TIMESTEPS  # 预测序列长度
    TRAINING_STEPS = 10000  # 训练轮数。
    TRAINING_EXAMPLES = 10000  # 训练数据个数。
    TESTING_EXAMPLES = 1000  # 测试数据个数。
    SAMPLE_GAP = 0.01  # 采样间隔。
    VALIDATION_CAPACITY = TESTING_EXAMPLES - TIMESTEPS  # 验证集大小
    TYPE_K = 2  # 分类类别

    # 持久化开关
    TRACE_FLAG = False
    # loss曲线开关
    SHOW_LOSS_CURVE = True

    # Optimizer params
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1e-8
    EPS2 = 1e-10
    REG_PARA = 0.5  # 正则化乘数
    LAMDA = 1e-4  # 正则化系数lamda
    INIT_RNG = 1e-4

    # 并行度
    # TASK_NUM_MAX = 3
    # 任务池
    # g_pool = ProcessPoolExecutor(max_workers=TASK_NUM_MAX)

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
        y=1
        for dir in paths:
            data_path=train_data+'/'+dir
            if y == 1:
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
        y=341
        for dir in paths:
            data_path=val_data+'/'+dir
            if y == 341:
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
        y=381
        for dir in paths:
            data_path=test_data+'/'+dir
            if y == 381:
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

def main_rnn():
    logger.info('start..')
    # 初始化
    try:
        os.remove(trace_file)
    except FileNotFoundError:
        pass

    # if (True == Params.SHOW_LOSS_CURVE):
    # 配置训练结果展示图的参数
    view = ResultView(Params.EPOCH_NUM,
                      ['train_loss', 'val_loss', 'train_acc', 'val_acc'],
                      ['k', 'r', 'g', 'b'],
                      ['Iteration', 'Loss', 'Accuracy'],
                      Params.DTYPE_DEFAULT)
    s_t = 0
    # 构建监督学习数据，维度为N,T,D :(N,10,1)->(N,1,10)
    seqData = SeqData(Params.DTYPE_DEFAULT)
    dataRngs = seqData.getTrainRanges(Params.MINI_BATCH_SIZE)
    dataRngs_len = len(dataRngs)
    train_data_len = int(dataRngs_len * 0.85)
    validate_data_len = train_data_len + int(dataRngs_len * 0.1)
    # 定义网络结构，优化器参数，支持各层使用不同的优化器。
    # optmParamsRnn1 = (Params.BETA1, Params.BETA2, Params.EPS)
    # optimizer = AdagradOptimizer

    # optmParamsRnn1 = (Params.BETA1,Params.EPS)
    # optimizer = AdaDeltaOptimizer

    optmParamsRnn1 = (Params.BETA1, Params.BETA2, Params.EPS)
    optimizer = AdamOptimizer

    # rnn
    # rnn1 = RnnLayer('rnn1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,3,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE,Params.DTYPE_DEFAULT,Params.INIT_RNG)

    # LSTM
    rnn1 = LSTMLayer('lstm1', Params.MINI_BATCH_SIZE, Params.HIDDEN_SIZE, 3, optimizer, optmParamsRnn1,
                     Params.DROPOUT_R_RATE, Params.DTYPE_DEFAULT, Params.INIT_RNG)

    # BiLSTM
    # rnn1 = BiLSTMLayer('Bilstm1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,3,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE,Params.DTYPE_DEFAULT,Params.INIT_RNG)

    # GRU
    # rnn1 = GRULayer('gru1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,3,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE,Params.DTYPE_DEFAULT,Params.INIT_RNG)

    # BiGRU
    # rnn1 = BiGRULayer('bigru1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,3,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE,Params.DTYPE_DEFAULT,Params.INIT_RNG)

    # 全连接层优化器
    optmParamsFc1 = (Params.BETA1, Params.BETA2, Params.EPS)
    # RNN输出全部T个节点，FC层先把B,T,H拉伸成N,T*H, 再用仿射变换的W T*H,D 得到 N*D输出。
    # TIMESTEPS*HIDDEN_SIZE作为输入尺寸，PRED_STEPS表示预测值长度（步长），作为输出尺寸
    fc1 = FCLayer(Params.MINI_BATCH_SIZE, 2400, 400, NoAct, AdamOptimizer,
                  optmParamsFc1, True, Params.DTYPE_DEFAULT, Params.INIT_W)
    # 拼接网络层
    seqLayers = [rnn1, fc1]
    # 生成训练模型实例
    # sess = Session(seqLayers,MseLoss)
    sess = Session(seqLayers, SoftmaxCrossEntropyLoss)
    # lrt = Params.LEARNING_RATE
    # lrt = 1
    # 开始训练过程，训练EPOCH_NUM个epoch
    iter = 0
    for epoch in range(Params.EPOCH_NUM):
        # 获取当前epoch使用的learing rate
        # for key in Params.DIC_L_RATE.keys():
        #     if (epoch + 1) < key:
        #         break
        #     lrt = Params.DIC_L_RATE[key]
        lrt = Params.LEARNING_RATE

        # if loss_v1<1000* lrt:
        #    lrt  = lrt /10
        logger.info("epoch %2d, learning_rate= %.8f" % (epoch, lrt))
        # 随机打乱训练样本顺序，功能类似 shuffle=True
        #dataRngs = seqData.getTrainRanges(Params.MINI_BATCH_SIZE)
        # 当前epoch中，对n组sample进行训练，每个sample包含BATCH_SIZE个样本
        for batch in range(train_data_len):
            start = time.time()
            # 根据getTrainRanges打乱后的序号，取训练数据
            x, y_ = seqData.getTrainDataByRng(dataRngs[batch])
            # 训练模型，输出序列，只需要比较第0维；y.shape->(32,10,1),y_[:,:,0].shape->(32,10),相当于只是对y_进行降维，没有改变数据
            # _, loss_t = sess.train_steps(x, y_[:,:,0], lrt)
            # x(32, 1, 2) y_(32,)
            _, loss_t = sess.train_steps(x, y_, lrt)
            iter += 1

            if (batch % Params.LOG_FREQ == 0):  # 若干个batch show一次日志
                logger.info("epoch %2d-%3d, loss= %.8f st[%.1f]" % (epoch, batch, loss_t, s_t))
                fd = open(r"C:/Users/Administrator/PycharmProjects/pythonProject/files/file.txt", "a", encoding="utf-8")
                print("epoch %2d-%3d, loss= %.8f st[%.1f]" , epoch, batch, loss_t, s_t, file=fd)
                fd.close()

            # 使用随机验证样本验证结果
            #if (batch % Params.VAL_FREQ == 0 and (batch + epoch) > 0):
                # x_v, y_v = seqData.getValData(Params.VALIDATION_CAPACITY)
                #x_v, y_v = seqData.getTestData()
                # 多个出数值的序列，只需要比较0维
                # y, loss_v,_ = sess.validation(x_v, y_v[:,:,0])
                #y, loss_v, _, result = sess.validation(x_v, y_v)

                #logger.info('epoch %2d-%3d, loss=%f, loss_v=%f' % (epoch, batch, loss_t, loss_v))
                #fd = open(r"C:/Users/Administrator/PycharmProjects/pythonProject/files/file.txt", "a", encoding="utf-8")
                #print('epoch %2d-%3d, loss=%f, loss_v=%f',epoch, batch, loss_t, loss_v, file=fd)
                #fd.close()

                #if (True == Params.SHOW_LOSS_CURVE):
                    # view.addData(fc1.optimizerObj.Iter,
                    #view.addData(iter, loss_t, loss_v, 0, 0)
            s_t = time.time() - start

    logger.info('session end')
    fd = open(r"C:/Users/Administrator/PycharmProjects/pythonProject/files/file.txt", "a", encoding="utf-8")
    print('session end',file=fd)
    fd.close()
    test_acc = []
    count_acc = 0
    #test_loss = []
    for batch_idx in range(train_data_len,validate_data_len):
        x_v, y_v = seqData.getTrainDataByRng(dataRngs[batch_idx])
        y, loss_v, _, result = sess.validation(x_v, y_v)
        if (batch_idx % 10 == 0):  # 若干个batch show一次日志
            print("validate module batch: {} Accuracy: {} ".format(batch_idx, correct))
        correct = (result == y_v).sum() / Params.MINI_BATCH_SIZE
        count_acc += correct
        test_acc.append(correct)
    #pyplot.plot(y, label='predictions')
    #pyplot.title('predictions')
    #pyplot.legend()
    #pyplot.show()
    #pyplot.plot(y_v, label='real_curve')
    #pyplot.title('real_curve')
    #pyplot.legend()
    #pyplot.show()
    #pyplot.plot(loss_v, label='loss_v curve')
    #pyplot.title('loss_v')
    #pyplot.legend()
    #pyplot.show()
    print(result.shape)
    print(loss_v.shape)
    feature_extract.Creat_Image('predictions', 'Time', 'Value', 'y_predictions.png', np.arange(0, len(result)) / 16000, result, 0)
    feature_extract.Creat_Image('real_curve', 'Time', 'Value', 'y_real_curve.png', np.arange(0, len(y_v)) / 16000, y_v, 0)
    feature_extract.Creat_Image('test_acc', 'Time', 'Value', 'test_acc.png', np.arange(0, len(test_acc)) / 16000, test_acc, 0)
    exit()

if __name__ == '__main__':
    #语音文件预处理
    #directory = 'E:/wave/data_aishell/data_aishell/wav'
    #out_folder = 'E:/wave/data_aishell/current'
    #train_data = 'C:/Users/Administrator/PycharmProjects/pythonProject1/files/train'
    #data_deal.get_data(directory,out_folder,'train',train_data)
    #directory = 'E:/wave/data_aishell/data_aishell/dev'
    #val_data = 'C:/Users/Administrator/PycharmProjects/pythonProject1/files/val'
    #data_deal.get_data(directory,out_folder,'dev',val_data)
    #directory = 'E:/wave/data_aishell/data_aishell/test'
    #test_data = 'C:/Users/Administrator/PycharmProjects/pythonProject1/files/test'
    #data_deal.get_data(directory,out_folder,'test',test_data)
    #开始读取数据训练
    main_rnn()
    input()
