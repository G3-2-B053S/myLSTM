"""
Function for Loss value
"""

import numpy as np

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)
from xDLUtils import Tools

"""
均方差损失函数
"""
class MseLoss:

    @staticmethod
    def loss(y,y_, n):
        corect_logprobs = Tools.mse(y, y_)
        data_loss = np.sum(corect_logprobs) / n
        delta = (y - y_) / n

        return data_loss, delta ,None

"""
二元交叉熵损失函数
"""
class SoftmaxCrossEntropyLoss:
    @staticmethod
    def loss(y,y_, n):
        y_argmax = np.argmax(y, axis=1)
        softmax_y = Tools.softmax(y)
        #print(np.shape(y),np.shape(softmax_y),np.shape(y_))
        acc = np.mean(y_argmax == y_)
        # loss
        corect_logprobs = Tools.crossEntropy(softmax_y, y_)
        data_loss = np.sum(corect_logprobs) / n
        # delta
        row,col=np.shape(y)
        delta=np.empty((row,col))
        for i in range(row):
            for j in range(col):
                if j+1 == y_[i]:
                    delta[i][j]=1-softmax_y[i][j]
                else:
                    delta[i][j] = softmax_y[i][j]
        return data_loss, delta, acc, y_argmax
