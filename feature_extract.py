import numpy as np
from scipy.fftpack import dct
import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt

#采样率
sampling_rate = 16000
#预加重系数
preemphasis_coeff = 0.97
#帧长度，一般为一秒33到100帧，取40帧,帧周期25ms符合采样定律
frame_len = int(sampling_rate / 40)
#帧移，帧移/帧长取0.25
frame_shift = int(frame_len * 0.25)
#N点离散傅里叶变换，频谱共轭对称，前一半有效
fft_len = 512
NFFT = fft_len
#梅尔滤波器个数
num_filter = 30
nfilt = num_filter
#取MFCC的前12个维度为谱包络特征
num_mfcc = 30

def write_file(feats, file_name):
    f = open(file_name, 'w')
    (row, col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i, j]) + ',')
        f.write(']\n')
    f.close()
"""
def plot_spectrogram(spec, x_name,y_name, file_name):
    fig = plt.figure(figsize=(20, 5))
    #按照二维矩阵的值在不同位置上显示不同颜色
    heatmap = plt.pcolor(spec)
    #颜色条，显示颜色深度对应的幅值
    fig.colorbar(mappable=heatmap)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.tight_layout()
    plt.savefig('./pics/' + file_name)
"""
'''
function:创建二维信号图片
input:@signal_name信号名称,@x_name名称X轴,@y_name名称Y轴,@save_name保存图片名称,@x轴X值,@y轴Y值
    @mode为模式，为1的话清楚X轴刻度
output:None
'''

def Creat_Image(signal_name,x_name,y_name,save_name,x,y,mode = 0):
    fig = matplotlib.pyplot
    fig.figure(figsize = (15,5))
    if mode == 1:
        fig.xticks([])
    fig.title(signal_name)
    fig.xlabel(x_name)
    fig.ylabel(y_name)
    fig.plot(x,y)
    fig.savefig('./pics/' + save_name)

'''
function:对音频信号预加重,提高信号高频部分的能量
input:@signal输入要预加重的一维信号数组,@coeff预加重系数
output:预加重后的一维信号np数组
'''
def preemphasis(signal,coeff=preemphasis_coeff):
    return np.append(signal[0],signal[1:] - coeff * signal[:-1])


'''
function:预加重分帧加窗,减小FFT后能量泄露
input:@signal输入信号@frame_len桢长@frame_shift帧移@win窗函数
output: 分帧加窗后的二维数组，两维度分别为一帧信号和帧数
'''
def enframe(signal, frame_len=frame_len, frame_shift=frame_shift, win=np.hamming(frame_len)):
    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift) + 1
    frames = np.zeros((int(num_frames), frame_len))
    for i in range(int(num_frames)):
        frames[i, :] = signal[i * frame_shift:i * frame_shift + frame_len]
        frames[i, :] = frames[i, :] * win
    return frames

'''
function:对每一帧进行fft
input:@frames二维帧信号数组@fft_len帧长度
output: 对每帧信号FFT之后的二维数组，可能是复幅值，[信号帧数，每一帧有效频率分量个数]
'''
def get_spectrum(frames, fft_len=fft_len):
    cFFT = np.fft.fft(frames, n=fft_len)
    valid_len = int(fft_len / 2)
    spectrum = np.abs(cFFT[:, 0:(valid_len + 1)])
    return spectrum
'''
function:对每一帧经过梅尔滤波器并取log操作
input:@spectrum输入的经过fft之后的频谱[信号帧数，每一帧有效频率分量个数]@num_filter滤波器个数
output: 对每一帧经过梅尔滤波器并取log操作输出的二维数组[信号帧数，滤波器个数]
'''
def fbank(spectrum, sample_rate,num_filter=num_filter):
    low_freq_mel = 0#最低梅尔频率
    '''离散傅里叶变换的最高频率为延拓信号周期1/T，batch为k/NT,
    用公式计算梅尔频率的最大值,#因为离散傅里叶变换的频谱图为共
    轭对称的所以只有前一半频谱有用
    '''
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    #从mel频率的最高和最低频率之间等间隔划分成指定份数即梅尔滤波器个数
    #为了方便后面计算mel滤波器组，左右两边各补一个中心点
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    #反计算公式，计算线性频率与梅尔频率的对应
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))
    #保存每个梅尔滤波器频谱数据的nfilt维数组[滤波器个数，每一帧有效频率分量个数]
    fbank = np.zeros((nfilt,int(NFFT/2 + 1)))
    #每一个中心频率对应第几个梅尔滤波器的索引
    bin = np.floor(hz_points * (NFFT / 2) / (sample_rate / 2))

    #因为前面前后补了两个中心点，所以第一个滤波器的中心为bin[1],最后一个为bin[nfilt+1]
    #m为第几个滤波器的数据,k为其中第几个点
    for m in range(1, nfilt + 1):
        left = int(bin[m - 1])
        center = int(bin[m])
        right = int(bin[m + 1])
        for k in range(left,center):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(center,right):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    pow_frames = spectrum  # 频谱能量
    #梅尔滤波器矩阵与输入信号矩阵的转置相乘获得每一帧的fbank数据
    #维度为[信号帧数，滤波器个数]
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  #数据安全性
    #取log,单位db，是为了方便MFCC特征提取改乘为加
    filter_banks = 20 * np.log10(filter_banks)
    feats = filter_banks
    return feats

#MFCC特征提取，获取谱包络和谱细节MFCC(Mel频率倒谱系数Mel Frequency Cepstrum Coefficient)
'''
function:对取log后的fbank进行mfcc特征提取
input:@fbank对每一帧经过梅尔滤波器并取log操作输出的二维数组[信号帧数，滤波器个数]@num_mfcc取包络信息的维度
output: 12维MFCC特征[信号帧数，包络信息维度]
'''
def mfcc(fbank, num_mfcc=num_mfcc):
    #不将离散频谱对称补齐的话可以使用DCT来获取Fbank的倒谱系数
    #取了log之后包络与细节变成相加的关系，包络的频率较低取前k个谱密度为包络特征
    mfcc = dct(fbank, type=2, axis=1, norm='ortho')[:, 1: (num_mfcc + 1)]
    feats = mfcc
    return feats

def get_feature(wav,sample_rate,out_file):
    #Creat_Image('wav_signal','Time','Value','初始音频信号.png',np.arange(0,len(wav))/sampling_rate,wav,0)
    signal = preemphasis(wav)
    #Creat_Image('preemphasis_signal','Time','Value','预加重信号.png',np.arange(0,len(signal))/sampling_rate,signal,0)
    frames = enframe(signal)
    #Creat_Image('dev_cut_signal','Time','Value','分帧加窗后展开.png',np.arange(0,(frames.shape[0]) * (frames.shape[1])),frames.flatten(),1)
    spectrum = get_spectrum(frames)
    #Creat_Image('spectrum_signal','Frequency','Value','每一帧信号的频谱.png',np.arange(0,(spectrum.shape[0]) * (spectrum.shape[1])),spectrum.T.flatten(),1)
    fbank_feats = fbank(spectrum,sample_rate)
    mfcc_feats = mfcc(fbank_feats)
    row,col=mfcc_feats.shape
    if row < 200:
        print("wav file so short!")
    else:
        write_file(mfcc_feats[:200], out_file)
        print("feature extract successful!")
    #plot_spectrogram(fbank_feats.T, 'Frames','Filter Bank','fbank.png')
    #write_file(fbank_feats,'./test.fbank')
    #plot_spectrogram(mfcc_feats.T, 'pass','MFCC','mfcc.png')

#if __name__ == "__main__":
 #   main()