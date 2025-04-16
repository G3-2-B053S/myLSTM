import librosa
import tarfile
import os
import feature_extract
def get_wav(file,out_folder):
    # 创建或打开tar.gz文件
    with tarfile.open(file, 'r:gz') as tar:
        # 解压到目标文件夹
        tar.extractall(path=out_folder)

def delete_files_in_directory(path):
    files=os.listdir(path)
    for file in files:
        file_path=path+'/'+file
        os.remove(file_path)

def get_data(directory,out_folder,str,data_dir):
    sampling_rate = 16000
    rar_path = os.listdir(directory)
    for rar in rar_path:
        rar_file=directory+'/'+rar
        # 检查目录是否存在
        rar_name=rar.split('.')
        get_wav(rar_file,out_folder)
        filepath=out_folder + '/'+str +'/'+rar_name[0]
        out_dir = data_dir + '/' + rar_name[0]
        if not os.path.exists(out_dir):
            # 创建目录
            os.mkdir(out_dir)
            print(f"Directory '{out_dir}' created successfully")
        else:
            print(f"Directory '{out_dir}' already exists")
        filenames = os.listdir(filepath)
        for filename in filenames:
            wavpath=filepath+'/'+filename
            #print('gws'+wavpath)
            file=filename.split('.')
            size = os.path.getsize(wavpath)
            if size/1024 < 80:
                continue
            #读取音频信号存放于一维数组中，fs为采样频率，sr = None为原始采样率
            wav,fs = librosa.load(wavpath,sr = sampling_rate)
            out_file=out_dir+'/'+file[0]+'.txt'
            feature_extract.get_feature(wav,fs,out_file)
        delete_files_in_directory(filepath)
        os.rmdir(filepath)
        str_path=out_folder + '/'+str
        os.rmdir(str_path)
