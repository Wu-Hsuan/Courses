# -*- coding: utf-8 -*-

import numpy as np

#######################################
# 匯入必要的套件

# Maty note: 請同學根據所建模型的輸入特徵，也取一樣的特徵

# 要使用這個分類器！
# 擷取特徵的套件

# 處理檔案的套件
import os
import glob
import joblib                      # 儲存/讀取 SVM 模型

#SVM的相關套件
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split #驗證資料集
from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
# 我們要使用這個分類器！
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix

# 處理聲音的套件
import mfcc                        # 在上傳 mfcc.py 之後，使用裡面類別定義物件
import wave


# 匯出 csv 檔
import pandas as pd




# 把前面的幾個範例程式，整理成函式



def preprocess_data(data_root):
    result_files = glob.glob(os.path.join(data_root, "*.wav"))
    # 依照檔名排序
    result_files.sort(key=lambda f: int(f.split('/')[-1].split('.')[0]))

    sample_rate = 48000
    # 建立 MFCC 物件
    mfcc_obj = mfcc.MFCC(samprate=sample_rate)
    counter = 10000
    i = 0
    max_dim0 = 50


    # Maty note: 請同學根據所建模型的輸入特徵，也取一樣的特徵

    for data in result_files:
        print(f'Processing {data} ...')

        thewave = wave.open(data)

        # 把 wave 格式轉成 numpy array (int 16-bit)
        wavedata = thewave.readframes(-1)
        signal_wave = np.fromstring(wavedata, np.int16)

        # 計算 MFCC
        mfcc_array = mfcc_obj.sig2s2mfc(signal_wave)
        # 只保留前 [50,13] 筆資料
        # 多的要刪掉
        # 不夠要補零
        dim0 = mfcc_array.shape[0]
        if dim0 >= max_dim0:
            mfcc_array = mfcc_array[:max_dim0]
        else:
            data_pad = np.zeros([(max_dim0 - dim0), 13], dtype=float) #補零
            mfcc_array = np.concatenate( (mfcc_array, data_pad), axis=0 )
        #print(mfcc_array)
        # 把 2d 變 1d 陣列
        mfcc_array = np.reshape(mfcc_array, max_dim0*mfcc_array.shape[1] )

        # 判斷是哪個 label，放在它後面 (0~9)
        the_label = data.split('/')[-1].split('_')[0]
        print(the_label)
        print(mfcc_array.shape)
        if the_label in dict_label_mfcc:
            dict_label_mfcc[the_label] = np.vstack( [dict_label_mfcc[the_label], mfcc_array] )
        else:
            dict_label_mfcc[the_label] = mfcc_array
        print(dict_label_mfcc[the_label].shape)

        i+=1
        # 只跑有限次，檢查一下結果
        if i >= counter:
            break

    return dict_label_mfcc



    return return_feature_list

# Maty note: 如果你不是使用 SVM，也請正確的讀入你所訓練好的模型，才能正確預測喔！
# 讀入模型
model_name = 'maty_svm.model'
loaded_model = joblib.load(model_name)

IsLoadX = False

if not IsLoadX:
    # 處理測試資料集
    data_root = 'test'

    X = preprocess_data(data_root)

    # 避免重複讀檔，可以先把 X 存成 npy ; 之後再讀出使用即可
    np.save('mfcc_test_x.npy', X)

else:
    # 如果已存好 X 則可直接讀出來
    X = np.load('mfcc_test_x.npy')

# 預測結果
predicted_Y = loaded_model.predict(X)
# 印出結果
#print(loaded_model.predict(X))

# 把結果設定成 pandas dataframe, 後續便於儲存 csv
index_list = list(range(len(X)))
df = pd.DataFrame( { 'Id': index_list, 'Predicted': predicted_Y} )

# 輸出結果至 csv
df.to_csv('baseline.csv', index=False)


def textpurger(filename):
    return tagInputSTR

a = textpurger("test.txt")
print(a)