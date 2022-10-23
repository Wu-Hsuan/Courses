import pandas as pd
import numpy as np


class AudioInfo:
    # 一筆資料以 tuple 型式輸入：(時間點timestamp, F0, F1, F2, F3, 音強intensity)；
    # 若多筆則以 list, pandas dataframe, or numpy array 資料建立
    def __init__(self, inputdata):
        self.data = inputdata
        
    # part - 1 : mel scale; 請以 list, pandas dataframe,  or numpy array 回傳零、一、或多筆結果 (25%)
    #            如果遇到 NAN，請過濾不使用
    def GetF0Mel(self):
        MelLIST = []
        for Hz_0 in self.data["F0_Hz"]:
            if type(Hz_0) == float:
                Mel_0 = 1125*np.log( 1 + Hz_0 / 700 )
                MelLIST.append(Mel_0 )
            elif Hz_0 == str("N/A"):
                pass

        for Hz_1 in self.data["F1_Hz"]:
            if type(Hz_1) == float:
                Mel_1 = 1125*np.log( 1 + Hz_1 / 700 )
                MelLIST.append(Mel_1 )
            elif Hz_1 == str("N/A"):
                pass

        for Hz_2 in self.data["F2_Hz"]:
            if type(Hz_2) == float:
                Mel_2 = 1125*np.log( 1 + Hz_2 / 700 )
                MelLIST.append(Mel_2 )
            elif Hz_2 == str("N/A"):
                pass

        for Hz_3 in self.data["F3_Hz"]:
            if type(Hz_3) == float:
                Mel_3 = 1125*np.log( 1 + Hz_3 / 700 )
                MelLIST.append(Mel_3 )
            elif Hz_3 == str("N/A"):
                pass
    
        return MelLIST
    
    # part - 2 : 計算有語音的時間（in second） (25%)
    def GetAudioPeriod(self):
   
        GetPeriodLIST =[]
        n = -1
        while n >=-1 and n<=22:
            n = n+1
            if self.data["Intensity_dB"][n] > 0:
                GetPeriodLIST.append(self.data["Time_sec"][n])
        max_sec = max(GetPeriodLIST)
        min_sec = min(GetPeriodLIST)
        GetPeriod = max_sec - min_sec
        return GetPeriod
    
    # part - 3 : 查詢並取得某個時間點內的資料；請以 list, pandas dataframe,  or numpy array 回傳零、一、或多筆結果 (25%)
    def GetData(self, start_sec, stop_sec):
        CTimeLIST = []
        n = -1
        while n >=-1 and n<=22:
            n = n+1
            if self.data["Time_sec"][n] >= start_sec and self.data["Time_sec"][n] <= stop_sec:
                CTimeLIST.append(self.data.iloc[n])
   
        return CTimeLIST

    # part - 4 : 取得最大、最小音量；回傳請以 tuple 形式 (25%)
    
    def GetMaxMinVolume(self):
        DBLIST=[]
        for db in self.data["Intensity_dB"]:
            if type(db) == float:
                DBLIST.append(db)
        max_db = max(DBLIST)
        min_db = min(DBLIST)
     
        return max_db, min_db
    
    # part - 5 : 取得平均值；藉由輸入標籤，決定回傳哪個平均值 (50%)
    # 輸入標籤：'F0', 'F1', 'F2', 'F3', 'Volume'
    def GetAvg(self, col_label):
        if col_label == 'F0':
            F_0LIST=[]
            for a in self.data["F0_Hz"]:
                F_0LIST.append(a)
            average_value = np.mean(F_0LIST)
       
        elif col_label == 'F1':
            F_1LIST=[]
            for a in self.data["F1_Hz"]:
                F_1LIST.append(a)
            average_value = np.mean(F_1LIST)
   
        elif col_label == "F2":
            F_2LIST=[]
            for a in self.data["F2_Hz"]:
                F_2LIST.append(a)
            average_value = np.mean(F_2LIST)

        elif col_label == "F3":
            F_3LIST=[]
            for a in self.data["F3_Hz"]:
                F_3LIST.append(a)
            average_value = np.mean(F_3LIST)

        return average_value
    
    # part - 6 : 取得變異數；藉由輸入標籤，決定回傳哪個變異數 (50%)
    # 輸入標籤：'F0', 'F1', 'F2', 'F3', 'Volume'
    def GetVar(self, col_label):
        if col_label == 'F0':
            F_0LIST=[]
            for a in self.data["F0_Hz"]:
                F_0LIST.append(a)
            variance_value = np.var(F_0LIST)
      
        elif col_label == 'F1':
            F_1LIST=[]
            for a in self.data["F1_Hz"]:
                F_1LIST.append(a)
            variance_value = np.var(F_1LIST)

        elif col_label == "F2":
            F_2LIST=[]
            for a in self.data["F2_Hz"]:
                F_2LIST.append(a)
            variance_value = np.var(F_2LIST)

        elif col_label == "F3":
            F_0LIST=[]
            for a in self.data["F3_Hz"]:
                F_3LIST.append(a)
            variance_value = np.var(F_3LIST)
  

        return variance_value
