import numpy as np
import pandas as pd
#计算臂的特征向量
class ArmFeatures:
    def cal_arm_features(self,item_list,item_features,d):
        x = np.zeros((d,1))
        length = len(item_list)
        if(length>0):
            position = list(item_features['item_id'].values)
            for item_id in item_list:
                index = position.index(item_id)
                feature = np.array(item_features.iloc[index].values[1:]).reshape((d,-1)) 
                x+=feature
            x = x/length           
        return x