import numpy as np
import pandas as pd
class Users: 
    def __init__(self,user_id,d,user_parameters):
        self.user_id = user_id
        self.d = d
        self.user_parameters = user_parameters
    #根据user_cluster信息，初始化用户参数
    def cal_user_cluster_parameters(self,):
        user_id = self.user_id
        user_num = len(user_id)  
        d = self.d
        m = np.identity(d)
        b = np.zeros((d,1))
        list1 = [m for x in range(0,user_num)]
        list2 = [b for x in range(0,user_num)]
        user_parameters = pd.DataFrame(columns=('user_id','M','b'))
        user_parameters['user_id'] = user_id
        user_parameters['M'] = list1
        user_parameters['b'] = list2
        return user_parameters

    # 查找用户的参数
    def find_user_parameters(u,self):
        user_parameters = self.user_parameters
        user_index = user_parameters[user_parameters['user_id']==u].index[0]
        M = user_parameters.at[user_index,'M']
        b = user_parameters.at[user_index,'b']
        return M,b