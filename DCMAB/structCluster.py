import pandas as pd
import numpy as np

class Clusters:
    def __init__(self,clusterToUser,cluster_parameters,cluster_num):
        self.clusterToUser = clusterToUser
        self.cluster_parameters = cluster_parameters
        self.cluster_num = cluster_num
    
    def cal_cluster_user(self,user_cluster1):
        user_id = list(user_cluster1['user_id'].values)
        cluster = list(user_cluster1['cluster'].values)
        df = {'cluster':cluster,
          'user_id':user_id}
        df = pd.DataFrame(df)
        grouped = df.groupby('cluster')
        result = grouped['user_id'].unique()
        result2 = result.reset_index()
        return result2
  
   # 计算某个cluster：no的参数
    def caculate_cluster_parameters(self,no,user_parameters,d): 
        ClusterToUser = self.clusterToUser
        users = ClusterToUser.at[no,'user_id']
        sum_M = np.identity(d)
        sum_b = np.zeros((d,1))
        I = np.identity(d)
        user_size = len(users)
        for u in users:
            index = user_parameters[user_parameters['user_id'] == u].index.tolist()[0]   
            M = user_parameters.at[index,'M']   
            b = user_parameters.at[index,'b']   
            sum_M += (M-I) 
            sum_b += b
        sum_w = np.linalg.inv(sum_M.T).dot(sum_b) 
        return sum_M,sum_b,sum_w

    #初始化用户簇的参数
    def cal_cluster_parameters(self,user_parameters,d):
        clusterToUser = self.clusterToUser
        cluster_num = self.cluster_num
        cluster_id = list(range(cluster_num))
        cluster_parameters = pd.DataFrame(columns=('cluster_id','M','b','W'))
        m = np.identity(d)
        b = np.zeros((d,1))
        list1 = [m for x in range(0,cluster_num)]
        list2 = [b for x in range(0,cluster_num)]
        cluster_parameters['cluster_id'] = cluster_id
        cluster_parameters['M'] = list1
        cluster_parameters['b'] = list2
        cluster_parameters['W'] = list2
        for i in range(cluster_num):
            sum_M,sum_b,sum_w = self.caculate_cluster_parameters(i,user_parameters,d)
            cluster_parameters.at[i,'M'] = sum_M
            cluster_parameters.at[i,'b'] = sum_b
            cluster_parameters.at[i,'W'] = sum_w
        return cluster_parameters
    
      # 查找每个类的参数
    def find_AllCluster_parameters(self):
        cluster_num = self.cluster_num
        cluster_parameters = self.cluster_parameters
        M = [0]*cluster_num
        b = [0]*cluster_num
        w = [0] *cluster_num
        for i in range(cluster_num):
            cluster_index = cluster_parameters[cluster_parameters['cluster_id']==i].index[0]
            M[i] = cluster_parameters.at[cluster_index,'M']
            b[i] = cluster_parameters.at[cluster_index,'b']
            w[i]= cluster_parameters.at[cluster_index,'W']
        return M,b,w


