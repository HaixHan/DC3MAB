import actions
import pandas as pd
import numpy  as np
import random
import time
import collections
from pandas import Series,DataFrame
from tqdm import tqdm,trange
import operator
from functools import reduce
import copy
from itertools import chain  # 将二维list转为一维list
from matplotlib import pyplot
import matplotlib.pyplot as plt
from tkinter import _flatten
from IPython.core.pylabtools import figsize
import scipy.stats as stats
from multiprocessing import Pool

#根据user_cluster信息，初始化用户参数
def cal_user_cluster_parameters(user_cluster):
    user_num = 1000  
    m = np.identity(d)
    b = np.zeros((d,1))
    list1 = [m for x in range(0,user_num)]
    list2 = [b for x in range(0,user_num)]
    user_id = list(user_cluster['user_id'].values)
    user_parameters = pd.DataFrame(columns=('user_id','M','b'))
    user_parameters['user_id'] = user_id
    user_parameters['M'] = list1
    user_parameters['b'] = list2
    return user_parameters


def cal_cluster_user(user_cluster1):
    user_id = list(user_cluster1['user_id'].values)
    cluster = list(user_cluster1['cluster'].values)
    df = {'cluster':cluster,
      'user_id':user_id}
    df = pd.DataFrame(df)
    grouped = df.groupby('cluster')
    result = grouped['user_id'].unique()
    result2 = result.reset_index()
    return result2

#计算物品簇的特征向量
def cal_arm_features(item_list,item_features):
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

# 查找用户的参数
def find_user_parameters(u,user_parameters):
    user_index = user_parameters[user_parameters['user_id']==u].index[0]
    M = user_parameters.at[user_index,'M']
    b = user_parameters.at[user_index,'b']
    return M,b


# 查找每个类的参数
def find_AllCluster_parameters(cluster_parameters):
    M = [0]*cluster_num
    b = [0]*cluster_num
    w = [0] *cluster_num
    for i in range(cluster_num):
        cluster_index = cluster_parameters[cluster_parameters['cluster_id']==i].index[0]
        M[i] = cluster_parameters.at[cluster_index,'M']
        b[i] = cluster_parameters.at[cluster_index,'b']
        w[i]= cluster_parameters.at[cluster_index,'W']
    return M,b,w

# 计算某个cluster：no的参数
def caculate_cluster_parameters(no,ClusterToUser,user_parameters,d): 
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

#根据用户的参数信息，初始化所有簇参数
def cal_cluster_parameters(user_parameters,clusterToUser):
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
        sum_M,sum_b,sum_w = caculate_cluster_parameters(i,clusterToUser,user_parameters,d)
        cluster_parameters.at[i,'M'] = sum_M
        cluster_parameters.at[i,'b'] = sum_b
        cluster_parameters.at[i,'W'] = sum_w
    return cluster_parameters

#功能;一个物品获得的奖惩以及该物品是否击中
def get_reward(t,item,data): 
    S = 0   
    user_data = data[data['user_id']==t.user_id] 
    user_data.reset_index(drop=True,inplace=True)
    n = len(user_data)  
    m = user_data[(user_data['item_id']==t.item_id)&(user_data['cat_id']==t.cat_id)&(user_data['time_stamp']==t.time_stamp)].index.values[0]
    verification = user_data[m+1:n]    
    Hit = []
    flag = 0
    if len(verification[verification['item_id']==item])>0:   
        flag=1
        interact_data = verification[verification['item_id']==item]['action_type'].value_counts().to_frame().reset_index() 
        interact_data.rename(columns={ 'index':'action_type', 'action_type': 'number'}, inplace=True)  
        if len(interact_data[interact_data['action_type']==2])!=0:   
            S += 4
        elif len(interact_data[interact_data['action_type']==1])!=0:   
            S += 3
        elif len(interact_data[interact_data['action_type']==3])!=0:   
            S += 2
        elif len(interact_data[interact_data['action_type']==0])!=0:  
            S +=1
    return  S,flag  


# In[14]:
def DynCluster(u):  
    N = 10  #每一时刻推荐产生10个物品
    size = len(data_to_recommend[data_to_recommend['user_id']==u])  # 用户u的推荐记录长度
    Hit = [0] * size  
    j_item = [0] * size  #第i时刻推荐产生的推荐列表
    arm_num = 14 #臂的数量
    Hit_u = []  #用户u的所有击中产品
    init_R = [0] *arm_num #初始化每个臂的奖励
    res = [0] * arm_num  # 每个动作的结果值
    Lt = []  #用户u的所有推荐产品
    arm_features=[0]*arm_num
    s=0
    H=[]
    a_recommend = [0] * arm_num  #每个臂对应的候选集
    recom_len = list(data['item_id'][data['user_id']==u].values) #用户交互的物品个数（未去重）
    F1= 0
    recall = 0
    precision = 0
    total_R = 0
    a = 0.1
    count = 0 #对第i行用户序列进行推荐
    user_index = user_parameters[user_parameters['user_id']==u].index[0]
    Mu = user_parameters.at[user_index,'M']
    bu = user_parameters.at[user_index,'b']
    #  初始簇的参数
    clusters_M,clusters_b,clusters_w =  find_AllCluster_parameters(cluster_parameters)
    cluster_no =  userToCluster[userToCluster['user_id']==u]['cluster'].values[0]
    '''对每个用户'''
    for t in data_to_recommend[data_to_recommend['user_id']==u].reset_index(drop=True).itertuples():
        #此时用户的参数
        recall0=0
        precision0=0
        cluster_no =  userToCluster[userToCluster['user_id']==u]['cluster'].values[0] 
        cluster_w = clusters_w[cluster_no]
        cluster_M = clusters_M[cluster_no]
        cluster_b = clusters_b[cluster_no]
        A = actions.get_A(t,data)
        action_key = list(A.keys())  
        action_value = list(A.values())
        for j in range(arm_num):
            a_recommend[j] = action_value[j]
            a_recommend[j] = list(filter(lambda x: x!=t.item_id,a_recommend[j] ))    
            a_recommend[j] = list(set(a_recommend[j]))  
            arm_features[j] = cal_arm_features(a_recommend[j],item_features)
        for j in range(arm_num):  
            temp1 = (arm_features[j].T).dot(np.linalg.inv(cluster_M)).dot(arm_features[j])
            temp =  a*np.sqrt(temp1[0][0]*np.log(count+1))
            res[j] = (cluster_w.T).dot(arm_features[j]) + temp 
        best_action_key = np.argmax(res)              
        recommend_list = a_recommend[best_action_key]  
        if(len(recommend_list)<N):
            recommend_list_N = recommend_list
        else:
            recommend_list_N = random.sample(recommend_list,N);
        R_best = 0
        if(len(recommend_list_N)):
            j_item[count]=recommend_list_N
            H=[]
            for item in recommend_list_N:
                s,flag = get_reward(t,item,data)  
                R_best += s
                if(flag):
                    H.append(item)
            Hit[count] = H
        else:
            j_item[count],Hit[count],R_best =[],[],0
        total_R += R_best
        Mu = Mu + arm_features[best_action_key].dot(arm_features[best_action_key].T)
        bu = bu + R_best*arm_features[best_action_key]
        Wu =np.linalg.inv(Mu).dot(bu)
        user_parameters.at[user_index,'M'] = Mu
        user_parameters.at[user_index,'b'] = bu
        distance = [0]*cluster_num
        for i in range(cluster_num):
            distance[i] = np.sqrt(np.sum((Wu-clusters_w[i]) ** 2))
        user_cluster_id = np.argmin(distance)
        old_id = clusterToUser[clusterToUser['cluster']==cluster_no].index[0]
        new_id = clusterToUser[clusterToUser['cluster']==user_cluster_id].index[0]
        if(user_cluster_id!=cluster_no):
            index = userToCluster[userToCluster['user_id']==u].index[0]
            userToCluster.at[index,'cluster'] = user_cluster_id  
            user_list = list(clusterToUser.at[old_id,'user_id'])
            user_list.remove(u)
            clusterToUser.at[old_id, 'user_id']=np.array(user_list)
            user_list = list(clusterToUser.at[new_id,'user_id']) 
            user_list.append(u)
            clusterToUser.at[new_id, 'user_id']=np.array(user_list)
            new_new_M,new_new_b,new_new_w = caculate_cluster_parameters(new_id,clusterToUser,user_parameters,d)
            cluster_parameters.at[new_id,'M'] = new_new_M
            cluster_parameters.at[new_id,'b'] = new_new_b
            cluster_parameters.at[new_id,'W'] = new_new_w
            clusters_M[new_id] = new_new_M
            clusters_b[new_id] = new_new_b
            clusters_w[new_id] = new_new_w
        old_new_M,old_new_b,old_new_w = caculate_cluster_parameters(old_id,clusterToUser,user_parameters,d)
        cluster_parameters.at[old_id,'M'] = old_new_M
        cluster_parameters.at[old_id,'b'] = old_new_b
        cluster_parameters.at[old_id,'W'] = old_new_w
        clusters_M[old_id] = old_new_M
        clusters_b[old_id] = old_new_b
        clusters_w[old_id] = old_new_w      
        recall0 = len(set(Hit[count])) / len(set(recom_len)) #去重后
        if(len(set(j_item[count]))>0):
            precision0 = len(set(Hit[count]))/len(set(j_item[count]))
            precision+=precision0
        recall+=recall0
        if(recall0>0 or precision0>0):
            F0= (2*recall0*precision0)/(recall0+precision0)
            F1+=F0
        count+=1  
    return precision/size,recall/size,F1/size,total_R



def main(demension):
    global data
    global data_to_recommend
    global item_features
    global total_item
    global userToCluster
    global clusterToUser
    global user_parameters
    global cluster_parameters
    global d
    global cluster_num
    k=1  
    d = demension
    cluster_num = 48 
    data = pd.read_csv('Data/data.csv')  
    data_to_recommend = pd.read_csv('Data/recommendData.csv')
    item_features = pd.read_csv('Data/arm_features.csv')
    total_item = item_features['item_id'].values.tolist()
    users = list(set(data['user_id'].values))
    userToCluster = pd.read_csv('Data/userClusters.csv')
    clusterToUser = cal_cluster_user(userToCluster)
    user_parameters = cal_user_cluster_parameters(userToCluster)
    cluster_parameters = cal_cluster_parameters(user_parameters,clusterToUser)
    users = list(set(data_to_recommend['user_id'].values))
    user_id = users[:k]
    sum_p = 0
    sum_r = 0
    sum_HR= 0
    sum_reward= 0
    for u in tqdm(user_id):
        precision,recall,HR,reward = DynCluster(u)
        sum_p += precision
        sum_r += recall
        sum_HR+=HR
        sum_reward+=reward
    avg_p = sum_p / len(user_id)
    avg_r = sum_r / len(user_id)
    ave_HR = sum_HR / len(user_id)
    print("--------------维度为"+str(demension)+"的结果--------------")
    print("平均精确率：",avg_p)
    print("平均召回率：",avg_r)
    print("平均F1",ave_HR)
    print("推荐累计奖励",sum_reward)
    print("实验用户个数",len(user_id))


if __name__ == '__main__':
    demensions = [8]
    t1 = time.time()
    pool = Pool(1000)  #创建拥有10个进程数量的进程池
    pool.map(main,demensions)
    pool.close()#关闭进程池，不再接受新的进程
    pool.join()#主进程阻塞等待子进程的退出
    t2 = time.time()
    print ("并行执行时间：", int(t2-t1))




