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
from itertools import chain  
from matplotlib import pyplot
import matplotlib.pyplot as plt
from tkinter import _flatten
from IPython.core.pylabtools import figsize
import scipy.stats as stats
from multiprocessing import Pool
import actions
import calArmFeatures
import calReward
import structCluster
import structUser


def DCMAB(u):  
    size = len(data_to_recommend[data_to_recommend['user_id']==u])  # 用户u的推荐记录长度
    Hit = [0] * size  #第i时刻推荐产生的击中列表
    j_item = [0] * size  #第i时刻推荐产生的推荐列表
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
    user_index = allUsers.user_parameters[allUsers.user_parameters['user_id']==u].index[0]
    Mu = allUsers.user_parameters.at[user_index,'M']
    bu = allUsers.user_parameters.at[user_index,'b']
    clusters_M,clusters_b,clusters_w =  allClusters.find_AllCluster_parameters()
    cluster_no =  userToCluster[userToCluster['user_id']==u]['cluster'].values[0]
    '''对每个用户'''
    for t in data_to_recommend[data_to_recommend['user_id']==u].reset_index(drop=True).itertuples():
        recall0=0
        precision0=0
        cluster_no =  userToCluster[userToCluster['user_id']==u]['cluster'].values[0] 
        cluster_w = clusters_w[cluster_no]
        cluster_M = clusters_M[cluster_no]
        cluster_b = clusters_b[cluster_no]
        '''每一时刻都要根据当前用户的行为序列，将总物品分为几个不同的物品子集'''
        A = actions.get_A(t,data)
        action_key = list(A.keys())  
        action_value = list(A.values())
        for j in range(arm_num):
            a_recommend[j] = action_value[j]
            a_recommend[j] = list(filter(lambda x: x!=t.item_id,a_recommend[j] ))    # 从候选集中去掉当前交互产品，保证有意义的推荐
            a_recommend[j] = list(set(a_recommend[j]))  
            arm_features[j] = arm_context.cal_arm_features(a_recommend[j],item_features,d)
        for j in range(arm_num):  # 对每个动作计算res
            temp1 = (arm_features[j].T).dot(np.linalg.inv(cluster_M)).dot(arm_features[j])
            temp =  a*np.sqrt(temp1[0][0]*np.log(count+1))
            res[j] = (cluster_w.T).dot(arm_features[j]) + temp 
        best_action_key = np.argmax(res)              # 最大A的索引
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
                s,flag = payoff.get_reward(t,item,data)  
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
        allUsers.user_parameters.at[user_index,'M'] = Mu
        allUsers.user_parameters.at[user_index,'b'] = bu
        distance = [0]*cluster_num
        for i in range(cluster_num):
            distance[i] = np.sqrt(np.sum((Wu-clusters_w[i]) ** 2))
        user_cluster_id = np.argmin(distance)
        old_id = allClusters.clusterToUser[allClusters.clusterToUser['cluster']==cluster_no].index[0]
        new_id = allClusters.clusterToUser[allClusters.clusterToUser['cluster']==user_cluster_id].index[0]
        if(user_cluster_id!=cluster_no):
            index = userToCluster[userToCluster['user_id']==u].index[0]
            userToCluster.at[index,'cluster'] = user_cluster_id  
            user_list = list(allClusters.clusterToUser.at[old_id,'user_id'])
            user_list.remove(u)
            allClusters.clusterToUser.at[old_id, 'user_id']=np.array(user_list)
            user_list = list(allClusters.clusterToUser.at[new_id,'user_id']) 
            user_list.append(u)
            allClusters.clusterToUser.at[new_id, 'user_id']=np.array(user_list)
            new_new_M,new_new_b,new_new_w = allClusters.caculate_cluster_parameters(new_id,allUsers.user_parameters,d)
            allClusters.cluster_parameters.at[new_id,'M'] = new_new_M
            allClusters.cluster_parameters.at[new_id,'b'] = new_new_b
            allClusters.cluster_parameters.at[new_id,'W'] = new_new_w
            clusters_M[new_id] = new_new_M
            clusters_b[new_id] = new_new_b
            clusters_w[new_id] = new_new_w
        old_new_M,old_new_b,old_new_w = allClusters.caculate_cluster_parameters(old_id,allUsers.user_parameters,d)
        allClusters.cluster_parameters.at[old_id,'M'] = old_new_M
        allClusters.cluster_parameters.at[old_id,'b'] = old_new_b
        allClusters.cluster_parameters.at[old_id,'W'] = old_new_w
        clusters_M[old_id] = old_new_M
        clusters_b[old_id] = old_new_b
        clusters_w[old_id] = old_new_w      
        recall0 = len(set(Hit[count])) / len(set(recom_len)) 
        if(len(set(j_item[count]))>0):
            precision0 = len(set(Hit[count]))/len(set(j_item[count]))
            precision+=precision0
        recall+=recall0
        if(recall0>0 or precision0>0):
            F0= (2*recall0*precision0)/(recall0+precision0)
            F1+=F0
        count+=1  
    return precision/size,recall/size,F1/size,total_R


# In[6]:


def main(k):
    global data
    global data_to_recommend
    global item_features
    global total_item
    global userToCluster
    global allUsers
    global allClusters
    global d
    global cluster_num
    global N
    global arm_num
    global arm_context
    global payoff
    N = 10
    d = 6
    cluster_num = 48
    arm_num = 14
    data = pd.read_csv('Data/data.csv')
    data_to_recommend = pd.read_csv('Data/recommendData.csv')
    item_features = pd.read_csv('Data/armFeatures.csv')
    total_item = item_features['item_id'].values.tolist()
    userToCluster = pd.read_csv('Data/userCluster.csv')
    test_users = list(set(data_to_recommend['user_id'].values))
    allUsers = structUser.Users(test_users,d,[])  
    allUsers.user_parameters = allUsers.cal_user_cluster_parameters()
    allClusters = structCluster.Clusters([],[],cluster_num) 
    allClusters.clusterToUser = allClusters.cal_cluster_user(userToCluster)
    arm_context = calArmFeatures.ArmFeatures()
    payoff = calReward.Payoff()
    allClusters.cluster_parameters = allClusters.cal_cluster_parameters(allUsers.user_parameters,d)
    user_id = test_users[:k]
    sum_p = 0
    sum_r = 0
    sum_HR= 0
    sum_reward= 0
    for u in tqdm(user_id):
        precision,recall,F1,reward = DCMAB(u)
        sum_p += precision
        sum_r += recall
        sum_HR+=F1
        sum_reward+=reward
    avg_p = sum_p / len(user_id)
    avg_r = sum_r / len(user_id)
    ave_HR = sum_HR / len(user_id)
    print("--------------------------最终结果-------------------------")
    print("平均精确率：",avg_p)
    print("平均召回率：",avg_r)
    print("平均F1",ave_HR)
    print("推荐累计奖励",sum_reward)
    print("实验用户个数",len(user_id))


if __name__ == '__main__':
    k = 1
    main(k)

