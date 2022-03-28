import numpy as np 
import pandas as pd
class Payoff:
    def get_reward(self,t,item,data): 
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