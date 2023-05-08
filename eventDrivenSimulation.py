import simpy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import truncexpon
import folium
import haversine
import requests
import math

#SCHEDULE_T=2 #Clock des Schedulers 
alpha=-np.log10(0.01)/100

def lognormal(mean):
    while True:
        file = np.random.lognormal(mean=np.log(mean),size=None)
        if file <= 5000000*8:
            break
    return file

def ue_to_df(users):
    df=pd.DataFrame()
    mr=[]
    mr2=[]
    queue=[]
    queue2=[]
    tbs=[]
    tbs2=[]
    tbs3=[]
    comp=[]
    sinr=[]
    sinr2=[]
    gain=[]
    qos=[]
    mr_list=[]
    mr2_list=[]
    uti=[]

    bit=[]
    bit2=[]
    mr_rel=[]
    pci1=[]
    pci2=[]
    
    for i in users:
        mr.append(i.mR)
        mr2.append(i.mR2)
        queue.append(i.queue.level)
        queue2.append(i.queue2.level)
        tbs.append(i.tbs)
        tbs2.append(i.tbs_comp)
        tbs3.append(i.tbs_phase)
        comp.append(i.comp)
        sinr.append(i.sinr)
        sinr2.append(i.sinr2)
        gain.append(i.gain)
        qos.append(i.qos)
        mr_list.append(i.mr_mon)
        mr2_list.append(i.mr2_mon)
        bit.append(i.bits)
        bit2.append(i.bits2)
        mr_rel.append(i.mR2/i.mR)
        pci1.append(i.cell1)
        pci2.append(i.cell2)
        uti.append(i.utilization)

    

    df['mR-no CoMP']=mr
    df['mR CoMP']=mr2
    df['mR gain']=mr_rel
    df['queue - no CoMP']=queue
    df['queue - CoMP']=queue2
    df['tbs no CoMP']=tbs
    df['tbs CoMP']=tbs2
    df['tbs CoMP phaseshift']=tbs3
    df['comp']=comp
    df['sinr-no CoMP']=sinr
    df['sinr CoMP']=sinr2
    #df['sinr CoMP phase']=sinr3
    df['sinr-gain']=gain
    df['qos']=qos
    df['TP1']=pci1
    df['TP2']=pci2
    df['utilization']=uti
    return df

def select_user_index(mode, ue_nr, ue_dict, cluster):
    index=np.zeros([len(cluster),ue_nr])
    if(type(mode)==str):
        if(mode=='deterministic'):
            index=np.zeros([len(cluster),ue_nr])
            for i in np.arange(0,len(cluster)):
                index[i,:]=np.arange(0,ue_nr) #take first 15 UEs out of list (deterministic mode)
        elif(mode=='random'):
            print('random')
            counter=0
            for i in cluster:
                index[counter]=random.sample(list(np.arange(1,len(ue_dict[i]))),ue_nr) #select random users out of list
                counter+=1   
        index=index.astype(int)
    else:
        index=mode
        
    return index

def seperate_comp_noCoMP(cluster,ue_per_tp):
    ue_comp=np.array([])
    ue_noCoMP={}
    for i in cluster:
        ue_list=ue_per_tp[i]
        new_ue_list=np.array([])
        for j in ue_list:
            if(j.comp == 0):
                new_ue_list=np.append(new_ue_list,j)
            else:
                ue_comp=np.append(ue_comp,j)
        ue_noCoMP[i]=new_ue_list #user without comp
    return ue_noCoMP, ue_comp

def df_to_ue_lists(df,cluster,thr,env):

    df_filter=df.groupby('TP1')
    ue_dict={}
    for i in cluster:
        counter=0
        ue_list=np.array([])
        df2=df_filter.get_group(i)
        for j in df2.index:
            ue_list=np.append(ue_list, ue(df.loc[j]['SINR [dB]'],df.loc[j]['SINR-CoMP [dB]'],df.loc[j]['SINR-CoMP with phaseshift [dB]'],df.loc[j]['TP1'],df.loc[j]['TP2'],df.loc[j]['lat'],df.loc[j]['lon'],env,df.loc[j]['qos'],thr,j,cluster))
        ue_dict[i]= ue_list
    return ue_dict


def get_user_from_cluster(ue_dict,cluster,ue_nr,index):
    ue_dict_red={}
    counter=0
    for i in cluster:
        ue_dict_red[i]=ue_dict[i][index[counter]]
        counter+=1

    ue_all=np.array([])
    for i in cluster:
        ue_all=np.append(ue_all,ue_dict_red[i])
    return ue_dict_red,ue_all


#function to monitor the level of the different queues
def monitor(value,monitor,env): 
    monitor.update({env.now: value})
    return monitor
 

def calculate_prb_number_strA(ue_all,cluster,max_prb,ue_nr):
    prb_number_comp={}
    if(len(cluster)==2):
        c=0
        for j in ue_all:
            if(j.comp==1):
                c+=1
        for i in cluster:
            prb_number_comp[i]=round(max_prb*(c*2)/(ue_nr*2+c))
    else:        
        for i in cluster:
            c=0
            c2=0
            for j in ue_all:
                if(j.comp==1):
                    if(j.cell1 ==i):
                        c+=1
                    if(j.cell2==i):
                        c2+=1
            prb_number_comp[i]=round(max_prb*(c+c2)/(ue_nr+c2))
    return prb_number_comp


def calculate_prb_number_strB(ue_all,cluster,max_prb,ue_nr):
    if(len(cluster)==2):
        prb={}
        a1=0
        a2=0
        sinr_max=0
        for i in ue_all:
            if(i.sinr2>sinr_max):
                sinr_max=i.sinr2
        c_max=(np.log2(1+np.exp(sinr_max/10)))

        for i in ue_all:
            if(i.comp==1):
                sinr=np.exp(i.sinr2/10)
                a2+=c_max*2/(np.log2(1+sinr))
            else:
                sinr=np.exp(i.sinr/10)
                a1+=c_max/(np.log2(1+sinr))
            for i in cluster:
                prb[i]=np.round(a2/(a2+a1)*max_prb)
    else:
        prb={}
        a1=0
        a2=0
        sinr_max=0
        count=0
        for j in cluster:
            for i in ue_all[count*ue_nr:ue_nr*(count+1)]:
                if(i.sinr2>sinr_max):
                    sinr_max=i.sinr2
                c_max=(np.log2(1+np.exp(sinr_max/10)))

            for i in ue_all[count*ue_nr:ue_nr*(count+1)]:
                if(i.comp==1):
                    sinr=np.exp(i.sinr2/10)
                    a2+=c_max*2/(np.log2(1+sinr))
                else:
                    sinr=np.exp(i.sinr/10)
                    a1+=c_max/(np.log2(1+sinr))
            prb[j]=np.round(a2/(a2+a1)*max_prb)
            count+=1
    return prb


def calculate_tbs(sinr,sinr2,sinr3):
    sinr=round(sinr*2)/2
    sinr2=round(sinr2*2)/2
    sinr3=round(sinr3*2)/2
    if(sinr>30):
        print('sinr out of range')
        sinr=30
    if(sinr2>30):
        sinr2=30
    if(sinr3>30):
        sinr3=30
    if(sinr<-10 ):
        print('sinr out of range')
        sinr=-10
    if(sinr2<-10):
        sinr2=-10
    if(sinr3<-10):    
        sinr3=-10
        #mapping=pd.read_csv('Data/sinr-tbs-mapping.csv',index_col='Unnamed: 0')
    mapping=pd.read_csv('Data/sinr-tbs-mapping.csv',index_col='Unnamed: 0')
    tbs=mapping.loc[sinr]['tbs']
    tbs2=mapping.loc[sinr2]['tbs']
    tbs3=mapping.loc[sinr3]['tbs']
    return tbs,tbs2,tbs3

class sched_inst:
    
    def __init__(self,env,cluster):
        self.sched_ut=0
        self.add_resources={}
        for i in cluster:
            self.add_resources.update({i:simpy.Container(env)})
        
    def metric_list_nC(self, users,sched_exp,counter):
        e1=sched_exp[0]
        e2=sched_exp[1]
        metric=np.array([])

        for i in users: 
            if(i.queue.level>0):
                metric=np.append(metric,((i.cp)**e1/(i.mR**e2)))
                i.mR=(1-1/counter)*i.mR #Ratenanpassung für alle Nutzer
                i.mR=(1-1/(counter+1))*i.mR
            elif(i.queue.level==0):
                metric=np.append(metric,-1)
                i.mR=(1-1/counter)*i.mR #Ratenanpassung für alle Nutzer
                i.mR=(1-1/(counter+1))*i.mR
            else:
                print('mistake')
        sched_user_list = (-metric).argsort() #sort UEs by metric that will be used for scheduling
        return sched_user_list

    def metric_list_C(self, users,sched_exp,counter,usage):
        e1=sched_exp[0]
        e2=sched_exp[1]
        metric=np.array([])
        for i in users: 
            if(usage=='nocomp'):
                cp=i.cp
            elif(usage=='comp'):
                cp=i.cp2
                
            if(i.queue2.level>0):
                metric=np.append(metric,((cp)**e1/(i.mR2)**e2))
                i.mR2=(1-1/counter)*i.mR2 #Ratenanpassung für alle Nutzer
                i.mR2=(1-1/(counter+1))*i.mR2
            elif(i.queue2.level==0):
                metric=np.append(metric,-1)
                i.mR2=(1-1/counter)*i.mR2 #Ratenanpassung für alle Nutzer
                i.mR2=(1-1/(counter+1))*i.mR2
            else:
                print('mistake')
        sched_user_list = (-metric).argsort() #sort UEs by metric that will be used for scheduling
        return sched_user_list


    def central_scheduler(self, env, users, SCHEDULE_T, cluster, prb_number,sched_metric,mode):

        alpha=-np.log10(0.01)/100
        while True:  
            counter=env.now/10+1 #counts the number of scheduling procedures
            yield env.timeout(SCHEDULE_T) #for each ms the scheduling is active -> per TTI
            metric=np.array([]) 

            for i in np.arange(len(users)):
                users[i].mon2= monitor(users[i].queue.level,users[i].mon2,env)
                users[i].mr2_mon=monitor(users[i].mR2,users[i].mr2_mon,env)

            sched_user_list=self.metric_list_C(users,sched_metric,counter,'comp')
            
            remaining_prb_list=prb_number.copy()
            
            k=0
            free_res=1
            while(free_res==1):
                if(k==len(sched_user_list)):
                    print('res free')
                    for z in remaining_prb_list:
                        if(remaining_prb_list[z]>0):
                            self.add_resources[z].put(remaining_prb_list[z])
                    break
                sched_user=sched_user_list[k]
                cell1=int(users[sched_user].cell1)
                cell2=int(users[sched_user].cell2)
                queue_size=users[sched_user].queue2.level
                tbs=users[sched_user].tbs
                if(mode=='phaseshift'):
                    tbs_comp=users[sched_user].tbs_phase
                else:
                    tbs_comp=users[sched_user].tbs_comp
                remaining_prbs=remaining_prb_list[cell1]
                remaining_prbs_c2=remaining_prb_list[cell2]
                
                sched_size=0
                if(remaining_prbs<=0):
                    print('nothing over')
                    k+=1
                    continue
                #cell to coordinate with has no resources left -> without comp
                elif(remaining_prbs_c2<=0):
                    if((queue_size/tbs)>remaining_prbs):
                        sched_size=remaining_prbs*tbs
                        remaining_prb_list[cell1]=0
                    else:
                        sched_size=queue_size
                        remaining_prb_list[cell1]=remaining_prbs-np.ceil(queue_size/tbs)
                elif((queue_size/tbs_comp)<=remaining_prbs and (queue_size/tbs_comp)<=remaining_prbs_c2 and queue_size>0):
                #comp can be used
                    sched_size=queue_size
                    remaining_prb_list[cell1]=remaining_prbs-np.ceil(queue_size/tbs_comp)
                    remaining_prb_list[cell2]=remaining_prbs_c2-np.ceil(queue_size/tbs_comp)
                #one of the cells has not enough resources left 
                elif((queue_size/tbs_comp)>remaining_prbs or (queue_size/tbs_comp)>remaining_prbs_c2):
                   
                    sched_size=min(remaining_prbs,remaining_prbs_c2)*tbs_comp
                    remaining_prb_list[cell1]=remaining_prbs-min(remaining_prbs,remaining_prbs_c2)
                    remaining_prb_list[cell2]=remaining_prbs_c2-(sched_size/tbs_comp)
                elif(queue_size<=0):
                    print('queues are empty -> resources adding')
                    for z in remaining_prb_list:
                        if(remaining_prb_list[z]>0):
                            self.add_resources[z].put(remaining_prb_list[z])
                    k=k+1
                    break
                else:
                    print('something went wrong')
                users[sched_user].mR2=users[sched_user].mR2+(1/counter)*sched_size
                if(users[sched_user].queue2.level!=0):
                    users[sched_user].queue2.get(sched_size)
                else:
                    print('queue size was 0')
                k=k+1
                free_res=0
                for i in cluster:
                    if(remaining_prb_list[i]!=0):
                        free_res=1
                
                ue_re=np.array([])
                for i in users:
                    ue_re=np.append(ue_re,i.queue2.level/i.tbs)
                    
                    
                



    #scheduler takes packets from the queues according to the capacity of each user
    def scheduler(self, env, users, SCHEDULE_T,cluster, prb_number, users2, prb_number2, sched_metric, cs):


        while True: #größte Warteschlange wird auch bedient
            counter=env.now/10+1 
            yield env.timeout(SCHEDULE_T) #for each ms the scheduling is active -> per TTI
            metric=np.array([]) 
            #print(env.now)
            for i in users:
                i.mon= monitor(i.queue.level,i.mon,env)
                i.mr_mon=monitor(i.mR,i.mr_mon,env)
            for i in users2:
                i.mr2_mon=monitor(i.mR2,i.mr2_mon,env)


            sched_user_list=self.metric_list_nC(users,sched_metric,counter)
            
            remaining_prbs=prb_number
            
            k=0
            while(remaining_prbs>0):
                if(k==len(sched_user_list)):
                    #print('remaining res comp1:',remaining_prbs)
                    break
                sched_user=sched_user_list[k]
                queue_size=users[sched_user].queue.level
                tbs=users[sched_user].tbs
                sched_size=0
                if((queue_size/tbs)<=remaining_prbs and queue_size>0):
                    sched_size=queue_size
                    remaining_prbs=remaining_prbs-np.ceil(queue_size/tbs)
                elif((queue_size/tbs)>remaining_prbs):
                    sched_size=remaining_prbs*tbs
                    remaining_prbs=remaining_prbs-np.ceil(sched_size/tbs)
                elif(queue_size==0):
                    
                    #self.rem_prb=monitor(remaining_prbs,self.rem_prb,env)
                    #ue_re=np.array([])
                    #for i in users:
                     #   ue_re=np.append(ue_re,i.queue.level)
                    #self.rem_req=monitor(sum(ue_re),self.rem_req,env)
                    #print('empty queue - no comp')
                    sched_size=0
                    break
                else:
                    print('something went wrong')

                users[sched_user].mR=users[sched_user].mR+(1/counter)*sched_size
                users[sched_user].queue.get(sched_size)
                users[sched_user].bits+=sched_size
                
                #self.rem_prb=monitor(remaining_prbs,self.rem_prb,env)
                ue_re=np.array([])
                for i in users:
                    ue_re=np.append(ue_re,i.queue.level/i.tbs)
                #self.rem_req=monitor(sum(ue_re),self.rem_req,env)
                k=k+1
                
                
            for u in users:
                if(u.queue.level>0):
                    #print(env.now)
                    u.utilization=u.utilization+1
                        


            #CoMP-Scheduling Process- Users with normal scheduling
            ############################

            sched_user_list = self.metric_list_C(users2,sched_metric,counter,'nocomp') #calculates the ordered list with ues
            
            
            index_get=0
            cell_number=int(users2[0].cell1)
            for i in cluster:
                if(i== cell_number):
                    index_get=i
            remaining_prbs=prb_number2+cs.add_resources[index_get].level
                
            if(cs.add_resources[index_get].level>0):
                cs.add_resources[index_get].get(cs.add_resources[index_get].level)
            #print(cs.add_resources[index_get].level)

            k=0
            #print('New scheduling round')
            while(remaining_prbs>0):
                if(k==len(sched_user_list)):
                    #print('remaining res comp1:',remaining_prbs)
                    break
                sched_user=sched_user_list[k]
                queue_size=users2[sched_user].queue2.level
                tbs=users2[sched_user].tbs
                sched_size=0
                if((queue_size/tbs)<=remaining_prbs and queue_size>0):
                    sched_size=queue_size
                    remaining_prbs=remaining_prbs-np.ceil(queue_size/tbs)
                elif((queue_size/tbs)>remaining_prbs):
                    sched_size=remaining_prbs*tbs
                    remaining_prbs=remaining_prbs-np.ceil(sched_size/tbs)

                elif(queue_size==0):
                    #self.rem_prb_c=monitor(remaining_prbs,self.rem_prb_c,env)
                    #ue_re=np.array([])
                    #for i in users:
                        #ue_re=np.append(ue_re,i.queue2.level/i.tbs)
                    
                    #self.rem_req_c=monitor(sum(ue_re),self.rem_req_c,env)
                    #print('empty queue - comp-scheduling: no comp user')
                    break
                else:
                    print('something went wrong')

                #print('normal scheduler - comp',env.now)
                #print('normal scheduler - comp -> id:',sched_user)
                #print('Rate before',users[sched_user].mR)
                users2[sched_user].mR2=users2[sched_user].mR2+(1/counter)*sched_size

                users2[sched_user].queue2.get(sched_size)
                users2[sched_user].bits2+=sched_size
                
                #self.rem_prb_c=monitor(remaining_prbs,self.rem_prb_c,env)
                ue_re=np.array([])
                for i in users2:
                    ue_re=np.append(ue_re,i.queue2.level/i.tbs)
                    
                #self.rem_req_c=monitor(sum(ue_re),self.rem_req_c,env)

                k=k+1
            ###########################
        

class ue:
    def __init__(self,sinr,sinr2,sinr3,cell1,cell2,x,y,env,qos,thr,id,cluster):
        self.sinr=sinr
        self.sinr2=sinr2
        self.tbs,self.tbs_comp,self.tbs_phase=calculate_tbs(sinr,sinr2,sinr3)
        
        self.qos=qos
        self.cp=0.5*0.7*20000000*np.log2(1+np.power(10,sinr/10))/8000
        self.cp2=0.5*0.7*20000000*np.log2(1+np.power(10,sinr2/10))/8000 #division by 8000 to determine number of bits that can be transmitted per TTI (1ms)
        self.cell1=cell1
        self.cell2=cell2
        self.mR=0.01 #mittlere Rate
        self.mR2=0.01 #mittlere Rate
        self.queue=simpy.Container(env)
        self.queue2=simpy.Container(env)
        self.mon={}
        self.mon2={}
        self.metric=self.sinr+self.queue.level
        self.metric2=self.sinr+self.queue.level
        self.gain=self.sinr2-self.sinr
        self.id=id
        self.bits=0
        self.bits2=0
        self.mr_mon={}
        self.mr2_mon={}
        self.x=x
        self.y=y
        self.utilization=0
        
        tp_dict={}
        c=0
            
        if(self.gain >thr and (self.cell2 in cluster)):
            self.comp=np.array(1)
        else:
            self.comp=np.array(0) 
        
    def best_effort_stat(self,env,time):
        while True:
            self.queue.put(4000) 
            self.queue2.put(4000)
            yield env.timeout(round(np.random.exponential(time)))

            

 