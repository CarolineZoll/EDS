import simpy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson

SCHEDULE_T=2 #Clock des Schedulers 
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
    comp=[]
    sinr=[]
    sinr2=[]
    gain=[]
    qos=[]
    mr_list=[]
    mr2_list=[]
    metric_sav=[]
    metric2_sav=[]
    bit=[]
    bit2=[]
    
    for i in users:
        mr.append(i.mR)
        mr2.append(i.mR2)
        queue.append(i.queue.level)
        queue2.append(i.queue2.level)
        tbs.append(i.tbs)
        tbs2.append(i.tbs2)
        comp.append(i.comp)
        sinr.append(i.sinr)
        sinr2.append(i.sinr2)
        gain.append(i.gain)
        qos.append(i.qos)
        mr_list.append(i.mr_mon)
        mr2_list.append(i.mr2_mon)
        bit.append(i.bits)
        bit2.append(i.bits2)

    

    df['mr']=mr
    df['mr2']=mr2
    df['queue']=queue
    df['queue2']=queue2
    df['tbs']=tbs
    df['tbs2']=tbs2
    df['comp']=comp
    df['sinr1']=sinr
    df['sinr2']=sinr2
    df['sinr-gain']=gain
    df['qos']=qos
    df['mr-Mon']=mr_list
    df['mr2-Mon']=mr_list
    df['bit']=bit
    df['bits']=bit2
    return df

def df_to_ue_lists(df,cluster,thr,env):

    cluster=[19,20]
    df_filter=df.groupby('PCI Serving')
    ue_dict={}
    for i in cluster:
        counter=0
        ue_list=np.array([])
        df2=df_filter.get_group(i)
        for j in df2.index:
            ue_list=np.append(ue_list, ue(df.iloc[j]['JT_1 SINR [lin]'],df.iloc[j]['JT_2 SINR [lin]'],df.iloc[j]['PCI Serving'],df.iloc[j]['PCI Coord'],env,df.iloc[j]['usage'], thr,df.iloc[j]['id']))
        ue_dict[i]= ue_list
    return ue_dict

#function to monitor the level of the different queues
def monitor(value,monitor,env): 
    monitor.update({env.now: value})
    return monitor

def calculate_prb_number(users,max_prb):
    count=0
    count2=0
    for i in np.arange(len(users)):
        if(users[i].comp ==1):
            if(users[i].qos==0):
                count+=1
            if(users[i].qos==1):
                count+=4
            if(users[i].qos==2):
                count+=75
                
        elif(users[i].comp ==0):
            if(users[i].qos==0):
                count2+=1
            if(users[i].qos==1):
                count2+=4
            if(users[i].qos==2):
                count2+=75
    prb_number=round(count/count2*max_prb)
    return prb_number

def get_dataframe(users):
    df=pd.DataFrame()
    sinr=np.array([])
    sinr2=np.array([])
    tbs=np.array([])
    queue=np.array([])
    for i in np.arange(np.size(users)):
        sinr=np.append(sinr,users[i].sinr)
        sinr2=np.append(sinr,users[i].sinr2)
        tbs=np.append(sinr,users[i].tbs)
        queue=np.append(sinr,users[i].queue)
    df['sinr']=sinr
    df['sinr2']=sinr2
    df['tbs']=tbs
    df['queue']=queue
    return df

def calculate_tbs(sinr,sinr2):
    sinr=int(sinr)
    sinr2=int(sinr2)
    if(sinr>30 or sinr2>30):
        print('sinr out of range')
        tbs=30
        tbs2=30
    elif(sinr<-10):
        print('sinr out of range')
        tbs=-10
        tbs2=-10
    else:
        mapping=pd.read_csv('Data/sinr-tbs-mapping.csv')
        tbs=mapping.iloc[sinr].values[1]
        tbs2=mapping.iloc[sinr2].values[1]
    return tbs,tbs2

def metric_list_nC(users,sched_exp,counter):
    e1=sched_exp[0]
    e2=sched_exp[1]
    metric=np.array([])

    for i in users: 
        if(i.qos==1 or i.qos==2):
            metric=np.append(metric, (alpha*i.queue.level*((i.cp)**e1/(i.mR)**e2)))  #list the metric of all UEs in the process 
            i.mR=(1-1/counter)*i.mR 
            i.mR=(1-1/(counter+1))*i.mR #Ratenanpassung für alle Nutzer    
        elif(i.qos==0):
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
        else:
            print('mistake')
    sched_user_list = (-metric).argsort() #sort UEs by metric that will be used for scheduling
    return sched_user_list

def metric_list_C(users,sched_exp,counter,usage):
    e1=sched_exp[0]
    e2=sched_exp[1]
    metric=np.array([])
    for i in users: 
        if(usage=='nocomp'):
            cp=i.cp
        elif(usage=='comp'):
            cp=i.cp2
            
        if(i.qos==1 or i.qos==2):
            metric=np.append(metric, (alpha*i.queue2.level*(cp**e1/(i.mR2)**e2)))  #list the metric of all UEs in the process 
            i.mR2=(1-1/counter)*i.mR2 #Ratenanpassung für alle Nutzer   
            i.mR2=(1-1/(counter+1))*i.mR2
        elif(i.qos==0):
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
        else:
            print('mistake')
    sched_user_list = (-metric).argsort() #sort UEs by metric that will be used for scheduling
    return sched_user_list


def central_scheduler(env, users, SCHEDULE_T,cluster, prb_number):
    
    alpha=-np.log10(0.01)/100
    while True: #größte Warteschlange wird auch bedient
        counter=env.now+1 #counts the number of scheduling procedures
        yield env.timeout(SCHEDULE_T) #for each ms the scheduling is active -> per TTI
        metric=np.array([]) 
        
        for i in np.arange(len(users)):
            users[i].mon2= monitor(users[i].queue.level,users[i].mon2,env)
            users[i].mr2_mon=monitor(users[i].mR2,users[i].mr2_mon,env)
        
        sched_user_list=metric_list_C(users,[1,1],env.now,'comp')
        
        remaining_prb_list={}
        for i in cluster:
            remaining_prb_list[i]=prb_number
            
        k=0
        free_res=1
        
        while(free_res==1):
            sched_user=sched_user_list[k]
            cell1=int(users[sched_user].cell1)
            cell2=int(users[sched_user].cell2)
            queue_size=users[sched_user].queue2.level
            tbs=users[sched_user].tbs
            tbs2=users[sched_user].tbs2
            remaining_prbs=remaining_prb_list[cell1]
            remaining_prbs_c2=remaining_prb_list[cell2]
            #print('Resourcen:',remaining_prb_list[cell1])
            #print('Resourcen:',remaining_prb_list[cell2])
            #serving cell has no resources left -> no scheduling 
            sched_size=0
            if(remaining_prbs_c2==0):
                #print('keine Res mehr frei')
                continue
            #cell to coordinate with has no resources left -> without comp
            elif(remaining_prbs_c2==0):
                #print('ohne CoMP')
                sched_size=queue_size
                remaining_prb_list[cell1]=remaining_prbs-np.ceil(queue_size/tbs)
                
            elif((queue_size/tbs2)<=remaining_prbs and (queue_size/tbs2)<=remaining_prbs_c2 and queue_size>0):
            #comp can be used
                #print('mit CoMP')
                sched_size=queue_size
                remaining_prb_list[cell1]=remaining_prbs-np.ceil(queue_size/tbs2)
                remaining_prb_list[cell2]=remaining_prbs_c2-np.ceil(queue_size/tbs2)
            #one of the cells has not enough resources left 
            elif((queue_size/tbs2)>remaining_prbs or (queue_size/tbs2)>remaining_prbs_c2):
                #print('mit CoMP - v2')
                sched_size=min(remaining_prbs,remaining_prbs_c2)*tbs2
                remaining_prb_list[cell1]=remaining_prbs-np.ceil(sched_size/tbs2)
                remaining_prb_list[cell2]=remaining_prbs_c2-np.ceil(sched_size/tbs2)
                
            elif(queue_size==0):
                #print('empty queue -comp')
                break
            else:
                print('something went wrong')
            users[sched_user].mR2=users[sched_user].mR2+(1/counter)*sched_size
            users[sched_user].queue2.get(sched_size)
            k=k+1
            free_res=0
            for i in cluster:
                if(remaining_prb_list[i]!=0):
                    free_res=1
                    
        
        


#scheduler takes packets from the queues according to the capacity of each user
def scheduler(env, users, SCHEDULE_T,cluster, prb_number, users2, prb_number2, sched_metric):

    bits1=0
    bits2=0
    while True: #größte Warteschlange wird auch bedient
        counter=env.now+1 
        yield env.timeout(SCHEDULE_T) #for each ms the scheduling is active -> per TTI
        metric=np.array([]) 
        
        for i in users:
            i.mon= monitor(i.queue.level,i.mon,env)
            i.mr_mon=monitor(i.mR,i.mr_mon,env)
        for i in users2:
            i.mr2_mon=monitor(i.mR2,i.mr2_mon,env)
        
        
        sched_user_list=metric_list_nC(users,[1,1],counter)

        remaining_prbs=prb_number
        k=0
        while(remaining_prbs>0):
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
                #print('empty queue - no comp')
                sched_size=0
                break
            else:
                print('something went wrong')
     
            users[sched_user].mR=users[sched_user].mR+(1/counter)*sched_size
            users[sched_user].queue.get(sched_size)
            users[sched_user].bits+=sched_size
            k=k+1

                
        #CoMP-Scheduling Process- Users with normal scheduling
        ############################
        
        sched_user_list = metric_list_C(users2,[1,1],counter,'nocomp') #calculates the ordered list with ues
        
        remaining_prbs=prb_number2
        k=0
        #print('New scheduling round')
        while(remaining_prbs>0):
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

            k=k+1
        ###########################
        

class ue:
    def __init__(self,sinr,sinr2,cell1,cell2,env,qos,thr,id):
        self.sinr=sinr
        self.sinr2=sinr2
        self.tbs,self.tbs2=calculate_tbs(sinr,sinr2)
        self.qos=qos
        self.cp=0.5*0.7*20000000*np.log2(1+np.power(10,sinr/10))/8000
        self.cp2=0.5*0.7*20000000*np.log2(1+np.power(10,sinr2/10))/8000 #division by 8000 to determine number of bits that can be transmitted per TTI (1ms)
        self.cell1=cell1
        self.cell2=cell2
        self.mR=0.1 #mittlere Rate
        self.mR2=0.1 #mittlere Rate
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
        if(self.gain >thr):
            self.comp=np.array(1)
        else:
            self.comp=np.array(0) 
        
    def rt_user(self,env,size):
        on_off=1
        counter=0
        start=0
        while True:
            if(start==0):
                start=1
                yield env.timeout(random.randint(0,200))
            elif(on_off==1 and counter<3000):
                self.queue.put(size) #20 bytes
                self.queue2.put(size) #20 bytes
                mon= monitor(self.queue.level,self.mon,env)
                #yield env.timeout(poisson.rvs(6, 1))
                #print('On Phase')
                #print(self.queue.level)
                counter=counter+20
                #print(counter)
                yield env.timeout(20) #every 20ms new packet
            elif(on_off==0):
                on_off=1

                yield env.timeout(3000) #3s no packet to be sent
            elif(on_off==1 and counter>=3000):
                on_off=0
                counter=0
                #print('change from ON-OFF')
            
    def user_packets(self,env,packet_arr):
        while True:
            #print('o-user')
            self.queue.put(2000)
            self.queue2.put(2000) 
            yield env.timeout(poisson.rvs(packet_arr, 1))
            

    def streaming_user(self,env):
        while True:
            #print('o-user')
            self.queue.put(3000) #1080p-> 1.5 Mbps (normal 1500)
            self.queue2.put(s3000) #1080p-> 1.5 Mbps 
            yield env.timeout(2)


    def best_effort(self,env):
            self.queue.put(100000000) 
            self.queue2.put(100000000)
            
    def ftp_user(self,env):
        while True:
            #print('o-user')
            size=lognormal(16000000)
            self.queue.put(size) #2MByte -> 16000000 Bit & 180s reading time
            self.queue2.put(size) 
            yield env.timeout(np.random.exponential(180*1000))
    
    #Noch sehr vereinfacht!!!
    def sinr_variator(self,env):
        change=round(np.random.normal(0,0.5))
        if((self.sinr+change)>-10 and (self.sinr+change)<30): 
            self.sinr=self.sinr +change
            self.sinr2=self.sinr2+change 
        yield env.timeout(2000)
