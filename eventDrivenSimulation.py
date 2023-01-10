import simpy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import truncexpon

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
    tbs3=[]
    comp=[]
    sinr=[]
    sinr2=[]
    gain=[]
    qos=[]
    mr_list=[]
    mr2_list=[]
    #metric_sav=[]
    #metric2_sav=[]
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
    return df

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
 
    
def calculate_prb_number(users,max_prb):
    count=0
    for i in users:
        if(i.comp==1):
            count+=1
    prb_number=round((count*2)/(count+len(users))*max_prb)
    return prb_number

def calculate_prb_number_comp(ue_all,cluster,max_prb,ue_nr):
    prb_number_comp={}
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
    
    def __init__(self,env):
        self.rem_prb={}
        self.rem_req={}
        self.rem_prb_c={}
        self.rem_req_c={}
        
        
    def metric_list_nC(self, users,sched_exp,counter):
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

    def metric_list_C(self, users,sched_exp,counter,usage):
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


    def central_scheduler(self, env, users, SCHEDULE_T, cluster, prb_number,sched_metric,mode):

        alpha=-np.log10(0.01)/100
        while True: 
            counter=env.now+1 #counts the number of scheduling procedures
            yield env.timeout(SCHEDULE_T) #for each ms the scheduling is active -> per TTI
            metric=np.array([]) 

            for i in np.arange(len(users)):
                users[i].mon2= monitor(users[i].queue.level,users[i].mon2,env)
                users[i].mr2_mon=monitor(users[i].mR2,users[i].mr2_mon,env)

            sched_user_list=self.metric_list_C(users,sched_metric,env.now,'comp')
            
            remaining_prb_list=prb_number.copy()
            
            k=0
            free_res=1
            
            while(free_res==1):
                #print('sched ')
                if(k==len(sched_user_list)):
                    #print('remaining res comp-central:',free_res)
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
                if(remaining_prbs==0):
                    #print('keine Res mehr frei')
                    k+=1
                    continue
                #cell to coordinate with has no resources left -> without comp
                elif(remaining_prbs_c2==0):
                    #print('ohne CoMP')
                    if((queue_size/tbs)>remaining_prbs):
                        sched_size=remaining_prbs*tbs
                        remaining_prb_list[cell1]=0
                    else:
                        sched_size=queue_size
                        remaining_prb_list[cell1]=remaining_prbs-np.ceil(queue_size/tbs)

                elif((queue_size/tbs_comp)<=remaining_prbs and (queue_size/tbs_comp)<=remaining_prbs_c2 and queue_size>0):
                #comp can be used
                    #print('mit CoMP')
                    sched_size=queue_size
                    remaining_prb_list[cell1]=remaining_prbs-np.ceil(queue_size/tbs_comp)
                    remaining_prb_list[cell2]=remaining_prbs_c2-np.ceil(queue_size/tbs_comp)
                #one of the cells has not enough resources left 
                elif((queue_size/tbs_comp)>remaining_prbs or (queue_size/tbs_comp)>remaining_prbs_c2):
                    #print('mit CoMP - v2')
                    sched_size=min(remaining_prbs,remaining_prbs_c2)*tbs_comp
                    remaining_prb_list[cell1]=remaining_prbs-min(remaining_prbs,remaining_prbs_c2)
                    remaining_prb_list[cell2]=remaining_prbs_c2-(sched_size/tbs_comp)
                elif(queue_size==0):
                    self.rem_prb_c=monitor([remaining_prb_list[cell1],remaining_prb_list[cell2]],self.rem_prb_c,env)
                    ue_re=np.array([])
                    for i in users:
                        ue_re=np.append(ue_re,i.queue2.level/i.tbs)
                    self.rem_req_c=monitor(sum(ue_re),self.rem_req_c,env)
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
                
                self.rem_prb_c=monitor([remaining_prb_list[cell1],remaining_prb_list[cell2]],self.rem_prb_c,env)
                ue_re=np.array([])
                for i in users:
                    ue_re=np.append(ue_re,i.queue2.level/i.tbs)
                    
                self.rem_req_c=monitor(sum(ue_re),self.rem_req_c,env)
                    
                



    #scheduler takes packets from the queues according to the capacity of each user
    def scheduler(self, env, users, SCHEDULE_T,cluster, prb_number, users2, prb_number2, sched_metric):


        while True: #größte Warteschlange wird auch bedient
            counter=env.now+1 
            yield env.timeout(SCHEDULE_T) #for each ms the scheduling is active -> per TTI
            metric=np.array([]) 

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
                    
                    self.rem_prb=monitor(remaining_prbs,self.rem_prb,env)
                    ue_re=np.array([])
                    for i in users:
                        ue_re=np.append(ue_re,i.queue.level)
                    self.rem_req=monitor(sum(ue_re),self.rem_req,env)
                    #print('empty queue - no comp')
                    sched_size=0
                    break
                else:
                    print('something went wrong')

                users[sched_user].mR=users[sched_user].mR+(1/counter)*sched_size
                users[sched_user].queue.get(sched_size)
                users[sched_user].bits+=sched_size
                
                self.rem_prb=monitor(remaining_prbs,self.rem_prb,env)
                ue_re=np.array([])
                for i in users:
                    ue_re=np.append(ue_re,i.queue.level/i.tbs)
                self.rem_req=monitor(sum(ue_re),self.rem_req,env)
                k=k+1


            #CoMP-Scheduling Process- Users with normal scheduling
            ############################

            sched_user_list = self.metric_list_C(users2,sched_metric,counter,'nocomp') #calculates the ordered list with ues

            remaining_prbs=prb_number2
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
                    self.rem_prb_c=monitor(remaining_prbs,self.rem_prb_c,env)
                    ue_re=np.array([])
                    for i in users:
                        ue_re=np.append(ue_re,i.queue2.level/i.tbs)
                    
                    self.rem_req_c=monitor(sum(ue_re),self.rem_req_c,env)
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
                
                self.rem_prb_c=monitor(remaining_prbs,self.rem_prb_c,env)
                ue_re=np.array([])
                for i in users2:
                    ue_re=np.append(ue_re,i.queue2.level/i.tbs)
                    
                self.rem_req_c=monitor(sum(ue_re),self.rem_req_c,env)

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
        
        tp_dict={}
        c=0
            
        if(self.gain >thr and (self.cell2 in cluster)):
            self.comp=np.array(1)
        else:
            self.comp=np.array(0) 
        
    def rt_user(self,env,size):
        on_off=1
        counter=0
        start=0
        max_counter=np.random.exponential(3000)
        while True:
            if(start==0):
                start=1
                yield env.timeout(random.randint(0,200)) 
            elif(on_off==1 and counter<max_counter):
                self.queue.put(size) #20 bytes
                self.queue2.put(size) #20 bytes
                mon= monitor(self.queue.level,self.mon,env)
                counter=counter+20
                yield env.timeout(20) #every 20ms new packet
            elif(on_off==0):
                on_off=1
                off_time=(truncexpon.rvs(4.9)+2)*1000 #to be checked -> mean=3 and upper limit of 6.9
                yield env.timeout(off_time) #3s no packet to be sent
            elif(on_off==1 and counter>=max_counter):
                on_off=0
                counter=0
                max_counter=np.random.exponential(3)*1000
                #print('change from ON-OFF')
   
    def ftp_user(self,env):
        while True:
            #print('o-user')
            size=lognormal(16000000)
            self.queue.put(size) #2MByte -> 16000000 Bit & 180s reading time
            self.queue2.put(size) 
            yield env.timeout(np.random.exponential(180*1000))

    def best_effort(self,env,size):
        while True:
            self.queue.put(size) 
            self.queue2.put(size)
            yield env.timeout(10)

    def best_effort_stat(self,env,time):
        while True:
            self.queue.put(4000) 
            self.queue2.put(4000)
            yield env.timeout(round(np.random.exponential(time)))
            
    def streaming_user(self,env):
        while True:
            #print('o-user')
            self.queue.put(3000) #1080p-> 1.5 Mbps 
            self.queue2.put(3000) #1080p-> 1.5 Mbps 
            yield env.timeout(2)
    
    #Noch sehr vereinfacht!!!
    def sinr_variator(self,env):
        change=round(np.random.normal(0,0.5))
        if((self.sinr+change)>-10 and (self.sinr+change)<30): 
            self.sinr=self.sinr +change
            self.sinr2=self.sinr2+change 
        yield env.timeout(2000)
        