import simpy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson

SCHEDULE_T=2 #Clock des Schedulers 

#function to monitor the level of the different queues
def monitor(queue,monitor,env): 
    monitor.update({env.now: queue.level})
    return monitor

def calculate_tbs(sinr):
    mapping=pd.read_csv('Data/sinr-tbs-mapping.csv')
    tbs=mapping.iloc[sinr].values[1]
    return tbs

#scheduler takes packets from the queues according to the capacity of each user
def scheduler(env, users, SCHEDULE_T):
    
    prb_number=100
    counter=1 #counts the number of scheduling procedures
    alpha=-np.log10(0.01)/100
    
    while True: #größte Warteschlange wird auch bedient
        
        yield env.timeout(SCHEDULE_T) #for each ms the scheduling is active -> per TTI
        metric=np.array([]) 
        users[0].mon= monitor(users[0].queue,users[0].mon,env)
        users[250].mon= monitor(users[250].queue,users[250].mon,env)
        users[450].mon= monitor(users[450].queue,users[450].mon,env)
        
        for i in np.arange(np.size(users)): 
            if(users[i].qos==1):
                metric=np.append(metric, (alpha*users[i].queue.level*(users[i].cp/users[i].mR)))  #list the metric of all UEs in the process 
                users[i].mR=(1-1/counter)*users[i].mR #Ratenanpassung für alle Nutzer
            
            elif(users[i].qos==0):
                if(users[i].queue.level>0):
                    metric=np.append(metric,(users[i].cp/users[i].mR))
                    users[i].mR=(1-1/counter)*users[i].mR #Ratenanpassung für alle Nutzer
                if(users[i].queue.level==0):
                    metric=np.append(metric,0)
                    users[i].mR=(1-1/counter)*users[i].mR #Ratenanpassung für alle Nutzer

        
        sched_user_list = (-metric).argsort() # wählt die 5 UEs mit größter Metrik aus #find UE with maximal metric that will be used for scheduling
        
        #print(sched_user_list)    
        #print(users[2].queue.level)
        remaining_prbs=prb_number
        
        k=0
        print('New scheduling round')
        while(remaining_prbs>0):
            sched_user=sched_user_list[k]

            queue_size=users[sched_user].queue.level
            tbs=users[sched_user].tbs

            if((queue_size/tbs)<=remaining_prbs and queue_size>0):
                sched_size=queue_size
                remaining_prbs=remaining_prbs-np.ceil(queue_size/tbs)
                
            elif((queue_size/tbs)>remaining_prbs):
                sched_size=remaining_prbs*tbs
                remaining_prbs=remaining_prbs-sched_size
                
            elif(queue_size==0):
                print('empty queue')
                break
            else:
                print('something went wrong')
            print('user:',sched_user)    
            print('queue size before:',queue_size)
            print('mR before:',users[sched_user].mR)
            users[sched_user].mR=users[sched_user].mR+(1/counter)*sched_size
            users[sched_user].queue.get(sched_size)
            print('queue size after:',users[sched_user].queue.level)
            print('mR afer:',users[sched_user].mR)
            k=k+1
        counter=counter+1
        

        
            
class ue:
    def __init__(self,sinr,sinr2,cell1,cell2,env,qos):
        self.sinr=sinr
        self.sinr2=sinr2
        self.tbs=calculate_tbs(sinr)
        self.tbs2=calculate_tbs(sinr2)
        self.qos=qos
        self.cp=0.5*0.7*20000000*np.log2(1+np.power(10,sinr/10))/8000 #division by 8000 to determine number of bits that can be transmitted per TTI (1ms)
        self.cp2=0.5*0.7*20000000*np.log2(1+np.power(10,sinr2/10))/8000
        self.cell1=cell1
        self.cell2=cell2
        self.mR=0.1 #mittlere Rate
        self.queue=simpy.Container(env)
        self.mon={}
        self.metric=self.sinr+self.queue.level
        
    def rt_user(self,env):
        on_off=1
        counter=0
        while True:
            if(on_off==1 and counter<3000):
                self.queue.put(160) #20 bytes
                mon= monitor(self.queue,self.mon,env)
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
            
    def user_packets(self,env):
        while True:
            #print('o-user')
            self.queue.put(2000)
            yield env.timeout(poisson.rvs(500, 1))
    
