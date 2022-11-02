import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

SCHEDULE_T=1 #Clock des Schedulers 

#function to monitor the level of the different queues
def monitor(queue,monitor): 
    monitor.update({env.now: queue.level})
    return monitor

#scheduler takes packets from the queues according to the capacity of each user
def scheduler(env, users, SCHEDULE_T):
    
    counter=1 #counts the number of scheduling procedures
    alpha=-np.log10(0.01)/100
    
    while True: #größte Warteschlange wird auch bedient
        yield env.timeout(SCHEDULE_T) #for each ms the scheduling is active -> per TTI
        metric=np.array([]) 
        print(env.now)
        users[0].mon= monitor(users[0].queue,users[0].mon)
        users[1].mon= monitor(users[1].queue,users[1].mon)
        users[2].mon= monitor(users[2].queue,users[2].mon)
        
        for i in np.arange(np.size(users)):
                  
            if(users[i].qos==1):
                metric=np.append(metric, (alpha*users[i].queue.level*(users[i].cp/users[i].mR)))  #list the metric of all UEs in the process 
                users[i].mR=(1-1/counter)*users[i].mR #Ratenanpassung für alle Nutzer
                
            elif(users[i].qos==0):
                metric=np.append(metric,(users[i].cp/users[i].mR))
                users[i].mR=(1-1/counter)*users[i].mR #Ratenanpassung für alle Nutzer
                
        
        sched_user_list = (-metric).argsort()[:5] # wählt die 5 UEs mit größter Metrik aus #find UE with maximal metric that will be used for scheduling
        
        print(sched_user_list)    
        print(users[2].queue.level)
        
        for k in [0,1,2,3,4,5,6,7]:
            sched_user=sched_user_list[k]
            sched_size=(round(users[sched_user].cp/8)) #scheduling size according to the channel capacity of the user -> user gets full bandwidth for 1ms (1TTI)
            #users[sched_user].mon= monitor(users[sched_user].queue,users[sched_user].mon)
            users[sched_user].mR=users[sched_user].mR+(1/counter)*sched_size
            users[sched_user].queue.get(sched_size)
        counter=counter+1
        #print(users[2].mR)
        #print(users[2].cp)
        #print(env.now)

        
            
class ue:
    def __init__(self,sinr,sinr2,cell1,cell2,env,qos):
        self.sinr=sinr
        self.sinr2=sinr2
        self.qos=qos
        self.cp=0.7*20000000*np.log2(1+np.power(10,sinr/10))/8000 #division by 8000 to determine number of bits that can be transmitted per TTI (1ms)
        self.cp2=0.7*20000000*np.log2(1+np.power(10,sinr2/10))/8000
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
                mon= monitor(self.queue,self.mon)
                #yield env.timeout(poisson.rvs(6, 1))
                #print('On Phase')
                #print(self.queue.level)
                counter=counter+20
                #print(counter)
                yield env.timeout(20) #every 20ms new packet
            elif(on_off==0):
                on_off=1
                #print('Off Phase')
                #print(self.queue.level)
                yield env.timeout(3000) #3s no packet to be sent
            elif(on_off==1 and counter>=3000):
                on_off=0
                counter=0
                #print('change from ON-OFF')
            
    def user_packets(self,env):
        while True:
            print('normal ue')
            #print('o-user')
            self.queue.put(2000)
            #self.mon= monitor(self.queue,self.mon)
            yield env.timeout(poisson.rvs(500, 1))
    