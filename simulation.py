import eventDrivenSimulation as eds
import simpy
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import showPaper as sh
import requests
import folium
import haversine
import math
import bibliothek as bib
import random


def run_simulation(mode,df,cluster,mu, ue_nr, thr, max_prb,prb_number_comp,metric,strategy,sim_time):
    env=simpy.Environment()
    ue_dict=eds.df_to_ue_lists(df,cluster,thr,env)
    index=eds.select_user_index(mode, ue_nr, ue_dict, cluster)

    ue_per_tp,ue_all=eds.get_user_from_cluster(ue_dict,cluster,ue_nr,index)
    if(prb_number_comp=='calculate'):
        if(strategy=='B'):
            prb_number_comp=eds.calculate_prb_number_strB(ue_all,cluster,max_prb,ue_nr)
        elif(strategy=='A'):
            prb_number_comp=eds.calculate_prb_number_strA(ue_all,cluster,max_prb,ue_nr)
    print(prb_number_comp)
    
    sched_l=[]
    sched_central=eds.sched_inst(env,cluster) #central scheduler
    for i in cluster:
        sched_l.append(eds.sched_inst(env,cluster))

    SCHEDULE_T=20

    ue_noCoMP,ue_comp=eds.seperate_comp_noCoMP(cluster,ue_per_tp)

    for j in ue_all:
        env.process(j.best_effort_stat(env,mu))

    env.process(sched_central.central_scheduler(env,ue_comp,SCHEDULE_T,cluster,prb_number_comp,metric,'phaseshift'))

    counter=0
    for i in cluster:
        ue_list=ue_noCoMP[i]
        ue_sep=ue_all[counter*ue_nr:((counter+1)*ue_nr)]
        prb_number_normal=max_prb-prb_number_comp[i]
        env.process(sched_l[counter].scheduler(env,ue_sep,SCHEDULE_T,cluster,max_prb,ue_list,prb_number_normal,metric,sched_central))
        counter=counter+1
    #timer=2000
    env.run(until=sim_time)
    return ue_noCoMP, ue_comp, ue_all,index
