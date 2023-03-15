import eventDrivenSimulation_Copy1 as eds
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


def run_simulation(mode,df,cluster,mu, ue_nr, thr, max_prb,prb_number_comp,metric,sched_mode):
    env=simpy.Environment()
    ue_dict=eds.df_to_ue_lists(df,cluster,thr,env)
    index=eds.select_user_index(mode, ue_nr, ue_dict, cluster)

    ue_per_tp,ue_all=eds.get_user_from_cluster(ue_dict,cluster,ue_nr,index)
    if(prb_number_comp=='calculate'):
        prb_number_comp=eds.calculate_prb_number_mode2(ue_all,cluster,max_prb,ue_nr)
    
    
    add_resources={}
    sched_l=[]
    sched_central=eds.sched_inst(env,cluster) #central scheduler
    for i in cluster:
        sched_l.append(eds.sched_inst(env,cluster))
        add_resources[str(int(i))+'to central']=0
        add_resources[str(int(i))+'from central']=0

    SCHEDULE_T=2

    ue_noCoMP,ue_comp=eds.seperate_comp_noCoMP(cluster,ue_per_tp)

    for j in ue_all:
        env.process(j.best_effort_stat(env,mu))

    env.process(sched_central.central_scheduler(env,ue_comp,SCHEDULE_T,cluster,prb_number_comp,metric,'phaseshift',sched_mode))

    counter=0
    for i in cluster:
        ue_list=ue_noCoMP[i]
        ue_sep=ue_all[counter*ue_nr:((counter+1)*ue_nr)]
        prb_number_normal=max_prb-prb_number_comp[i]
        env.process(sched_l[counter].scheduler(env,ue_sep,SCHEDULE_T,cluster,max_prb,ue_list,prb_number_normal,metric,sched_mode, sched_central))
        counter=counter+1
    timer=2000
    env.run(until=timer)
    return ue_noCoMP, ue_comp, ue_all,index




def create_sector_shape(lon, lat, dir=0, width=120):
    p = [(lat, lon)]
    n_points = 10
    
    for a in range(n_points):
        p.append(haversine.inverse_haversine(p[0], 0.05, (dir - width/2 + width/n_points*a)/180.0 * math.pi))
    
    p.append(p[0])
    return p

def plot_map_cluster(CONFIG,cell_data,df_r,df_r2):
    ul_scenario_map = folium.Map(location = [CONFIG['LAT'], CONFIG['LON']], tiles = "cartodbpositron", zoom_start = 15)
    folium.Circle(radius = CONFIG['RADIUS'], 
                  location = (CONFIG['LAT'], CONFIG['LON']), 
                  color = 'blue', 
                  fill_color = 'blue',
                  fill_opacity = 0.1,
                  fill = True,
                  weight = 0,
                 ).add_to(ul_scenario_map)            

    for cell in cell_data:
        if(cell['pci'] in [319,775,320,133]):
            cell_color = '#1c86ee'
        else:
            cell_color = '#888888'


        folium.PolyLine(
            create_sector_shape(cell['lon'], cell['lat'], cell['az'], 60), 
            color = cell_color,
            fill_color = cell_color,
           fill_opacity = 0.5, 
            fill = True,
            weight = 2,
            #popup = 'RBs: ' + str(cell['ul_rb_requirement']['mean']),
            tooltip = 'PCI: ' + str(cell['pci'])).add_to(ul_scenario_map)

        folium.Circle(radius = 10, 
                      location = (cell['lat'], cell['lon']), 
                      color = 'black', 
                      fill_color = 'black',
                      fill_opacity = 1,
                      fill = True,
                      weight = 0,
                      popup = cell['site_name']
                     ).add_to(ul_scenario_map)


        def plotDotGreen(point):
            folium.CircleMarker(location=[point.latitude, point.longitude],radius=1,weight=5,color='green').add_to(ul_scenario_map)
        def plotDotRed(point):
            folium.CircleMarker(location=[point.latitude, point.longitude],radius=1,weight=5,color='red').add_to(ul_scenario_map)

        df_r.apply(plotDotGreen, axis = 1)
        df_r2.apply(plotDotRed, axis = 1)

    display(ul_scenario_map)
    
    

def plot_map(CONFIG,cell_data,df_r,df_r2):

    ul_scenario_map = folium.Map(location = [CONFIG['LAT'], CONFIG['LON']], tiles = "cartodbpositron", zoom_start = 15)

    ul_query_string = CONFIG['URL'] + '/generate_scenario' + \
                                      '?lat=' + str(CONFIG['LAT']) + \
                                      '&lon=' + str(CONFIG['LON']) + \
                                      '&radius=' + str(CONFIG['RADIUS']) + \
                                      '&num_ues=' + str(CONFIG['NUM_UES']) + \
                                    '&cell_type=NGMN3600'

    ul_response_data = requests.get(ul_query_string).json()
    ue_data = ul_response_data['ue_data']
    cell_data=ul_response_data['cell_data']

    folium.Circle(radius = CONFIG['RADIUS'], 
                  location = (CONFIG['LAT'], CONFIG['LON']), 
                  color = 'blue', 
                  fill_color = 'blue',
                  fill_opacity = 0.1,
                  fill = True,
                  weight = 0,
                 ).add_to(ul_scenario_map)            

    for cell in cell_data:
        cell_color = '#1c86ee'


        folium.PolyLine(
            create_sector_shape(cell['lon'], cell['lat'], cell['az'], 60), 
            color = cell_color,
            fill_color = cell_color,
           fill_opacity = 0.5, 
            fill = True,
            weight = 2,
            #popup = 'RBs: ' + str(cell['ul_rb_requirement']['mean']),
            tooltip = 'PCI: ' + str(cell['pci'])).add_to(ul_scenario_map)

        folium.Circle(radius = 10, 
                      location = (cell['lat'], cell['lon']), 
                      color = 'black', 
                      fill_color = 'black',
                      fill_opacity = 1,
                      fill = True,
                      weight = 0,
                      popup = cell['site_name']
                     ).add_to(ul_scenario_map)


    display(ul_scenario_map)