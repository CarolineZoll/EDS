import pandas as pd
import numpy as np
import random
#import DataSetLoader as D
#from FiveGDictionary import ModulationCodingScheme
#from DataSetLoader import DataSet

BB=20
noise=np.power(10,-103/10)

#to load the raytracer data from the files
def einlesen(sectors):
    CRED = '\033[91m'
    CEND = '\033[0m'
    #Einlesen der Daten 
    start=0
    data=[]
    #sectors=[18,19,20,354,355,356,492,493,494] #PCIs eintragbar
    sectors=np.sort(sectors)
    string="BS-Data/"+str(sectors[0])+" Power.txt"
    fobj = open(string) # name of the file with the according PCI that is used

    for line in fobj:
        if('END_DATA' in line):
            break
        if (start>0): # data preprocessing
            line= line.replace(" ", ";") #Ab BEGIN DATA werden Trennzeichen eingefügt
            line= line.replace("\n", "")
            line= line.replace("N.C.", "NaN") #wenn keine Messung wird ein sehr niedriger Pegel angenommen ->line.replace("N.C.", "-10000")
            line= line.split(';')
            if(line[2]== 'N.C.'):
                line[2].replace("N.C.", 'NaN')
            line[0]=float(line[0])
            line[1]=float(line[1])
            line[2]=float(line[2])
            data.append(line)
            start=start+1
        if("BEGIN_DATA" in line):
            start=1            
    fobj.close()

    string=str(sectors[0])  # Um Sektornummer hinzufügen zu können
    df = pd.DataFrame(data, columns=['x','y',string])

    del(line,data)

    for i in sectors[1:len(sectors)]: #erster Sektor wurde schon eingelesen
        start=0
        liste=[]
        kwargs={}
        fobj = open("BS-Data/"+str(i) +" Power.txt")
        for line in fobj:
            if('END_DATA' in line):
                break
            #if ( "NAME" in line):
             #   sector=line.split()
              #  sector=sector[3]
            if(start>0):
                line= line.replace(" ", ";") #Ab BEGIN DATA werden Trennzeichen eingefügt
                line= line.replace("\n", "")
                line= line.split(';')
                if(line[2]== 'N.C.'):
                    liste.append(float('NaN'))
                else:
                    liste.append(float(line[2]))
                start=start+1
            if("BEGIN_DATA" in line):
                start=1            
        fobj.close()

        kwargs = {(str(i)) : liste}
        df=df.assign(**kwargs)
        
    indexNames = df[ df[str(sectors[0])].isna()].index
    df.drop(indexNames , inplace=True)
    index= np.arange(0,len(df)) #gibt neue Indizes für das Dataframe mit den UEs
    s = pd.Series(index)
    df=df.set_index(s)
    

    if(len(df)>1000 and len(df.loc[i,:]) == (len(sectors)+2)):
        print(CRED +'Einlesen hat funktioniert' + CEND)
    else:
        print(CRED + 'Daten prüfen !' + CEND)
    return df

###########################################################################################################
#Creates dataframe for multiple defined clusters
###########################################################################################################
def get_int(df,sectors,cluster):    
    df2=df.copy()
    #Daten get in shape
 
    inter=[]
    for j in df2.index:
        x=df2.loc[j,:]
        del x['x']
        del x['y']
        x=x.sort_values(ascending=False) #finds serving cell by power
        k=x.index
        k1=k[0] #Serving cell
        comp=0
        if(type(cluster[0])==list or type(cluster[0])== np.ndarray): #if several clusters have been defined
            for z in np.arange(0,len(cluster)): #goes through the clusters
                if(k1 in cluster[z]): #if serving cell is contained in the cluster
                    k=k.drop(cluster[z]) #Delete cells from the cluster
                    inter.append(sum(np.power(10,df2.loc[j,k]/10))) #all remaining services are added (IF)
                    df2.loc[j,k]=int(-10000) #power is set to 0W
                    comp=1

        elif(type(cluster[0])== str):#if there is only on cluster
            if(k1 in cluster): # if serving cell is in the cluster
                if(len(cluster)!= len(k)): #if the cluster does not contain all cells
                    k=k.drop(cluster) 
                    inter.append(sum(np.power(10,df2.loc[j,k]/10)))
                    df2.loc[j,k]=int(-10000)
                else:
                    inter.append(0.0) #IF is set to zero -> only one cluster for all cells
                comp=1
        if(comp!=1): #if non of those cases appears
            k=k.drop(k1)
            inter.append(sum(np.power(10,df2.loc[j,k]/10)))
            df.loc[j,k]=int(-10000)         
    df2['inter']=inter
    return df2

##################################################################################################
#Fügt Daten für das Scheduling hinzu
##################################################################################################
def add_comp_data(df1,sectors,cluster,std,use_intra):
    df2=df1.copy()
    df=get_int(df2,sectors,cluster)
    if(type(cluster[0]) == str and len(cluster)==len(sectors)):
        Use_intra=True
    else:
        Use_intra=False
    
    SINR_1_list=np.array([])
    SINR_2_list=np.array([])
    SINR_3_list=np.array([])
    P_2_list=np.array([])
    P_3_list=np.array([])
    P_SC_list=np.array([])
    P_CC_list=np.array([])
  
    #für intra/inter
    SINR_2_intra_l=np.array([])#Liste für Intra-Site SINR
    SINR_2_inter_l=np.array([])#Liste für alle Inter-Site SINR
    P_2_inter_l=np.array([])
    P_2_intra_l=np.array([])
    pci_intra_l=np.array([])
    intra_exists=0
    pci_inter_l=np.array([])  
   
    if (std !=0):
        phi=[]

        if(isinstance(std, list)):
            phi=std
        else:    
                for i in np.arange(0,len(df)):
                    phi_v=np.random.normal(0,std,1)
                    phi.append(phi_v[0])
    else:
        phi=np.zeros(len(df))

    SINR_2_list_phi=[]
    SINR_2_list_phi=np.array(SINR_2_list_phi)
    SINR_2_list_phi=[]
    SINR_2_list_phi=np.array(SINR_2_list_phi)

    for i in np.arange(0, len(df)):
        
        p_r=df.loc[i,:] #goes through file line by line and selects the line
        p_r=p_r[2:2+len(sectors)] #x and y must be truncated
        p_r=np.power(10, p_r/10) # calculate the linear values for capacity calculations
        p_r=p_r.sort_values(ascending=False) # sort values by their order
        a_r=np.sqrt(p_r) #calculate the amplitude
        
        ##########################################################################
        index=p_r.index.values #Extract all index values 
        index=index.astype(np.int) #PCIen as integer
        var=np.arange(1,len(p_r))
        inter=df.loc[i,'inter']
        if(use_intra==True):
            for i2 in var: # for all pcis given
                if(abs(index[i2]-index[0])>2): #if the pci is not from one BS -> inter-site
                    SINR_2_inter=np.power((a_r[0]+a_r[i2]),2)/(sum(p_r)-(p_r[0]+p_r[i2])+noise+inter)#calculates sinr with comp 
                    P_2_inter=10*np.log10(np.power((a_r[0]+a_r[i2]),2)/p_r[0])#gives back the power gain  
                    pci_inter= index[i2]
                    break
                else:
                #elif(i==len(p_r)-1):
                    SINR_2_inter=-1000 
                    P_2_inter=-1000

            for j in var:            
                if(abs(index[j]-index[0])<=2): #if the pcis belong to one bs -> intra
                    SINR_2_intra=np.power((a_r[0]+a_r[j]),2)/(sum(p_r)-(p_r[0]+p_r[j])+noise+inter) #gives back SINR (linear)
                    P_2_intra=10*np.log10(np.power((a_r[0]+a_r[j]),2)/p_r[0]) #gibt back powergain
                    pci_intra= int(index[j])
                    intra_exists=1
                    break
                else:
                    SINR_2_intra=-1000
                    P_2_intra=-1000

            SINR_2_inter_l=np.append(SINR_2_inter_l,SINR_2_inter) #SINR  (linear)
            SINR_2_intra_l=np.append(SINR_2_intra_l,SINR_2_intra) #SINR  (linear)
            P_2_inter_l=np.append(P_2_inter_l,P_2_inter)#powergain (dB)
            if(intra_exists==1):
                P_2_intra_l=np.append(P_2_intra_l,P_2_intra)#powergain (dB)
                pci_intra_l=np.append(pci_intra_l,pci_intra)
            else:
                P_2_intra_l=np.append(P_2_intra_l,np.NaN)#powergain (dB)
                pci_intra_l=np.append(pci_intra_l,np.NaN)
            pci_inter_l=np.append(pci_inter_l,pci_inter)

            
        x2=df.loc[i,:] #goes through the file line by line and selects the line
        x2=x2.to_numpy()
        x2=x2[2:(2+len(sectors))]
        x2=-np.sort(-x2) # sorted vector with the performances descending 
        P_SC=x2[0] #power of the strongest cell
        P_CC=x2[1]#power of coordinated cell# 
        
        P_3=10*np.log10(np.power(np.sum(a_r[0:3]),2)/p_r[0]) #comp with 3 bs
        P_3_list=np.append(P_3_list,P_3)
        x2=x2/10
        x2=np.power(10, x2) # linear values
        x_A2=np.sqrt(x2) #amplitude values

        SINR_1= x2[0]/(np.sum(x2[1:len(x2)])+inter) #one bs sending -> rest interference
        SINR_2= np.power(np.sum(x_A2[0:2]),2)/(np.sum(x2[2:len(x2)])+inter+noise)
        P_2=10*np.log10(np.power(np.sum(x_A2[0:2]),2)/x2[0]) 
        SINR_3= np.power(np.sum(a_r[0:3]),2)/(np.sum(p_r[3:len(p_r)])+inter+noise)
        SINR_1_list=np.append(SINR_1_list,SINR_1) #appends the calculated elements
        SINR_2_list=np.append(SINR_2_list,SINR_2)
        SINR_3_list=np.append(SINR_3_list,SINR_3)
        
        P_SC_list=np.append(P_SC_list,P_SC)
        #including phi
        SINR_2_phi=(np.power(x_A2[0]+x_A2[1]*np.cos(phi[i]),2)+np.power(x_A2[1]*np.sin(phi[i]),2))/(np.sum(x2[2:len(x2)])+inter+noise)
        SINR_2_p_phi=10*np.log10((np.power(x_A2[0]+x_A2[1]*np.cos(phi[i]),2)+np.power(x_A2[1]*np.sin(phi[i]),2))/x2[0])
        SINR_2_list_phi=np.append(SINR_2_list_phi,SINR_2_phi)
        P_2_list=np.append(P_2_list,P_2)    
        
   
    
    kwargs = {'JT_1 SINR [lin]' : SINR_1_list} #hinzufügen in Pandas Datei durch Argumente als Dictonary
    df=df.assign(**kwargs)    

    kwargs = {'JT_2 SINR [lin]' : SINR_2_list}
    df=df.assign(**kwargs)

    kwargs = {'JT_1 C [MHz]' : BB* np.log2(SINR_1_list+1)} #hinzufügen der Kapazitäten für spätere Berechnungen
    df=df.assign(**kwargs)
    

    kwargs = {'JT_2 C [MHz]' : BB* np.log2(SINR_2_list+1)}
    df=df.assign(**kwargs)

    kwargs = {'JT_2 C [MHz] 2' : BB* np.log2(SINR_2_list+1)}
    df=df.assign(**kwargs)

    kwargs = {'P_SC [dBm]' : P_SC_list}
    df=df.assign(**kwargs)
    
    
    kwargs = {'JT_2 gain [dB] - ph' : (10*np.log10(SINR_2_list_phi/SINR_1_list))}
    df=df.assign(**kwargs)
    
    
    kwargs = {'JT_2 C [MHz] 2 - ph' : BB* np.log2(SINR_2_list_phi+1)}
    df=df.assign(**kwargs)
    
    kwargs = {'JT_2 P gain [dB]' : P_2_list}
    df=df.assign(**kwargs)
    
    kwargs = {'JT_3 SINR [lin]' : SINR_3_list}
    df=df.assign(**kwargs)
    
    kwargs = {'JT_3 P gain [dB]' : P_3_list}
    df=df.assign(**kwargs)
    
    kwargs = {'JT_2 gain [dB]' : (10*np.log10(SINR_2_list/SINR_1_list))} # vergleich mit JT vs. ohne JT
    df=df.assign(**kwargs)

    kwargs = {'JT_3 gain [dB]' : (10*np.log10(SINR_3_list/SINR_1_list))}
    df=df.assign(**kwargs)
    
    if(use_intra==True):
        kwargs = {'SINR JT_2_intra [lin]' : SINR_2_intra_l }
        df=df.assign(**kwargs)
    
        kwargs = {'SINR JT_2_inter [lin]' : SINR_2_inter_l }
        df=df.assign(**kwargs)

        kwargs = {'JT_2 C intra' : BB*np.log2(1+SINR_2_intra_l) }
        df=df.assign(**kwargs)
    
        kwargs = {'JT_2 C inter' : BB*np.log2(1+SINR_2_inter_l) }
        df=df.assign(**kwargs)

        #kwargs = {'JT_2_inter gain [dB]' : (10*np.log10(SINR_2_inter_l/SINR_1_list))}
        #df=df.assign(**kwargs)    
  
        #kwargs = {'JT_2_intra gain [dB]' : (10*np.log10(SINR_2_intra_l/SINR_1_list))}
        #df=df.assign(**kwargs) 
        
            
        kwargs = {'JT_2_intra gain [dB]' : (10*np.log10(df['SINR JT_2_intra [lin]']/df['JT_1 SINR [lin]']))}
        df=df.assign(**kwargs)

        kwargs = {'JT_2_inter gain [dB]' : (10*np.log10(df['SINR JT_2_inter [lin]']/df['JT_1 SINR [lin]']))}
        df=df.assign(**kwargs)
        
        kwargs = {'PCI Intra' : [int(slot) for slot in pci_intra_l]}
        df=df.assign(**kwargs)
    
        kwargs = {'PCI Inter' : pci_inter_l }
        df=df.assign(**kwargs)

        
        
        kwargs = {'P gain JT_2_intra [dB]' : P_2_intra_l}
        df=df.assign(**kwargs)    
  
        kwargs = {'P gain JT_2_inter [dB]' : P_2_inter_l}
        df=df.assign(**kwargs) 
    
    indexNames = df[ df[str(sectors[0])].isna()].index
    df.drop(indexNames , inplace=True)

    return df
###################################################################################################
def add_serv_coord(df,sectors):
    serving=[]
    coord=[]
    for i in np.arange(0,len(df)):
        x=df.loc[i,:] #geht Datei zeilenweise durch und selektiert die Zeile
        x=x[2:(2+len(sectors))] #x und y muss abgeschnitten werden
        x=x.sort_values(ascending=False) # Werte nach Größe sortieren
        y=x.keys()
        serving.append(y[0])
        coord.append(y[1])

    kwargs = {'PCI Serving' : (serving)}
    df=df.assign(**kwargs)

    kwargs = {'PCI Coord' : (coord)}
    df=df.assign(**kwargs)

    df['PCI Serving']=df['PCI Serving'].values.astype(np.int)
    df['PCI Coord']=df['PCI Coord'].values.astype(np.int)
    return df

####################################################################################################
def create_comp_from_Linktaster(name, boolean,phaseshift):
    df= DataSet('BS-Data/'+name +'.json',boolean)
    df.convert_harvester_json_to_dataframe()
    df= df.data_frame
    
    df=df.dropna(subset=["pci"])
    
    time=df["time"].unique()
    pci=list(map(int,df["pci"].unique()))
    if 0 in pci:
        pci.remove(0)
    df_comp=pd.DataFrame()

    x_list=np.zeros(len(time))
    y_list=np.zeros(len(time))

    pci_rsrp=np.zeros([len(time),int(len(pci))])


    df2=df.groupby('time')
    counter2=-1
    for i in time:
        df3= df2.get_group(i)

        x=list(df3['pci'])
        x=list(map(int,x))
        counter1=0
        counter2=counter2+1
        x_list[counter2]= df3.loc[df3.index[0],'lat']
        y_list[counter2]= df3.loc[df3.index[0],'lon']                       

        for j in pci:
            if j in x:
                ind=df3.index[x.index(j)]
                pci_rsrp[counter2,counter1]=df3.loc[ind, 'rsrp']
            else:
                pci_rsrp[counter2,counter1]=-100000
            counter1=counter1+1

    df_comp['x']=x_list
    df_comp['y']=y_list

    for i in np.arange(0,len(pci)):
        string=str(pci[i])
        df_comp[string]=pci_rsrp[:,i]

        
    file= open('Generated Data/version.txt','r')
    version=int(file.read())
    version=version+1
    version=str(version)
    
    file= open('Generated Data/version.txt','w')
    file.write(version)
    file.truncate()
    file.close()

    df_comp.to_csv(r'Generated Data/'+name+'scheduling-db-'+version, index=False)
    
    cluster=list(df_comp.columns.values)
    cluster.remove('x')
    cluster.remove('y')

    df1=add_comp_data(df_comp,cluster,cluster,phaseshift,False)
    df1=add_serv_coord(df1,cluster)
    
    df1.to_csv(r'Generated Data/'+name+'scheduling-db-extension-'+version, index=False)
    
    return df1