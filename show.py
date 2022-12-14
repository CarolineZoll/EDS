import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

##################################################################################################
#Darstellungen
##################################################################################################
def cdf_1(data,legende1,color1,titel,xbereich1,xbereich2,xachse,bins):

    count, bins_count = np.histogram(data, bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    # plotting PDF and CDF
    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(6)
    
    parameters = {'axes.labelsize': 16,'axes.titlesize': 16,'xtick.labelsize':13,'ytick.labelsize':13,'figure.titlesize':20}
    plt.rcParams.update(parameters)
    
    plt.plot(bins_count[1:], cdf, color=color1, label=legende1)
    plt.title(titel)
    plt.legend(fontsize=16)
    plt.grid()
    plt.ylim([0,1])
    plt.xlim([xbereich1,xbereich2])
    plt.xlabel(xachse)
    plt.ylabel('CDF')
    print('10% Quantil:', np.percentile(data, 10)) 
    print('90% Quantil:', np.percentile(data, 90)) 
    print('50% Quantil (Median):', np.percentile(data, 50)) 
######################################################################################################    
def cdf_2(data,legende1,color1,data2,legende2,color2,titel,xbereich1,xbereich2,xachse,bins):


    count, bins_count = np.histogram(data, bins)
    count2, bins_count2 = np.histogram(data2, bins)
    pdf = count / sum(count)
    pdf2 = count2 / sum(count2)
    cdf = np.cumsum(pdf)
    cdf2 = np.cumsum(pdf2)


    # plotting PDF and CDF
    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(6)

    parameters = {'axes.labelsize': 16,'axes.titlesize': 16,'xtick.labelsize':13,'ytick.labelsize':13,'figure.titlesize':20}
    plt.rcParams.update(parameters)
    
    plt.plot(bins_count[1:], cdf, color=color1, label=legende1)
    plt.plot(bins_count2[1:], cdf2, color=color2, label=legende2)

    plt.title(titel)
    plt.ylim([0,1])
    plt.legend(fontsize=16)
    plt.grid()
    plt.xlim([xbereich1,xbereich2])
    plt.xlabel(xachse)
    plt.ylabel('CDF')
    
    parameters = {'axes.labelsize': 25,'axes.titlesize': 35,'xtick.labelsize':30}
    plt.rcParams.update(parameters)
    print(legende1)
    print('10% Quantil:', np.percentile(data, 10)) 
    print('90% Quantil:', np.percentile(data, 90)) 
    print('50% Quantil (Median):', np.percentile(data, 50)) 
    print(legende2)
    print('10% Quantil:', np.percentile(data2, 10)) 
    print('90% Quantil:', np.percentile(data2, 90)) 
    print('50% Quantil (Median):', np.percentile(data2, 50))
######################################################################################################
def cdf_3(data,legende1,color1,data2,legende2,color2,data3,legende3,color3,titel,xbereich1,xbereich2,xachse,bins):
    
    count, bins_count = np.histogram(data, bins)
    count2, bins_count2 = np.histogram(data2, bins)
    count3, bins_count3 = np.histogram(data3, bins)
    pdf = count / sum(count)
    pdf2 = count2 / sum(count2)
    pdf3 = count3 / sum(count3)
    cdf = np.cumsum(pdf)
    cdf2 = np.cumsum(pdf2)
    cdf3 = np.cumsum(pdf3)

    # plotting PDF and CDF
    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(6)

    parameters = {'axes.labelsize': 16,'axes.titlesize': 16,'xtick.labelsize':13,'ytick.labelsize':13,'figure.titlesize':20}
    plt.rcParams.update(parameters)
    
    plt.plot(bins_count[1:], cdf, color=color1, label=legende1)
    plt.plot(bins_count2[1:], cdf2, color=color2, label=legende2)
    plt.plot(bins_count3[1:], cdf3, color=color3, label=legende3)
    
    
    plt.title(titel)
    
    plt.legend(fontsize=16)
    plt.grid()
    plt.ylim([0,1])
    plt.xlim([xbereich1,xbereich2])
    plt.xlabel(xachse)
    plt.ylabel('CDF')
    
    print(legende1)
    print('10% Quantil:', np.percentile(data, 10)) 
    print('90% Quantil:', np.percentile(data, 90)) 
    print('50% Quantil (Median):', np.percentile(data, 50)) 
    print(legende2)
    print('10% Quantil:', np.percentile(data2, 10)) 
    print('90% Quantil:', np.percentile(data2, 90)) 
    print('50% Quantil (Median):', np.percentile(data2, 50))
    print(legende3)
    print('10% Quantil:', np.percentile(data3, 10)) 
    print('90% Quantil:', np.percentile(data3, 90)) 
    print('50% Quantil (Median):', np.percentile(data3, 50))
######################################################################################################
def cdf_4(data,legende1,color1,data2,legende2,color2,data3,legende3,color3,data4, legende4,color4,titel,xbereich1,xbereich2,xachse,bins):

    count, bins_count = np.histogram(data, bins)
    count2, bins_count2 = np.histogram(data2, bins)
    count3, bins_count3 = np.histogram(data3, bins)
    count4, bins_count4 = np.histogram(data4, bins)
    pdf = count / sum(count)
    pdf2 = count2 / sum(count2)
    pdf3 = count3 / sum(count3)
    pdf4 = count4 / sum(count4)
    cdf = np.cumsum(pdf)
    cdf2 = np.cumsum(pdf2)
    cdf3 = np.cumsum(pdf3)
    cdf4 = np.cumsum(pdf4)
    
    # plotting PDF and CDF
    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(6)

    parameters = {'axes.labelsize': 16,'axes.titlesize': 16,'xtick.labelsize':13,'ytick.labelsize':13,'figure.titlesize':20}
    plt.rcParams.update(parameters)

    plt.plot(bins_count[1:], cdf, color=color1, label=legende1)
    plt.plot(bins_count2[1:], cdf2, color=color2, label=legende2)
    plt.plot(bins_count3[1:], cdf3, color=color3, label=legende3)
    plt.plot(bins_count4[1:], cdf4, color=color4, label=legende4)
    
    plt.title(titel)

    plt.legend(fontsize=16)
    plt.grid()
    plt.xlim([xbereich1,xbereich2])
    plt.ylim([0,1])
    plt.xlabel(xachse)
    plt.ylabel('CDF')  
    parameters = {'axes.labelsize': 16,'axes.titlesize': 16,'xtick.labelsize':13,'ytick.labelsize':13,'figure.titlesize':18}
    plt.rcParams.update(parameters)
    
    print(legende1)
    print('10% Quantil:', np.percentile(data, 10)) 
    print('90% Quantil:', np.percentile(data, 90)) 
    print('50% Quantil (Median):', np.percentile(data, 50)) 
    print(legende2)
    print('10% Quantil:', np.percentile(data2, 10)) 
    print('90% Quantil:', np.percentile(data2, 90)) 
    print('50% Quantil (Median):', np.percentile(data2, 50))
    print(legende3)
    print('10% Quantil:', np.percentile(data3, 10)) 
    print('90% Quantil:', np.percentile(data3, 90)) 
    print('50% Quantil (Median):', np.percentile(data3, 50)) 
    print(legende4)
    print('10% Quantil:', np.percentile(data4, 10)) 
    print('90% Quantil:', np.percentile(data4, 90)) 
    print('50% Quantil (Median):', np.percentile(data4, 50))
    
######################################################################################################
def color_plot(bs,sectors,x,y,z,val,cmap,clabel,titel,bs_color,size):

    plt.scatter(x, y, c=z, vmin=val[0], vmax=val[1], cmap=cmap, s=size)
    plt.colorbar(label=clabel)
    parameters = {'axes.labelsize': 16,'axes.titlesize': 16,'xtick.labelsize':13,'ytick.labelsize':13,'figure.titlesize':14}
    plt.rcParams.update(parameters)
    for i in np.arange(0,len(bs)):
        bs1=bs[i]
        
        plt.quiver(*bs1, 0.8660254037844387, 0.49999999999999994, color = 'grey', scale=10)
        plt.quiver(*bs1, -1.8369701987210297e-16, -1.0, color = 'grey', scale=10)
        plt.quiver(*bs1, -0.8660254037844387, 0.49999999999999994, color = 'grey', scale=10)
        #plt.quiver(bs1, 2,2*np.arctan(np.pi/6), color = 'grey')
        plt.scatter(bs1[0],bs1[1], color=bs_color)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title(titel)
    plt.clim(val[0], val[1])
    plt.show()

#########################################################################################################
def color_plot_2D(df,bs,x,y,color,titel,bs_color):

    plt.scatter(x, y, c=color, s=15)
    x_val= df['x'].values
    y_val= df['y'].values
    plt.colorbar(label='plasma')
    #plt.xlim(x_val[0],x_val[len(x_val)-1])
    #plt.ylim(y_val[0],y_val[len(y_val)-1])
    #plt.xlim(min(x_val),max(x_val))
    #plt.ylim(min(y_val),max(y_val))       
    for i in np.arange(0,len(bs)):
        bs1=bs[i]
        #plt.quiver(*bs1, 1,1*np.tan(np.pi/6), color = 'grey', scale=15)
        #plt.quiver(*bs1, -1,1*np.tan(np.pi/6), color = 'grey', scale=15)
        #plt.quiver(*bs1, 0,-1, color = 'grey', scale=15)
        plt.quiver(*bs1, 0.8660254037844387, 0.49999999999999994, color = 'grey', scale=10)
        plt.quiver(*bs1, -1.8369701987210297e-16, -1.0, color = 'grey', scale=10)
        plt.quiver(*bs1, -0.8660254037844387, 0.49999999999999994, color = 'grey', scale=10)
        #plt.quiver(bs1, 2,2*np.tan(np.pi/6), color = 'grey')
        plt.scatter(bs1[0],bs1[1], color=bs_color)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title(titel)
    plt.show()

#########################################################################################################
def auswertung_cdf(df, label1, label2, thr):
    print(label1)    
    for i in thr:
        indexNames = df[ df[label1]<i].index
        print(100*len(indexNames)/len(df),'Prozent liegen unter '+ str(i) +' dB')
    print(label2)
    for i in thr:
        indexNames = df[ df[label2]<i].index
        print(100*len(indexNames)/len(df),'Prozent liegen unter '+ str(i) +' dB')
        
##########################################################################################################