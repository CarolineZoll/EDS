import pandas as pd
import copy
import numpy as np
import json as js
#from haversine import haversine,Unit
import csv as csv
import math as math
from DataRate import DataRate
from Objects import Cqi
from FiveGDictionary import ModulationCodingScheme as mcs
from FiveGDictionary import HeaderTags
from FiveGDictionary import StandardizedValues
from FiveGDictionary import ScalingFactor
from operator import attrgetter
from associator import ObjectMatcher
from interpreter import BandConverter
from criterion import AtomicValidator
import datetime
from math import radians, cos, sin, asin, sqrt
class DataSet:

    def __init__(self, file_path: str,is_network_data):
        self.file_path = file_path
        self.is_json_file = False
        self.is_excel_file = False
        self.is_csv_file = False
        #Check to see if the file is either json or excel
        if ".json" in file_path:
            self.data_frame = self.__load_json_file()
            self.is_json_file = True
        elif ".xlsx" in file_path:
            if is_network_data:
                self.data_frame = self.__load_excel_file()
                self.is_excel_file = True
                self.standardize_complete_dataframe(self.data_frame)#standardize the dataframes for later easy usage
            else:
                self.data_frame = self.__load_excel_file()
                self.is_excel_file = True
        elif ".csv" in file_path:
            self.data_frame = pd.read_csv(file_path,index_col=False)
            self.is_csv_file = True
            self.standardize_complete_dataframe(self.data_frame)#standardize the dataframes for later easy usage

    def __load_excel_file(self):
        return pd.read_excel(self.file_path)

    def __load_json_file(self):
        return js.load(open(self.file_path))

    def __load_csv_file(self):
        return pd.read_csv(self.file_path)

    def add_data_frame(self,data_frame: pd.DataFrame):
        self.data_frame = pd.concat([self.data_frame,data_frame],ignore_index=True)

    @staticmethod
    def add_data_frame_static(data_frame:pd.DataFrame,data_frame2:pd.DataFrame):
        return pd.concat([data_frame,data_frame2],ignore_index=True)

    def convert_cellinfo_json_to_dataframe(self):
        result = pd.DataFrame()
        for entry in self.data_frame['hits']['hits']:
            dict = entry['_source']
            pci = dict.get('pci_cell')
            location = dict.get('koordinaten_ant')
            location = location.split(",")
            location = np.array(location)
            location = location.astype(np.float)
            region = dict.get('Region')
            data = {'pci':pci,'cell_lat':[location[0]],'cell_lon':[location[1]],'region':region}
            temp_frame = pd.DataFrame(data)
            result = self.add_data_frame_static(result, temp_frame)
        self.data_frame = result

    def convert_harvester_json_to_dataframe(self):
        result = pd.DataFrame()
        model = self.data_frame['DeviceInfo']['MODEL']
        for entry in self.data_frame['Entries']:
            timestamp = float(entry['UnixTime'])/1000
            time = datetime.datetime.utcfromtimestamp(timestamp).replace(tzinfo=datetime.timezone.utc)
            #latency = entry['latency']
            for rate in entry['rate']:
                rxBytes = rate['rxBytesTotal']
                txBytes = rate['txBytesTotal']

            for location in entry['location']:

                if 'locLatitude' in location:lat = location['locLatitude']
                else:lat = np.nan

                if 'locLongitude' in location: lon = location['locLongitude']
                else:lon = np.nan

            for cellinfo in entry['cellinfo']:

                if 'type' in cellinfo: type = cellinfo['type']
                else: type = ""

                if 'isRegistered' in cellinfo:isregistered = cellinfo['isRegistered']
                else:isregistered = ""

                if 'MeasurementsID' in cellinfo: MeasurementsID = float(cellinfo['MeasurementsID'])
                else:MeasurementsID = np.nan

                if 'alphaLong' in cellinfo: alphaLong = cellinfo['alphaLong']
                else: alphaLong = ""

                if 'level' in cellinfo: level = float(cellinfo['level'])
                else: level = np.nan

                if 'timingAdvance' in cellinfo: timingAdvance = float(cellinfo['timingAdvance'])
                else: timingAdvance = np.nan

                if 'cqi' in cellinfo:cqi = float(cellinfo['cqi'])
                else:cqi = np.nan

                if 'bandwidth' in cellinfo:bandwidth = float(cellinfo['bandwidth'])
                else:bandwidth = np.nan

                if 'pci' in cellinfo:pci = float(cellinfo['pci'])
                else:pci = np.nan

                if 'band' in cellinfo:band = cellinfo['bands']
                else:band = np.nan

                if 'rssi' in cellinfo:rssi = float(cellinfo['rssi'])
                else:rssi = np.nan

                if 'rssnr' in cellinfo:rssnr = float(cellinfo['rssnr'])
                else:rssnr = np.nan

                if 'rsrp' in cellinfo:rsrp = float(cellinfo['rsrp'])
                else:rsrp = np.nan

                if 'rsrq' in cellinfo:rsrq = float(cellinfo['rsrq'])
                else:rsrq = np.nan

                data = {'model': model,'time':time, 'type': type, 'rsrp':[rsrp], 'rsrq':[rsrq], 'rssi':[rssi], 'rssnr':[rssnr],'cqi':[cqi] ,'band':[band], 'level':[level],'bandwidth':[bandwidth],'alphalong':alphaLong,'pci':[pci],'timingAdvance':[timingAdvance],'isregistered':isregistered
                    ,'MeasurementID':[MeasurementsID],'txBytes': [txBytes],'rxBytes':[rxBytes], 'lat': lat, 'lon':lon}
                temp_frame = pd.DataFrame(data)
                result = self.add_data_frame_static(result,temp_frame)
            self.data_frame = result

    def standardize_complete_dataframe(self,data_frame : pd.DataFrame):
        self.standardize_data_frame_head(data_frame)# set column tags as None if they aren't contained in the standardized dict
        self.remove_column(data_frame, "None")#remove all values not contained
        column_is_atomic = AtomicValidator(data_frame['band'].iloc[0],'band').is_atomic()# checks if the band column is atomic, this can be expanded for other cells as well
        if not column_is_atomic:
            #get all unique column values
            unique_values = data_frame['band'].unique()
            for value in unique_values:
                data_frame['band'].replace(value,StandardizedValues.band_values_dict[value],inplace = True)# replace all unique values with the standardized dictionary values

        data_frame['time'] = pd.to_datetime(data_frame['time'])
        data_frame['time'] = data_frame['time'].dt.strftime('%m/%d/%Y, %H:%M:%S')

    def get_atomic_value(self,value):
        return BandConverter(value).get_atomic_value()

    def replace_column_values(self,data_frame:pd.DataFrame,column,index,value):
        data_frame[column].iloc[index] = value

    @staticmethod
    def get_min_distance_cell_frame(user_lat,user_lon,data_frame:pd.DataFrame):
        minDistance = np.inf
        minIndex = np.NaN
        for i in range(data_frame.shape[0]):
            cell_lat,cell_lon = data_frame['cell_lat'].iloc[i],data_frame['cell_lon'].iloc[i]
            dist = DataSet.haversine(user_lat,user_lon, cell_lat,cell_lon)
            if(dist<minDistance):
                minDistance = dist
                minIndex = i
        result = data_frame.iloc[minIndex]
        pci = result['pci']
        cell_lat = result['cell_lat']
        cell_lon = result['cell_lon']
        result = pd.DataFrame({'pci': [pci],'cell_lat':[cell_lat],'cell_lon':[cell_lon],'distance':[minDistance]})
        return result

    #gives the cell with the
    @staticmethod
    def add_distance_column(data_frame: pd.DataFrame):
        distance = []
        for i in range(data_frame.shape[0]):
            user = (data_frame['lat'].iloc[i],data_frame['lon'].iloc[i])
            cell = (data_frame['cell_lat'].iloc[i],data_frame['cell_lon'].iloc[i])
            distance.append(DataSet.haversine(user,cell))
        data_frame['distance'] = distance
        return data_frame

    @staticmethod
    def add_data_rate_column(data_frame: pd.DataFrame,tag,name):
        column = data_frame[tag]
        timeseries = data_frame['time']
        data_frame['time'] = pd.to_datetime(data_frame['time'])
        data_rate = []
        time_dif = None
        for i in range(data_frame.shape[0]-1):
            t1 = data_frame['time'].iloc[i]
            t2 = data_frame['time'].iloc[i+1]
            bytes = (data_frame[tag].iloc[i+1] - data_frame[tag].iloc[i])/1000000
            time_dif = t2-t1
            time_dif = abs(time_dif.total_seconds())
            if time_dif != 0:
                rate = bytes/time_dif
            else:
                rate = 0
            data_rate.append(rate)
        last_rate = data_frame[tag].iloc[-1]/time_dif
        data_rate.append(last_rate)
        data_frame[name] = data_rate
        return data_frame

    #this method takes dataframe with values and established the confidence intervalls as a seperate dataframe column section
    def add_to_confidence_interval_data_frame(self,data_frame: pd.DataFrame,confidence_frame,tag,z_score):
        avg = np.average(data_frame[tag].to_numpy())
        std = np.std(data_frame[tag].to_numpy())
        n = data_frame.shape[0]
        intervalls = [avg - (z_score * (std / math.sqrt(n))), avg + (z_score * (std / math.sqrt(n)))]
        cqi = data_frame['cqi'].unique()
        band = data_frame['band'].unique()
        data = {'cqi': cqi, 'band': band, 'lower_conf_int': [intervalls[0]], 'upper_conf_int': [intervalls[1]]}
        temp_frame = pd.DataFrame(data)

        if confidence_frame is None:
            return temp_frame
        else:
            return self.add_data_frame_static(confidence_frame,temp_frame)

    def standardize_data_frame_head(self,data_frame: pd.DataFrame):
        curr_header = data_frame.head()
        transformed_header = []
        for tag in curr_header:
            standardized_tag = HeaderTags(tag).get_standardized_tag() #get the standardized version of the current tag
            print(tag)
            transformed_header.append(standardized_tag)
        self.rename_columns(data_frame,transformed_header)

    def remove_column(self,data_frame: pd.DataFrame,column: str):
        df_columns = data_frame.columns[np.where(data_frame.columns == column)]
        data_frame = data_frame.drop(columns = df_columns,inplace = True)

    #Return the header of the original DataSet
    def get_header(self):
        if self.is_excel_file:
            return self.data_frame.head()
        elif self.is_json_file:
            print("no header function for json file")
        elif self.is_csv_file:
            return self.data_frame.head()

    def rename_columns(self,data_frame_local: pd.DataFrame, tags):
        data_frame_local.columns = tags
        return data_frame_local

    #This is a functional method to only change the data types
    def set_datatype_flt(self,data_frame: pd.DataFrame,column):
        data_frame[column] = data_frame[column].astype(float)
        return data_frame

    def get_data(data_frame: pd.DataFrame, tags):
        list = data_frame.head()
        df = pd.DataFrame
        for i,elem in list:
            if(tags[i] in list):
                return null

    def get_rows_without(self,data_frame: pd.DataFrame,column:str,value:str):
        query = column+"!="+"\""+value+"\""
        return data_frame.query(query)

    def remove_nan_values(self,data_frame: pd.DataFrame,column:str):
        data_frame.dropna(subset =[column],inplace=True)
        return data_frame

    def remove_zero_values(self,data_frame: pd.DataFrame,column:str):
        df = data_frame.loc[data_frame[column] != 0]
        return df

    @staticmethod
    def get_dataframe_with_values(data_frame: pd.DataFrame,column:str,value:float):
        df = data_frame.loc[data_frame[column] == value]
        return df

    def get_dataframe_with_stringvalue(self, dataframe: pd.DataFrame, column:str , iD: str):
        df = dataframe.loc[dataframe[column] == iD]
        return df

    def get_dataframe_within_range(self, data_frame: pd.DataFrame,range_lat,range_lon):
        index = np.where((data_frame["lon"] > range_lon[0]) & (data_frame["lon"] < range_lon[1])
                         & (data_frame["lat"] > range_lat[0]) & (data_frame["lat"] < range_lat[1]))
        index = np.array(index)
        index = index.ravel()
        index = index.tolist()
        index = data_frame.index[index]
        return data_frame.iloc[index]

    def get_dataframe_with_str(self, data_frame: pd.DataFrame,column:str,key: str):
        df = data_frame.loc[data_frame[column] == key]
        return df

    def remove_anom_values(self,data_frame: pd.DataFrame,anom,anom_range):
        index = np.where((data_frame['hw_model'] == anom.name) & (data_frame['band'] == anom.band) & (data_frame['lte_cqi'] == anom.cqi) & ((data_frame['lte_rssnr'] >= anom_range[0]) & (data_frame['lte_rssnr'] <= anom_range[1])))
        index = np.array(index)
        index = index.ravel()
        index = index.tolist()
        index = data_frame.index[index]
        #create index values for dataframe
        data_frame = data_frame.drop(index)
        return data_frame

    def insert_dataframe_index(self,data_frame: pd.DataFrame):
        length = len(data_frame)
        index_list = range(length)
        index_list = list(index_list)
        data_frame['index'] = index_list
        data_frame.set_index('index')
        return data_frame

    def add_anom_colum_values(self, data_frame: pd.DataFrame, list, anom_range, column):
        #all data is first set to not_anom
        data_frame[column] = "not_anom"
        for anom in list:
            index_anom = np.where((data_frame['model'] == anom.name) & (data_frame['band'] == anom.band) & (
                        data_frame['cqi'] == anom.cqi) & ((data_frame['rssnr'] >= anom_range[0]) & (
                        data_frame['rssnr'] <= anom_range[1])))
            index_anom = np.array(index_anom)
            index_anom = index_anom.ravel()
            index_anom = index_anom.tolist()
            #The following code snippet are for anomalies samples with healthy data, to identify the healthy datapoints
            index_anom_but_healthy_lower = np.where((data_frame['model'] == anom.name) & (data_frame['band'] == anom.band) & (
                        data_frame['cqi'] == anom.cqi) & (data_frame['rssnr'] < anom_range[0]))
            index_anom_but_healthy_lower = np.array(index_anom_but_healthy_lower)
            index_anom_but_healthy_lower = index_anom_but_healthy_lower.ravel()
            index_anom_but_healthy_lower = index_anom_but_healthy_lower.tolist()

            index_anom_but_healthy_upper = np.where(
                (data_frame['model'] == anom.name) & (data_frame['band'] == anom.band) & (
                        data_frame['cqi'] == anom.cqi) & (data_frame['rssnr'] > anom_range[1]))
            index_anom_but_healthy_upper = np.array(index_anom_but_healthy_upper)
            index_anom_but_healthy_upper = index_anom_but_healthy_upper.ravel()
            index_anom_but_healthy_upper = index_anom_but_healthy_upper.tolist()

            for i in index_anom:
                data_frame[column].iloc[i] = "anom" # all anomalies are set to anomalies

            for i in index_anom_but_healthy_lower:
                data_frame[column].iloc[i] = "anom_with_healthy_data"

            for i in index_anom_but_healthy_upper:
                data_frame[column].iloc[i] = "anom_with_healthy_data"

        return data_frame

    def create_new_column(self,data_frame: pd.DataFrame,name):
        data_frame[name] = ""

    def clean_data(self, anom):
        band_name = anom.band
        cqi_lvl = anom.cqi
        hw_name = anom.name

    def sort_list(self,list,attribute):
        list.sort(key=attrgetter(attribute))

    def get_band_from_list(self,list,cqi,band):#returns none if element isn't found
        for elem in list:
            if elem.band == band and elem.cqi == cqi:
                return elem

        return None

    def get_cqi_from_list(self,list,cqi):#returns none if element isn't found
        for elem in list:
            if elem.cqi == cqi:
                return elem
        return None

    def combine_bands_to_cqi(self,list):
        res = []
        for elem in list:
            elem.was_checked = False

        ref_pos = 1

        for elem in list:

            for elem2 in list[ref_pos:]:

                if elem.cqi == elem2.cqi and not (elem2.was_checked) and not (elem.was_checked):
                    object = Cqi(elem.cqi, elem.dataset)
                    object.add_data(elem2.dataset)
                    elem.was_checked = True
                    elem2.was_checked = True
                    res.append(object)

                if elem.cqi == elem2.cqi and not(elem2.was_checked):
                    object = self.get_cqi_from_list(res,elem.cqi)
                    object.add_data(elem2.dataset)
                    elem2.was_checked = True
            ref_pos+=1
        return res

    def get_list_with_cqi_band(self,list,cqi,band):
        res = []
        for elem in list:
            if elem.cqi == cqi and elem.band == band:
                res.append(elem)
        return res

    #def gather_data_from_frame(self,dataframe : pd.Dataframe,cqi,band):

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        R = 6372.8
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)
        a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
        c = 2*asin(sqrt(a))
        return R * c


class DataFrameSplitter:

    def __init__(self,data_frame,header,seperator: str):
        self.data_frame = data_frame
        self.header = header
        self.seperator = seperator


    def split_lat_and_lon(self):
        df = pd.DataFrame()
        lat = []
        lon = []
        for row in self.data_frame.iterrows():
            obj = row[1][2]
            split_obj = obj.split(",")
            lat.append(split_obj[0])
            lon.append(split_obj[1])

        df['latitude BS'] = lat
        df['longitude BS'] = lon

        self.data_frame.drop(self.header,axis=1,inplace=True)
        self.data_frame = pd.concat([self.data_frame,df],axis=1)

        return self.data_frame

    def split_lat_and_lon_second(self):
        df = pd.DataFrame()
        lat = []
        lon = []
        for row in self.data_frame.iterrows():
            obj = row[1][0]
            split_obj = obj.split(",")
            lat.append(split_obj[0])
            lon.append(split_obj[1])

        df['latitude BS'] = lat
        df['longitude BS'] = lon

        self.data_frame.drop(self.header,axis=1,inplace=True)
        self.data_frame = pd.concat([self.data_frame,df],axis=1)

        return self.data_frame


