#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sunpy
from sunpy.time import TimeRange
from sunpy.net import hek
import numpy as np
from bs4 import BeautifulSoup as  bs
import requests
import pandas as pd
import datetime as dt
import pyaudio  as pa


# In[7]:


client = hek.HEKClient()


# In[9]:


def Coronal_Holes_data_finder(time_range, n_days, days_array, HEK):
#объявляем пременные
    Flag = True #Флаг отсутсвия событий
    CH_min_angular_distance = 1000 #минимум углового расстояния
    CH_area = 0
    #активация HEK клиента и поиск по параметрам
    event_type = 'CH'
    Coronal_Holes_data = HEK.search(hek.attrs.Time(time_range.start,                                                      time_range.end),                                       hek.attrs.EventType(event_type))
    #переключение флага
    if len(Coronal_Holes_data) != 0:
#        date =  Coronal_Holes_data['SOL_standard'][0]
#        date = date[3:13]
        Flag = False

    for event in Coronal_Holes_data:    
        
        if Flag:
            break
#        current_date = event['SOL_standard'][3:13] 
#        if type(CH_area) == "int" or type(CH_area) == "float":
        if event['area_atdiskcenter'] != None:
            CH_area += event['area_atdiskcenter']
        #рассчет углового расстояния
        coord1 = np.radians(event["event_coord1"])
        coord2 = np.radians(event["event_coord2"])
        angular_distance = np.degrees(np.arccos(np.cos(coord1)*np.cos(coord2)))
        if angular_distance < CH_min_angular_distance:
            CH_min_angular_distance = float(angular_distance)
            


    return "CH_distance", [CH_min_angular_distance],            "CH_area", [CH_area]
def Sunflare_data_finder(time_range, n_days, days_array, HEK):
    
    Flag = True 
    SF_class = 0
    SF_number = 0
    SF_class_max = 0
    event_type = 'FL'
    try:
        Sunflare_data = HEK.search(hek.attrs.Time(time_range.start, time_range.end), hek.attrs.EventType("FL"))
    
        if len(Sunflare_data) != 0:
            date = Sunflare_data['SOL_standard'][0]
            date = date[3:13]
            Flag = False

        for event in Sunflare_data:
            if Flag:
                break
            
            event_class = str(event["fl_goescls"])   
            SF_number += 1
        
            if 'A' in event_class:
                SF_class = (1e-8)*float(event_class[1:])       
            elif 'B' in event_class:
                SF_class = (1e-7)*float(event_class[1:])       
            elif 'C' in event_class:
                SF_class = (1e-6)*float(event_class[1:])          
            elif 'M' in event_class:
                SF_class = (1e-5)*float(event_class[1:])       
            elif 'X' in event_class:
                SF_class = (1e-4)*float(event_class[1:])
            
            if SF_class > SF_class_max:
                SF_class_max = SF_class
                
    except UnicodeDecodeError:
        SF_class_max = 0
        SF_number = 0
        
    return "SF_class", SF_class_max,                "SF_number", SF_number

def Sunspot_Finder(time_range, n_days, HEK):
    
    Flag = True
    date = ''
    SS_min_angular_distance = 1000  
    SS_number = 0  
    SS_sum_area = 0
    
    event_type = 'SS'
    Sunspot_data = HEK.search(hek.attrs.Time(time_range.start, time_range.end),                                  hek.attrs.EventType(event_type))
    if len(Sunspot_data) != 0:
        date = Sunspot_data['SOL_standard'][0]
        date = date[3:13]
        Flag = False
        
    for event in Sunspot_data:     
        if Flag:
            break

            
        SS_number += 1
        SS_sum_area += abs(event['area_atdiskcenter']) if event['area_atdiskcenter']!= None else 0 
        coord1 = np.radians(event["event_coord1"])
        coord2 = np.radians(event["event_coord2"])
        SS_angular_distance = np.degrees(np.arccos(np.cos(coord1)*np.cos(coord2)))
        if SS_angular_distance < SS_min_angular_distance:
            SS_min_angular_distance = SS_angular_distance  
            
    return "SS_distance", SS_min_angular_distance,            "SS_number", SS_number,            "SS_areas", SS_sum_area

def CME_Finder(time_range, n_days, HEK):
    
    Flag  = True
    date = ''
    CME_max_angular_width = 0 
    CME_max_velocity = 0
    CME_number = 0
    
    event_type = 'CE'
    CME_data = HEK.search(hek.attrs.Time(time_range.start, time_range.end),                                  hek.attrs.EventType(event_type))
    if len(CME_data) != 0:
        date = CME_data['event_starttime'][0]
        date = date[:10]
        Flag = False
        
    for event in CME_data:     
        if Flag:
            break

        CME_velocity = event['cme_radiallinvelmax'] 
        if CME_velocity > CME_max_velocity:
            CME_max_velocity = CME_velocity
        
        CME_angular_width = event['cme_angularwidth']
        if CME_angular_width > CME_max_angular_width:
            CME_max_angular_width = CME_angular_width
            
        CME_number += 1
    return "CME_max_angular_width", [CME_max_angular_width],            "CME_max_velocity", [CME_max_velocity],            "CME_number", [CME_number]

def Geostorm_Finder(input_date, days_array):
    max_geostorm_class = 0
    date = str(input_date)[:10]
    test_full_flag = False
    year, month, day = date.split('-')
    
    html = 'https://tesis.lebedev.ru/magnetic_storms.html?m='+    str(int(month))+'&d='+str(int(day))+'&y='+str(int(year))
    page = requests.get(html)
    page.encoding ='cp1251'
    
    soup = bs(page.text, 'html.parser')
    info = soup.find('b', text='Магнитная буря')
    if info != None:
        info = soup.findChildren('ul', attrs={"class":"buri24h"})
        info = str(info[0].get_text())
        info = list(k for k in info.split('\n') if k!='')
        for j in range(len(info)):
            gm_class = int(info[j][23:24])
            if gm_class > max_geostorm_class:
                max_geostorm_class = gm_class
#    max_geostorm_class_array[k] = max_geostorm_class 
                
                    
    return "GS_class", max_geostorm_class,            "Day", int(day),            "Month", int(month),            "Year", int(year)

def Dict_Generator(keys, values):
    data_dict = {type_of_data: data_values for type_of_data, data_values in zip(keys, values)}
    dataframe =pd.DataFrame.from_dict(data_dict)
    return dataframe


# In[30]:


def DataUpdate():
    Data = pd.read_csv("dataset13-v3.csv", sep = ';', engine='python')
    tstart = str(int(Data["Year"][len(Data)-1]))+'/'+    str(int(Data["Month"][len(Data)-1]))+'/'+    str(int(Data["Day"][len(Data)-1]))+' 23:59:59.000'
    tend = dt.datetime.now()
    print(tend)
    if tend.day != int(tstart[8:10]):
        time_range = TimeRange((tstart, tend))
        n_days = int(round(time_range.days.value))
        days_array = np.array(list(str(i.start).split('T')[0]                               for i in time_range.split(n_days)))
        year, month, day = tstart[:10].split('/')

        for m in time_range.split(n_days):
            time_range1 = TimeRange((m.start, m.end))
    
            print(str(m.start)[:10]+' / '+str(tend), end='')
            print('\r', end ='')
            try:
                key1_1, value1_1, key1_2, value1_2 = Sunflare_data_finder(time_range1, n_days, days_array, client)
                key2_1, value2_1, key2_2, value2_2 = Coronal_Holes_data_finder(time_range1, n_days, days_array, client)
                key3_1, value3_1, key3_2, value3_2, key3_3, value3_3 = Sunspot_Finder(time_range1, n_days, client)
                key5_1, value5_1, key5_2, value5_2, key5_3, value5_3 = CME_Finder(time_range1, n_days, client)
            except:
                key1_1, value1_1, key1_2, value1_2 = Sunflare_data_finder(time_range1, n_days, days_array, client)
                key2_1, value2_1, key2_2, value2_2 = Coronal_Holes_data_finder(time_range1, n_days, days_array, client)
                key3_1, value3_1, key3_2, value3_2, key3_3, value3_3 = Sunspot_Finder(time_range1, n_days, client)
            key5_1, value5_1, key5_2, value5_2, key5_3, value5_3 = CME_Finder(time_range1, n_days, client)
        


            key4_1, value4_1, key4_2,        value4_2, key4_3, value4_3,        key4_4, value4_4 = Geostorm_Finder(m.end, days_array)
            data = Dict_Generator([key4_1, key1_1, key1_2, key2_1,                               key2_2, key3_1, key3_2, key3_3,                               key4_2, key4_3, key4_4, key5_1, key5_2, key5_3],                      
                              [value4_1, value1_1, value1_2, value2_1,\
                               value2_2, value3_1, value3_2, value3_3,\
                               value4_2,\
                               value4_3,\
                               value4_4,\
                               value5_1,\
                               value5_2,\
                               value5_3])
            Data = Data.append(data, ignore_index=True, sort = False) 
        Data.to_csv('dataset13-v3.csv', sep=';', index = False)
    pass


# In[ ]:




