# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:29:42 2021

@author: Abdi Mohamed 
"""

import os.path
import numpy as np
import pandas as pd

import glob

from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator 
from zipfile import ZipFile
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
import datetime
import os
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight') 
# Above is a special style template for matplotlib, highly useful for visualizing time series data
from tqdm import tqdm
from scipy.optimize import curve_fit
''' setting the global parameters of matplotlib to academic puplication quality settings ''' 
from sklearn.pipeline import Pipeline







def instaSystem_auswertung(path_ISA_csv,path_sim_data):
    
    
    """ 
    Loading of Data for the Instantaneous-System of Domestic water heating.
    the code in the function was written orginally by Hagen Brass and was implemented
    through this function by Abdi Mohamed.  
    Parameters:
    path_ISA_csv: the path for the ISA_csv file that defines the column names
    path_sim_data: the path for the simulation data that should be imported or loaded
    
    Return type:return an array of Dataframes, each dataframe is a TRNSYS Simulation run. 
 

    """
    
    ISA = pd.read_csv(path_ISA_csv, sep=';')
    building_map = dict(zip(ISA.iloc[: , 0],ISA.iloc[:,1]))
    path = path_sim_data

    sns.set()
    sns.set_style('ticks')
    sns.set_context('paper', font_scale=1.8)
    startdate = pd.datetime(1970, 1, 1, 0, 0)
    enddate = pd.datetime(1971, 1, 1, 0, 0)
    tl = pd.date_range(startdate,enddate, freq='3min')

    date_formatter = mdates.DateFormatter('%d.%m. %H:%M')


    # Konstanten #
    # Gelten für ca. 50 °C
    rho = 985 # kg/m³
    cp = 4181 / 3600 # Wh/(kg K)
    ts = 3 # Zeitschritt in Minuten
    # Konstanten #

    VHQ_ar = np.array([])
    T_RL_ar = np.array([])
    bal = []

    sim_files = glob.glob( path_sim_data + '\\'   + '*.outr.zip')
    buildings = []
    iterator = np.arange(len(sim_files))
    for i in tqdm(iterator) :
        #print('lade Daten ...')
        building_name = sim_files[i].replace(path_sim_data + '\\'  ,'')
        
        with ZipFile(sim_files[i], 'r') as file:
            fileinzip = file.namelist()[0]
            with file.open(fileinzip, 'r') as fh:
                df = pd.read_csv(fh, sep='\t', )
        df = df.set_index(tl)
        df['building_name'] = building_name
        buildings.append(df)
        #print('... Daten geladen')
        df['dQ_ph'] = df['dm_DHW'] * cp *  (df['T_ph_cold_out'] - df['T_ph_cold_in']) / 1000 # Wärmestrom über Vorwärmer in kW
        df['dQ_ah'] = (df['dm_DHW']+df['dm_circ']) * cp *  (df['T_ah_cold_out'] - df['T_ah_cold_in']) / 1000 # Wärmestrom über Nachwärmer in kW
        df['dQ_circ'] = df['dm_circ'] * cp *  (df['T_ah_cold_out'] - df['T_pipe_circ_out']) / 1000 # Wärmestrom an Zirkulation in kW
        df['dQ_pipe_circ_loss'] = df['dQ_pipe_circ_loss'] / 3600 # Verlustwärmestrom der Zirkulationsleitung in kW
        df['dQ_pipe_supply_loss'] = df['dQ_pipe_supply_loss'] / 3600 # Verlustwärmestrom der Versorgungsleitung in kW
        df['dQ_pipe_flat_loss'] = df['dQ_pipe_flat_loss'] / 3600 # Verlustwärmestrom der Anbindeleitungen in kW
        df['dQ_dist_loss'] = df['dQ_pipe_supply_loss'] + df['dQ_pipe_circ_loss'] + df['dQ_pipe_flat_loss'] # Verteilverluste gesamt in kW
        df['dQ_DHW_in'] = df['dm_DHW'] * cp *  (df['T_pipe_supply_in'] - df['T_DCW']) / 1000 # Leistung von DH an DHW in kW
        df['dQ_DHW_out'] = df['dm_DHW'] * cp *  (df['T_pipe_flat_out'] - df['T_DCW']) / 1000 # Leistung von gezapftem DHW nach Verlusten in kW
        df['dQ_DH'] = df['dm_DH'] * cp * (df['T_ah_hot_in'] - df['T_ph_hot_out']) / 1000 # Abgegebene Leistung der FW in kW
    # %%   
        #print('--------------------------')
        T_RL_mean = np.sum(df['T_ph_hot_out'] * df['dm_DH']) / df['dm_DH'].sum()
        #print('Gewichtete Rücklauftemperatur in °C: \n{0:.2f}'.format(T_RL_mean))
        #print('--------------------------')
        #print('Energiebilanz Temperaturen und Volumenströme:')

        Q_FW = df['dQ_DH'].sum() * ts / 60

        Q_TWW = df['dQ_DHW_in'].sum() * ts / 60
        #print('Energie an TWW inkl. Verlusten in kWh: \n{0:.0f}'.format(Q_TWW))

        Q_TWW2 = df['dQ_DHW_out'][df['T_pipe_flat_out']>=45].sum() * ts / 60
        #print('Energie an TWW nach Verlusten in kWh: \n{0:.0f}'.format(Q_TWW2))

        Q_spill = df['dQ_DHW_out'][df['T_pipe_flat_out']<45].sum() * ts / 60

        Q_circ = df['dQ_circ'].sum() * ts / 60
        #print('Energie an Zirkulation in kWh: \n{0:.0f}'.format(Q_circ))

        Q_loss_dist_ges = df['dQ_dist_loss'].sum() * ts / 60
        #print('Rohrleitungsverluste Gesamt in kWh: \n{0:.0f}'.format(Q_loss_dist_ges))

        Q_loss_flat = df['dQ_pipe_flat_loss'].sum() * ts / 60
        #print('Rohrleitungsverluste in WE in kWh: \n{0:.0f}'.format(Q_loss_flat))

        Q_use_target = (df['dm_loadprofile'] * cp *  (45 - df['T_DCW']) / 1000 * ts / 60).sum()

        Q_use_target_night = (df['dm_loadprofile'][(df.index.hour.values>=23) | (df.index.hour.values<5)] * cp *  (45 - df['T_DCW'][(df.index.hour.values>=23) | (df.index.hour.values<5)]) / 1000 * ts / 60).sum()

        #print('--------------------------')
        V_DHW_sum = df['dm_DHW'].sum() * ts /60 / 365
        V_circ_sum = df['dm_circ'].sum()* ts /60 / 365
        VHQ = Q_circ / Q_TWW2
    return buildings


def storage_auswertung(path_ISA_csv,path_sim_data):
    
       
    """ 
    Loading of Data for the storage-System of Domestic water heating.
    the code in the function was written orginally by Hagen Brass and was implemented
    through this function by Abdi Mohamed.  
    Parameters:
    path_ISA_csv: the path for the ISA_csv file that defines the column names
    path_sim_data: the path for the simulation data that should be imported or loaded
    
    Return type:return an array of Dataframes, each dataframe is a TRNSYS Simulation run. 
 

    """

    sns.set()
    sns.set_style('ticks')
    sns.set_context('paper', font_scale=1.8)
    startdate = pd.datetime(1970, 1, 1, 0, 0)
    enddate = pd.datetime(1971, 1, 1, 0, 0)
    tl = pd.date_range(startdate,enddate, freq='3min')

    date_formatter = mdates.DateFormatter('%d.%m. %H:%M')


    # Konstanten #
    # Gelten für ca. 50 °C
    rho = 985 # kg/m³
    cp = 4181 / 3600 # Wh/(kg K)
    ts = 3 # Zeitschritt in Minuten
    # Konstanten #

    VHQ_ar = np.array([])
    T_RL_ar = np.array([])
    bal = []


    df = pd.read_csv(path_ISA_csv, sep=';')

    building_list = df['Name'].tolist()
    
    sim_files = glob.glob(path_sim_data + '\\' 'MFH_Storage_70.dck-*' '\\'  + '*.outr.zip')
    #sim_files = glob.glob( path_sim_data + '\\'   + '*.outr.zip')
    
    buildings = []
    iterator =np.arange(len(sim_files))
    for i in tqdm(iterator):
        #print('lade Daten ...')
        building_name = sim_files[i].replace(path_sim_data + '\\'  ,'')

        with ZipFile(sim_files[i], 'r') as zip:
            fileinzip = zip.namelist()[0]
            with zip.open(fileinzip, 'r') as fh:
                df = pd.read_csv(fh, sep='\t', )


        
        df = df.set_index(tl)
        df['building_name'] = building_name
        #print('... Daten geladen')

        buildings.append(df)
        df['dQ_ph'] = df['dm_stload'] * cp *  (df['T_ph_cold_out'] - df['T_ph_cold_in']) / 1000 # Wärmestrom über Vorwärmer in kW
        df['dQ_ah'] = (df['dm_stload']+df['dm_circ']) * cp *  (df['T_ah_cold_out'] - df['T_ah_cold_in']) / 1000 # Wärmestrom über Nachwärmer in kW
        df['dQ_circ'] = df['dm_circ'] * cp *  (df['T_ah_cold_out'] - df['T_pipe_circ_out']) / 1000 # Wärmestrom an Zirkulation in kW
        df['dQ_pipe_circ_loss'] = df['dQ_pipe_circ_loss'] / 3600 # Verlustwärmestrom der Zirkulationsleitung in kW
        df['dQ_pipe_supply_loss'] = df['dQ_pipe_supply_loss'] / 3600 # Verlustwärmestrom der Versorgungsleitung in kW
        df['dQ_pipe_flat_loss'] = df['dQ_pipe_flat_loss'] / 3600 # Verlustwärmestrom der Anbindeleitungen in kW
        df['dQ_dist_loss'] = df['dQ_pipe_supply_loss'] + df['dQ_pipe_circ_loss'] + df['dQ_pipe_flat_loss'] # Verteilverluste gesamt in kW
        df['dQ_sto_stload'] = df['dQ_sto_stload'] / 3600 # Aufgenommene Leistung des Speichers in kW
        df['dQ_sto_DHW'] = -df['dQ_sto_DHW'] / 3600 # Vom Speicher an TWW abgegebene Leistung in kW
        df['dQ_DHW'] = df['dm_DHW'] * cp *  (df['T_pipe_flat_out'] - df['T_DCW']) / 1000 # Leistung an DHW nach Verlusten in kW
        df['dQ_DH'] = df['dm_DH'] * cp * (df['T_ah_hot_in'] - df['T_ph_hot_out']) / 1000 # Abgegebene Leistung der FW in kW
        df['dQ_sto_loss'] = df['dQ_sto_loss'] / 3600 # Speicherverluste in kW

    # %%   

        #print('--------------------------')
        T_RL_mean = np.sum(df['T_ph_hot_out'] * df['dm_DH']) / df['dm_DH'].sum()

        T_RL_ar = np.append(T_RL_ar, T_RL_mean)

        #print('Gewichtete Rücklauftemperatur in °C: \n{0:.2f}'.format(T_RL_mean))
        #print('--------------------------')
        #print('Energiebilanz Temperaturen und Volumenströme:')

        Q_FW = df['dQ_DH'].sum() * ts / 60

        Q_TWW = df['dQ_sto_DHW'].sum() * ts / 60
        #print('Energie von Speicher an TWW in kWh: \n{0:.0f}'.format(Q_TWW))

        Q_TWW2 = df['dQ_DHW'][df['T_pipe_flat_out']>=45].sum() * ts / 60
        #print('Energie an TWW nach Verlusten in kWh: \n{0:.0f}'.format(Q_TWW2))

        Q_spill = df['dQ_DHW'][df['T_pipe_flat_out']<45].sum() * ts / 60

        Q_circ = df['dQ_circ'].sum() * ts / 60
        #print('Energie an Zirkulation in kWh: \n{0:.0f}'.format(Q_circ))

        Q_loss_dist_ges = df['dQ_dist_loss'].sum() * ts / 60
        #print('Rohrleitungsverluste Gesamt in kWh: \n{0:.0f}'.format(Q_loss_dist_ges))

        Q_loss_flat = df['dQ_pipe_flat_loss'].sum() * ts / 60
        #print('Rohrleitungsverluste in WE in kWh: \n{0:.0f}'.format(Q_loss_flat))

        Q_sto_stload = df['dQ_sto_stload'].sum() * ts / 60
        #print('Speicherbeladung in kWh: \n{0:.0f}'.format(Q_sto_stload))

        Q_sto_loss = df['dQ_sto_loss'].sum() * ts / 60 * -1
        #print('Speicherverluste in kWh: \n{0:.0f}'.format(Q_sto_loss))

        Q_use_target = (df['dm_loadprofile'] * cp *  (45 - df['T_DCW']) / 1000 * ts / 60).sum()

        Q_use_target_night = (df['dm_loadprofile'][(df.index.hour.values>=23) | (df.index.hour.values<5)] * cp *  (45 - df['T_DCW'][(df.index.hour.values>=23) | (df.index.hour.values<5)]) / 1000 * ts / 60).sum()

        #print('--------------------------')

        V_DHW_sum = df['dm_DHW'].sum() * ts /60 / 365
        V_circ_sum = df['dm_circ'].sum()* ts /60 / 365
        VHQ = Q_circ / Q_TWW2

        VHQ_ar = np.append(VHQ_ar, VHQ)

        #print('Tägliches Zapfvolumen in m³: \n{0:.2f}'.format(V_DHW_sum/1000))
        #print('Tägliches Zirkulationsvolumen in m³: \n{0:.2f}'.format(V_circ_sum/1000))


        bal.append(np.array([Q_TWW, Q_circ, Q_sto_loss, Q_TWW2, Q_loss_flat, Q_spill, Q_loss_dist_ges-Q_loss_flat, Q_use_target, Q_use_target_night]))
    return buildings 


    
def Heizung_bestand_lader(path_sim_data):
    
    """ 
    Loading of Data for the convector Heater system used for Building Heating.
    some of the code in this function was written orginally by Hagen Brass and was further 
    implemnted by Abdi Mohamed
    Parameters:
    path_sim_data: the path for the simulation data that should be imported or loaded
    
    Return type:return an array of Dataframes, each dataframe is a TRNSYS Simulation run. 
 

    """

    cp = 4181/3600
    startdate = pd.datetime(1970, 1, 1, 0, 0)
    enddate = pd.datetime(1971, 1, 1, 0, 0)
    tl = pd.date_range(startdate,enddate, freq='3min')

    date_formatter = mdates.DateFormatter('%d.%m. %H:%M')


    # Konstanten #
    # Gelten für ca. 50 °C
    rho = 985 # kg/m³
    cp = 4181 / 3600 # Wh/(kg K)
    ts = 3 # Zeitschritt in Minuten
    # Konstanten #

    sim_files = glob.glob( path_sim_data + '\\'   + '*.outr.zip')
    buildings = []
    iterator = np.arange(len(sim_files))
    for i in tqdm(iterator) :
        building_name = sim_files[i].replace(path_sim_data + '\\'  ,'')
        
        with ZipFile(sim_files[i], 'r') as file:
            fileinzip = file.namelist()[0]
            with file.open(fileinzip, 'r') as fh:
                df = pd.read_csv(fh, sep='\t', )
                
               
        df = df.set_index(tl)
        df = df.drop(['TIME'],axis = 1)
        df['building_name'] = building_name
        buildings.append(df)
    return buildings


def Heizung_bestand_lader_long(path_sim_data):
    
    '''
    the same function as Heizung_bestand_lader but with the replacment
    of the first 24 hour of the simulation with the last 24 hour of the simulation.
    that is to solve the problem of unaccurate simulation results at the first 24 hour, 
    so that these 24 hours will be simulated at the last and then replaced to be
    at the first of the year using the function. 
    
    '''

    cp = 4181/3600
    startdate = pd.datetime(1970, 1, 1, 0, 0)
    enddate = pd.datetime(1971, 1, 1, 0, 0)
    tl = pd.date_range(startdate,enddate, freq='3min')

    date_formatter = mdates.DateFormatter('%d.%m. %H:%M')


    # Konstanten #
    # Gelten für ca. 50 °C
    rho = 985 # kg/m³
    cp = 4181 / 3600 # Wh/(kg K)
    ts = 3 # Zeitschritt in Minuten
    # Konstanten #

    sim_files = glob.glob( path_sim_data + '\\'   + '*.outr.zip')
    buildings = []
    iterator = np.arange(len(sim_files))
    for i in tqdm(iterator) :
        building_name = sim_files[i].replace(path_sim_data + '\\'  ,'')
        
        with ZipFile(sim_files[i], 'r') as file:
            fileinzip = file.namelist()[0]
            with file.open(fileinzip, 'r') as fh:
                df = pd.read_csv(fh, sep='\t', )
        l = len(df)        
        place_holder = df.loc[:l-481,:]                                    #replace the frist 24 with the last 24 
        place_holder.loc[0:479,:] = np.array(df.loc[l-480:l,:])
        df = place_holder
        
        
        df = df.set_index(tl)
        df = df.drop(['TIME'],axis = 1)
        df['building_name'] = building_name
        buildings.append(df)
    return buildings




def efh_mfh_seperator(buildings):
    
    """ 
    separating the loaded data from the Heizung_bestand_lader and Heizung_neubau_lader 
    
    Parameters:
    buildings: an array of Dataframes, each dataframe is a TRNSYS Simulation run. 
    
    Return type:
        buildings_efh : single family building array of data frames
        buildings_mfh : more family building array of data frames
    
    Description : 
        seperation of the loaded data for the heating systems into two arrays
        buildings_efh, buildings_mfh 

    """
    
    cp = 4181/3600 #Wh/(kg K)
    iterator = np.arange(len(buildings))
    buildings_efh = []
    buildings_mfh = []
    for i in tqdm(iterator):
        efh = pd.DataFrame()
        mfh = pd.DataFrame()
        coulumns1 = ['dQ_DH','T_ah_hot_in','T_ph_cold_in','T_ah_cold_out','T_ph_hot_out','dm_DH','dm_DHW','dQ_DHW','T_amb']
        coulumns_efh = ['Qpr_efh','T_VL_pr_efh','T_RL_se_efh','T_VL_se_efh','T_RL_pr_efh','mdotpre_efh','mdotsek_efh','Qse_efh','T_amb']
        coulumns_mfh = ['Qpr_mfh','T_VL_pr_mfh','T_RL_se_mfh','T_VL_se_mfh','T_RL_pr_mfh','mdotpre_mfh','mdotsek_mfh','Qse_mfh','T_amb']
        efh[coulumns1] = buildings[i][coulumns_efh]
        mfh[coulumns1] = buildings[i][coulumns_mfh]
        efh['dQ_DH'] =  efh['dm_DH'] * cp * (efh['T_ah_hot_in'] - efh['T_ph_hot_out']) / 1000 ## Abgegebene Leistung der FW in kW
        mfh['dQ_DH'] =  mfh['dm_DH'] * cp * (mfh['T_ah_hot_in'] - mfh['T_ph_hot_out']) / 1000 ## Abgegebene Leistung der FW in kW
        buildings_efh.append(efh)
        buildings_mfh.append(mfh)
    return buildings_efh,buildings_mfh

def Heizung_neubau_lader(path_sim_data):
    
    """ 
    Loading of Data for the floor heating system system used for Building Heating.
    some of the code in this function was written orginally by Hagen Brass and was further 
    implemnted by Abdi Mohamed
    Parameters:
    path_sim_data: the path for the simulation data that should be imported or loaded
    
    Return type:return an array of Dataframes, each dataframe is a TRNSYS Simulation run. 
 

    """

    cp = 4181/3600
    startdate = pd.datetime(1970, 1, 1, 0, 0)
    enddate = pd.datetime(1971, 1, 1, 0, 0)
    tl = pd.date_range(startdate,enddate, freq='15min')

    date_formatter = mdates.DateFormatter('%d.%m. %H:%M')


    # Konstanten #
    # Gelten für ca. 50 °C
    rho = 985 # kg/m³
    cp = 4181 / 3600 # Wh/(kg K)
    ts = 15 # Zeitschritt in Minuten
    # Konstanten #

    sim_files = glob.glob( path_sim_data + '\\'   + '*.out')
    buildings = []
    iterator = np.arange(len(sim_files))
    for i in tqdm(iterator) :
        building_name = sim_files[i].replace(path_sim_data + '\\'  ,'')

        df = pd.read_csv(sim_files[i], sep='\t', )
                
               
        df = df.set_index(tl)
        df.columns = [c.replace(' ', '') for c in df.columns]
        df = df.drop(['TIME'],axis = 1)
        df['building_name'] = building_name
        buildings.append(df)
    return buildings


def resampler_3auf15(buildings):
    
    """ 
    resample the Data from 3 minute resolution to 15 minute. 
   
    Parameters:
    buildings: an array of Dataframes represnting simulation runs. 

    Return type:return an array of Dataframes, each dataframe is a TRNSYS Simulation run
    with the resampled resoultion.  
 
    Description:
    Resample the input dataframe from 3 minute to 15. only the primary return temperature
    (_ph_hot_out) resampled using the weights of the primary massflow (dm_DH)
    """

    building15 = []
    iterator = np.arange(len(buildings))
    for i in tqdm(iterator):
        dhw3 = buildings[i]
        dhw = dhw3.resample('15min').mean()
        dhw['T_ph_hot_out'] = (dhw3['dm_DH'] * dhw3['T_ph_hot_out']).resample('15min').sum() / dhw3['dm_DH'].resample('15min').sum()
        dhw['T_ph_hot_out'] = dhw['T_ph_hot_out'].fillna(dhw['T_ah_hot_in'])  
        building15.append(dhw)
    return building15


def resampler(df,auflösung,rücklauf,massenstrom):
    '''
    Description:
    Resample the input dataframe from 3 minute to 15. only the input with name of (rücklauf)
    will be resampled using the weights of the other input  (massenstrom) in the required resolution of (auflösung) 
    
    '''
    dhw3 = df
    dhw = dhw3.resample(auflösung).mean()
    dhw[rücklauf] = (dhw3[massenstrom] * dhw3[rücklauf]).resample(auflösung).sum() / dhw3[massenstrom].resample(auflösung).sum()
    dhw[rücklauf] = dhw[rücklauf].fillna(dhw['T_ah_hot_in'])  
    return dhw


def null_leistung(buildings,min_leistung_kw = 0.02):
    
    """ 
    filter out the points with power less less or equal to the minimum Power (min_leistung_kw) 
   
    Parameters:
    buildings: an array of Dataframes represnting simulation runs. 
    min_leistung_kw = the minimum power in kW
    
    Return type:
        mae,mse,r2,accuracy,ab
        return an array of Dataframes, each dataframe is a TRNSYS Simulation run.
    
    """
    
    building0= []
    iterator = np.arange(len(buildings))
    for i in tqdm(iterator):
        ohne_null = buildings[i].loc[(buildings[i][['dQ_DH']] >= min_leistung_kw).all(axis=1), :]
        building0.append(ohne_null)
    return building0
# model evaluation for testing set
# model evaluation for testing set


def evaluate(y_test,y_predicted):

    """ 
    calculate the measures of the fit to evaulate it goodness
   
    Parameters:
        y_test: the simulation data 
        y_predicted: the predicted date resulting from the fitt-Model
        
    Return type:
        mae : Mean absolute error 
        mse : mean square error
        r2 : r squared
        accuracy : the 95th quantile of the absolute deviation 
        ab : array of the absolute deviation 
 
    Description:
    Resample the input dataframe from 3 minute to 15. only the primary return temperature
    (_ph_hot_out) resampled using the weights of the primary massflow (dm_DH)
    """
    
    def gettype(y_test,y_predicted):
        if type(y_test) is np.ndarray:
            y_test = y_test.reshape(-1)
            y_predicted = y_predicted.reshape(-1)
            return y_test,y_predicted 
        elif type(y_predicted) is not np.ndarray:
            return y_test,y_predicted 
    y_test,y_predicted = gettype(y_test,y_predicted)   
    
    mape = metrics.mean_absolute_percentage_error(y_test, y_predicted)  
    mae = metrics.mean_absolute_error(y_test, y_predicted)
    mse = metrics.mean_squared_error(y_test, y_predicted)
    r2 = metrics.r2_score(y_test, y_predicted)
    ab = abs(y_test - y_predicted )
    ab = pd.DataFrame(ab)
    q95 = float(ab.quantile(0.95,axis = 0) )
    #q025 = float(ab.quantile(0.025,axis = 0,interpolation= 'midpoint'))
    q025 = 0
    accuracy = [q95,  q025]

    print("The model performance for testing set")
    print("--------------------------------------")
    print('MAE is {}'.format(mae))
    print('MSE is {}'.format(mse))
    print('MAPE is {}'.format(mape))
    print('R2 score is {}'.format(r2))
    print('95% percent of the data lay between ± ' + '%.2f' %(accuracy[0])  )

    #print('95% percent of the data lay between ' + '%.2f' %(accuracy[0]) +' and %.2f C°' %(accuracy[1]) )
          
    return mae,mse,r2,accuracy,ab

def alles(buildings,parameters,iterator):
    
    """ 
    calculate the measures of the fit to evaulate it goodness
   
    Parameters:
        buildings: the simulation data 
        parameters: the predicted date resulting from the fitt-Model
        iterator
        
    Return type:
        alles :  
 
    Description:
        build one big dataframe from the multiple simulation runs. 
    """
    
    for i in tqdm(iterator):
        if i == iterator[0]:
            alles = buildings[i][parameters]
            print(i)
        else :
            alles = pd.concat([alles ,
                        buildings[i][parameters] ] )
    return alles
    
def get_iterator(buildings,Ge,step =2):
    
    lenght = len(buildings)

    if Ge == 0:   
        iterator = np.arange(0,lenght,step)
    elif Ge == 1:
        iterator = np.arange(1,lenght,step) 
    else:
        iterator = np.arange(lenght)
    return iterator






def kmcluster(df,n_clusters = 20,random_state = 42):
    
    """ 
    calculate the measures of the fit to evaulate it goodness
   
    Parameters:
        y_test: the simulation data 
        y_predicted: the predicted date resulting from the fitt-Model
        
    Return type:
        mae : Mean absolute error 
        mse : mean square error
        r2 : r squared
        accuracy : the 95th quantile of the absolute deviation 
        ab : array of the absolute deviation 
 
    Description:
    Resample the input dataframe from 3 minute to 15. only the primary return temperature
    (_ph_hot_out) resampled using the weights of the primary massflow (dm_DH)
    """

    cluster_df =np.array(df)
    scaler0 = StandardScaler().fit(cluster_df)
    cluster_df = scaler0.transform(cluster_df)
    km = KMeans(
        n_clusters=n_clusters, init='random',
        n_init=20, max_iter=500, 
        tol=1e-04, random_state=random_state
    )


    X =np.array(cluster_df)

    clusters = km.fit_predict(X)
    return clusters,km,scaler0




    
def null_leistung_index(buildings,min_leistung_kw = 0.02):
    '''
    produce the index of the points with less than or equal power of min_leistung_kw
    '''
    building0= []
    iterator = np.arange(len(buildings))
    for i in tqdm(iterator):
        ohne_null = buildings[i].loc[(buildings[i][['dQ_DH']] < min_leistung_kw).all(axis=1), :]
        building0.append(ohne_null.index)
    return building0
    

def zirkclust(df):
    '''
    classify the simulation data into two groubs on between 5 hour to 23 and 
    the other from 23 to 5. 
    
    '''
    df['cluster'] = df.index.hour
    hours = df['cluster'].values
    iterator = np.arange(len(hours))
    for i in tqdm(iterator) :

        time = hours[i]
        if time >= 5 and  time < 23:
            hours[i] = 1
        else :
            hours[i] = 0
    return hours


def insert_dcw(data):
    
    '''
    produce the cold water temperatur profile and insert it in the simulation data
    Dataframe
    '''
    tl = 0.05
    TIME = np.arange(0,8760.05,tl)
    dcw = 9.7+6.3*np.sin(2*np.pi*(TIME+(273.75-60)*24)/8760) 
    for i in tqdm(np.arange(len(data))):
        data[i]['dcw'] =  dcw
    return data






def plot_prediction(x_data,y_data,y_predicted,nrows = 1 , r2 =5, building = 'defualt ' ):

    """ 
    Loading of Data for the storage-System of Domestic water heating.
    the code in the function was written orginally by Hagen Brass and was implemented
    through this function by Abdi Mohamed.  
    Parameters:
    path_ISA_csv: the path for the ISA_csv file that defines the column names
    path_sim_data: the path for the simulation data that should be imported or loaded
    
    Return type:return an array of Dataframes, each dataframe is a TRNSYS Simulation run. 
 
    Description:
    Removes the item from the list if the index exists.
    """


    plt.rcParams.update({'font.size': 20})

    fig , axes = plt.subplots(nrows = nrows ,ncols = 1,figsize = (10,6))
    axes = np.array(axes)
    
    for i, ax in enumerate(axes.reshape(-1)):
        ax.scatter(x_data, y_data,color = 'r', label = 'Trnsys Modell')
        ax.scatter(x_data, y_predicted,color = 'b', label = 'Fitted Model')
        ax.grid()
        ax.set_title('Rüklauftemp Über Leistung ' 'r2 = %3f.' %r2  )
        ax.set_xlabel('Leistung [kW]')
        ax.set_ylabel('Rüklauftemp [C°] ')
        plt.xticks(fontsize=20, rotation=0)
        plt.yticks(fontsize=20, rotation=0)
        
        


        ax.grid(True,which='major',axis='both',alpha=1)



        ax.legend()
        ax.plot()
        fig.subplots_adjust(hspace=0.8)
        print(i)
    fig.tight_layout(pad=110)
    
    
def plot_prediction_q95(x_data,y_data,y_predicted,nrows = 1 , r2 =5,q95 = 3, building = 'defualt ',modus = 0 ):
    
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Arial'
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
    mpl.rc('font', **font)


    plt.rcParams.update({'font.size': 20})

    fig , axes = plt.subplots(nrows = nrows ,ncols = 1,figsize = (10,6))
    axes = np.array(axes)
    
    for i, ax in enumerate(axes.reshape(-1)):

        plot3 = ax.scatter(x_data, y_data,color = 'r',alpha = 0.3 , label = 'Trnsys Modell')
        plot4 = ax.scatter(x_data, y_predicted,alpha = 0.3,color = 'b', label = 'Fitted Modell')

        ax.set_facecolor("w")
        ax.grid()
        ax.set_title('Rüklauftemp Über Leistung ',pad = 20 )
        ax.set_xlabel('Leistung [kW]')
        ax.set_ylabel('Rüklauftemp [C°] ')
        plt.xticks(fontsize=20, rotation=0)
        plt.yticks(fontsize=20, rotation=0)
        


        ax.grid(True,which='major',axis='both',alpha=1)


        handles, labels = ax.get_legend_handles_labels()
        print(handles,labels)
        
        ax.legend(prop={'size': 15} )
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)
        ax.plot()
        fig.subplots_adjust(hspace=0.8)
        print(i)
        fig.tight_layout(pad=110)
        if modus == 0 : 
            ax.annotate('r2 = %1.2f' %r2 + '\n'+'Q95 = %1.2f K' %q95, xy=(0.3, 1), xytext=(12, -12), va='top',
                 xycoords='axes fraction', textcoords='offset points')

    
def plot_prediction2(x_data,y_data,y_predicted,nrows = 1 , r2 =5, building = 'defualt' ):
    fig , axes = plt.subplots(nrows = nrows ,ncols = 1,figsize = (19,10))
    axes = np.array(axes)
    
    for i, ax in enumerate(axes.reshape(-1)):
        ax.scatter(x_data, y_data,color = 'r', label = 'Trnsys Modell')
        ax.scatter(x_data, y_predicted,color = 'b', label = 'Fitted Model')
        ax.grid()
        ax.set_title('Rüklauftemp Über Leistung ' + building +  'r2 = %3f.' %r2  )
        ax.set_xlabel('Leistung [kW]')
        ax.set_ylabel('Rüklauftemp [C°] ')
        ax.legend()
        ax.plot()
        fig.subplots_adjust(hspace=0.8)
        print(i)
    fig.tight_layout(pad=110)



def plot_dist_ab(accuracy,nrows = 1 , r2 =5, building = 'defualt ' ):
    

    params = {'legend.fontsize': 22,
          'legend.handlelength': 2}
    plt.rcParams.update(params)

    plt.rcParams.update({'font.size': 20})

    fig , axes = plt.subplots(nrows = nrows ,ncols = 1,figsize = (10,6))
    axes = np.array(axes)
    ab = accuracy[1]
    q95 = accuracy[0]
    for i, ax in enumerate(axes.reshape(-1)):
        sns.distplot(ab,kde=True, norm_hist=True, ax = ax)
        ax.grid()
        ax.set_title('Absolute Abweichungen ' 'r2 = %3f.' %r2  )
        ax.set_xlabel('Absolute Abweichung [K]')
        ax.set_ylabel('Wahrscheinlichkeit Dichte ')
        
        ax.set_xlim([0, q95+1])

        plt.xticks(fontsize=20, rotation=0)
        plt.yticks(fontsize=20, rotation=0)
        
        plt.axvline(x=q95, color='k', linestyle='--',label ='95% Der Daten')


        ax.grid(True,which='major',axis='both',alpha=1)



        ax.legend()
        ax.plot()
        fig.subplots_adjust(hspace=0.8)
        print(i)
    fig.tight_layout(pad=110)
    
    
def plot_prediction(x_data,y_data,y_predicted,nrows = 1 ,
                    r2 =5,q95 = 3,
                    building = 'defualt ',
                    figsize = (10,6)
                    
                    ):
    
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Arial'
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
    mpl.rc('font', **font)


    plt.rcParams.update({'font.size': 20})

    fig , axes = plt.subplots(nrows = nrows ,ncols = 1,figsize = figsize)
    axes = np.array(axes)
    
    for i, ax in enumerate(axes.reshape(-1)):

        plot4 = ax.scatter(x_data, y_predicted,alpha = 0.3,color = 'b', label = 'gefittete Modell')
        
        #plot3 = ax.scatter(x_data, y_data,color = 'black',alpha = 0.3 , label = 'Trnsys Modell')
        plot5 = ax.plot(x_data, y_data+q95,color = 'r',linewidth=3.5,alpha = 1 ,label = 'Trnsys Modell ± %1.2f K' %q95)
        plot6 = ax.plot(x_data, y_data-q95,color = 'r',linewidth=3.5,alpha = 1 )
        
        ax.set_facecolor("w")
        ax.grid()
        ax.set_title('Prediktion Über Rücklautempratur ',pad = 20 )
        ax.set_xlabel('TRNSYS Rücklautempratur [C°]')
        ax.set_ylabel('Fitted Rüklauftemp [C°] ')
        plt.xticks(fontsize=20, rotation=0)
        plt.yticks(fontsize=20, rotation=0)
        
        ax.annotate('r2 = %1.2f' %r2 + '\n'+'Q95 = %1.2f K' %q95, xy=(0.7, 0.3), xytext=(12, -12), va='top',
             xycoords='axes fraction', textcoords='offset points')


        ax.grid(True,which='major',axis='both',alpha=1)

        handles, labels = ax.get_legend_handles_labels()
        print(handles,labels)
        
        ax.legend(prop={'size': 15} )
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)
        ax.plot()
        fig.subplots_adjust(hspace=0.8)
        print(i)
    fig.tight_layout(pad=110)
    
    
    
    
def plot_dist_ab2(ab,q95,nrows = 1 , r2 =5, building = 'defualt ' ):
    

    params = {'legend.fontsize': 22,
          'legend.handlelength': 2}
    plt.rcParams.update(params)

    plt.rcParams.update({'font.size': 20})

    fig , axes = plt.subplots(nrows = nrows ,ncols = 1,figsize = (8,8))
    axes = np.array(axes)
    ab = ab
    q95 = q95
    for i, ax in enumerate(axes.reshape(-1)):
        #sns.distplot(ab,kde=True, norm_hist=True, ax = ax )
        sns.kdeplot(data = ab ,x =ab.iloc[:,0] ,shade = True ,ax = ax)
        ax.grid()
        ax.set_title('Absolute Abweichungen ' 'r2 = %3f.' %r2  )
        ax.set_xlabel('Absolute Abweichung [K]')
        ax.set_ylabel('Wahrscheinlichkeit Dichte ')
        
        ax.set_xlim([0,10])

        plt.xticks(fontsize=20, rotation=0)
        plt.yticks(fontsize=20, rotation=0)
        
        plt.axvline(x=q95, color='k', linestyle='--',label ='95% Der Daten')


        ax.grid(True,which='major',axis='both',alpha=1)



        ax.legend(loc = 'upper right')
        ax.plot()
        fig.subplots_adjust(hspace=0.8)
        print(i)
    fig.tight_layout(pad=110)
    
