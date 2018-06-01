# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:05:25 2017

@author: kisha_000
"""

import string
import pandas as pd
import numpy as np

def encode(df,pre_type):

    #pre-processing I
    if(pre_type == 1):
        #dictionary used to replace event by alphabetical order letters
        alphabet = list(string.ascii_uppercase)
        di = {k+1: alphabet[k] for k in range(len(alphabet))}   
        
        #remove duplicates [id_doente,event]
        df.drop_duplicates(df.columns[0:2],inplace=True)
        
        #remove rows with NaN values
        df.dropna(inplace = True)
        
        #reset index to make from 0 to len(dataframe)
        df = df.reset_index(drop=True)
        
        #replace n_bio_corrente int values by alphatebical letters
        df.replace({df.columns[1]: di}, inplace = True)
        
        #create new column that adds column event and time
        #this is an auxiliar step to be able to encode our sequences
        df['aux_encode'] = df.iloc[:,1] + ',' + df.iloc[:,2].astype(int).astype(str)
    
        #groupby id_doente and create our sequences - now we have a dataframe with id_doente and encode (encoded sequence)
        df = df.groupby(df.columns[0])['aux_encode'].apply(lambda x: '.'.join(x)).reset_index()
        
        df['aux_encode'] = '0.' + df['aux_encode'] + '.Z'
    
    #pre-processing II
    if(pre_type == 2):
        #3 needed variables
        id_patient = df.columns[0]
        event = df.columns[1]
        date = df.columns[2]
        
        #remove duplicates [id_doente,date]
        df.drop_duplicates([df.columns[0],df.columns[2]],inplace = True)
    
        #remove rows with NaN values
        df.dropna(inplace = True)
    
        #reset index to make from 0 to len(dataframe)
        df = df.reset_index(drop=True)
    
        #create new column with das28_4v values replaced into 4 different events A,B,C,D
        #remission das28<2.6
        df.loc[df[event] < 2.6, 'event'] = 'A'
        #low disease activity 2.6<=das28<3.2
        df.loc[(df[event] <3.2) & (df[event]>=2.6), 'event'] = 'B'
        #medium disease activity 3.2<=das28<=5.1
        df.loc[(df[event] <=5.1) & (df[event]>=3.2), 'event'] = 'C'
        #high disease activity
        df.loc[df[event] > 5.1, 'event'] = 'D'
    
        #compress consecutive events
        index_to_drop = []
        for i in df.index:
            if(i==0):
                continue
            if(df.iloc[i]['event'] == df.iloc[i-1]['event']):
                if(df.iloc[i][id_patient]!=df.iloc[i-1][id_patient]):
                    continue
                else:
                    index_to_drop.append(i)
        df = df.drop(index_to_drop)
    
        #reset index to make from 0 to len(dataframe)
        df = df.reset_index(drop=True)
    
        #convert dt_consulta column into datetime column
        df[date] = pd.to_datetime(df[date],dayfirst=True)
    
        #create auxily column with time intervals between appointments
        df['time_intervals'] = df[date].sub(df[date].shift())
    
        #convert previous result time_intervals in floats
        df['time_intervals'] = df['time_intervals'] / np.timedelta64(1, 'D')
    
        #set the time interval of the first appointment of each patient to zero
        df.loc[df.groupby(id_patient,as_index=False).head(1).index,'time_intervals'] = 0
    
        #create new column that adds column time_intervals and event
        #this is an auxiliar step to be able to encode our sequences
        df['aux_encode'] = df['time_intervals'].astype(int).astype(str) + '.' + df['event']
    
        #groupby id_doente and create our sequences - now we have a dataframe with id_doente and encode (encoded sequence)
        df= df.groupby(id_patient)['aux_encode'].apply(lambda x: ','.join(x)).reset_index()
       
    return df
