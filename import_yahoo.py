# -*- coding: utf-8 -*-
"""
Created on Fri May 26 21:05:35 2017

@author: Lenovo User
"""
#NOTA: PROGRAMA ELABORADO EN PYTHON 2.7

import time as _time
import datetime
import requests
import pandas as pd
import re
import string

#%%

def yahooKeyStats(stock,start,end):
    try:
        url='https://query1.finance.yahoo.com/v7/finance/download/'+stock+'?period1='+str(end)+'&period2='+str(start)+'&interval=1d&events=history&crumb=p46S32cqcsI'
        files = {'file': ('report.csv', 'some,data,to,send\nanother,row,to,send\n')}
        r = requests.post(url, files=files)
        lines = [re.sub('\s+' ,' ' , line.strip() , 1) for line in r.text.split('\n')]
        S = pd.Series(lines)
        data = S.str.split(',').tolist()
        data=data[:-1]
        data=pd.DataFrame(data)
        header=data.iloc[0,:]
        index=data.iloc[:,0]
        data=pd.DataFrame(data.iloc[1:,1:])
        data.columns=header[1:]
        data.index=index[1:]
        return data
    except Exception:
        print (stock + ' not found')
        return -1
    
#%%
    
stock = ['AC','ALFAA','ALPEKA']

today = datetime.date.today()
days =datetime.timedelta(days=365) #Buscamos 1 año de historia

timestamp=today-days #Solo es para observar que la fecha sea correcta
start = int(_time.mktime(today.timetuple())) #fecha inicial

timestamp2 = datetime.datetime.fromtimestamp(start) #Solo es para observar que la fecha sea correcta
end= int(_time.mktime(timestamp.timetuple())) #fecha final            
#%%

for j in stock:
    data=yahooKeyStats(j+'.MX',start,end) #descarga los datos de cada ticker
    data.to_csv(('%s.csv')%j) #exporta los datos de cada ticker a un csv.
    ### exec("data_%s=data" % (j))
    
