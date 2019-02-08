class mylib:
    def yahooKeyStats(stock,start,end):
        import time as _time
        import datetime
        import requests
        import pandas as pd
        import re
        import string
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
