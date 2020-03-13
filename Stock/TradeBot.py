import pandas as pd
from keras import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sklearn.preprocessing import MinMaxScaler
import os


conn = psycopg2.connect(host="73.170.217.34",database="Trade", user="postgres", password="hGNIS@123PG")
scaler = MinMaxScaler(feature_range = (0, 1))


def loadDataSql(symbol):
    sql = """SELECT ticker, open, low, high, close, date, 
            volume  FROM stockdaily
                 where ticker = %s """
    companies =[]
    try:
        
        cur = conn.cursor()
        cur.execute(sql,(symbol,))
        companies = cur.fetchall()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

    df = pd.DataFrame(companies)
    df = df.sort_values(by=5)
    '''
    df['HighLow'] = df[3] - df[2]
    df['CloseOpen'] = df[4] - df[1]
    df['change'] = df[4].diff().fillna(0)
    df['SMA_10'] = df.iloc[:,4].rolling(window=10).mean().fillna(0)
    df['SMA_30'] = df.iloc[:,4].rolling(window=30).mean().fillna(0)
    
    y =np.array(df['change'].values)
    '''
    df = df.drop([0,5,6], axis=1)
    #print(df.head(20))
    data = np.array(df.iloc[:, 3:4].values)
    return data

    
    
stockData = loadDataSql('TSLA')


BALANCE = 10000
ORDERS = []
QTY = 0


ACTION =''
    
print(stockData)
def findAction(data):
    ma_10 = data[len(data)-10:].mean()
    ma_30 = data[len(data)-30:].mean()
    
    if ma_10 > ma_30:
        return 'SELL'
    else:
        return 'BUY'
        

balanceChanges = []

for i in range(60,len(stockData)-1):
    ACTION = findAction(stockData[:i+1])
    if ACTION == 'BUY':
        if (BALANCE > (stockData[i]*10)) and (QTY <20):
            BALANCE = BALANCE - (stockData[i]*10)
            QTY +=10
            
    if ACTION == 'SELL':
        if QTY >=10:
            BALANCE = BALANCE - (stockData[i]*QTY)
            QTY = 0
            
    balanceChanges.append(BALANCE)
        
plt.plot(balanceChanges)
plt.show()
    
    
    
    
    
    

    