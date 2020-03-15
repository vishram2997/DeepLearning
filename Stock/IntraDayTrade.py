import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import numpy as np
import os


#from keras.callbacks import ModelCheckpoint

root_path = 'gdrive/My Drive/Colab Notebooks/Stock/'  #change dir to your project folder


os.remove("Log.csv")
f = open("Log.csv", "a")
conn = psycopg2.connect(host="localhost",database="Trade", user="postgres", password="hGNIS@123PG")
def loadDataSql(symbol):
    
    sql = """SELECT ticker, open, low, high, close, datetime,  (100*date_part('month',datetime) + date_part('day',datetime)) as s, 
            volume, ROW_NUMBER () OVER (ORDER BY datetime) as r  FROM stockdailyintra
                 where ticker = %s order by r asc """
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
    
    return df

    


def loadDataSqlDaily(symbol):
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
    
    return df

    



BALANCE = 100000
AVAIL_BAL = BALANCE
QTY = 0
BUY_QTY = 2000
COST = 0

balanceX = []
priceX =[]


def Buy(qty_b, price):
    global QTY, COST, AVAIL_BAL, BALANCE
    COST = (QTY*COST +qty_b*price)/(QTY+qty_b)
    QTY = QTY + qty_b
    AVAIL_BAL = AVAIL_BAL - (qty_b*price)
    

def Sell(qty_b, price):
    global QTY, AVAIL_BAL, BALANCE
    QTY = QTY - qty_b
    AVAIL_BAL = AVAIL_BAL + (qty_b*price)
    BALANCE = BALANCE + (abs(qty_b)*price - abs(qty_b)*COST)
    
def Short(qty_b, price):
    global QTY,COST, AVAIL_BAL, BALANCE
    COST = (QTY*COST +qty_b*price)/(abs(QTY)+abs(qty_b))
    QTY = QTY - qty_b
    AVAIL_BAL = AVAIL_BAL - (qty_b*price)

def Cover(qty_b, price):
    global QTY,COST, AVAIL_BAL, BALANCE
    QTY = QTY - qty_b
    AVAIL_BAL = AVAIL_BAL + (qty_b*price)
    BALANCE = BALANCE + ( abs(qty_b)*abs(COST) - abs(qty_b)*abs(price))




MINUTE = 360
LAST_TICK =5
LAST_BAL = BALANCE
    
def getAction(data, row):
    act = ['B', 'S', 'SH', 'C', 'H']
    lastMin =  min(data[row-10:row, 4])
    lastMax =  max(data[row-10:row, 4])
    avg_30 = np.mean(data[row-30:row, 4])
    avg_20 = np.mean(data[row-20:row, 4])
    avg_40 = np.mean(data[row-40:row, 4])
    avg_10 = np.mean(data[row-10:row, 4])
    avg_10_L = np.mean(data[row-10-1:row-1, 4])
    avg_5 = np.mean(data[row-5:row, 4])
    
    avg_30_vol = np.mean(data[row-40:row, 7])

    if QTY >0:
        if avg_10 < avg_5:
            return act[1]
        if float((data[row,4]-COST)/data[row,4]) < 0.01:
            return act[1]

    elif QTY <0:
        if avg_10 > avg_5:
            return act[3]
        if float((data[row,4]-COST)/data[row,4]) > 0.01:
            return act[1]
    else:
        if avg_10 > avg_5 :
            if (data[row,4]-data[row-1,4]) > 0:
                return act[0]
        else:
            if (data[row,4]-data[row-1,4]) < 0:
                return act[2]



def runTrial(day, data):
    #print(data[:10])
    global LAST_BAL, BALANCE, BUY_QTY, AVAIL_BAL
    for row in range(40,len(data)-20):
        open,  low , high , close = data[row][1], data[row][2], data[row][3], data[row][4]
        
        
        action = getAction(data, row)


        if action == 'B':
            Buy(BUY_QTY, close)
        
        if action == 'S':
            Sell(QTY, close)
        
        if action =='SH':
            Short(BUY_QTY, close)
        
        if action == 'C':
            Cover(QTY, close)
        
        if row > (len(data)-25):
            if QTY >0:
                Sell(QTY, close)
            if QTY <0:
                Cover(QTY, close)
        
        PROFIT =   BALANCE -LAST_BAL
        LAST_BAL = BALANCE
        dataTime = data[row, 5]
        #print(f"{QTY}, {COST}, {close}, {BALANCE} , {PROFIT}, {data[raw,5]}" )
        f.write(f"{data[row][1]}, {QTY}, {COST}, {close}, {BALANCE} , {PROFIT}, {dataTime}")
        f.write("\n")
        if PROFIT != 0:
            balanceX.append(BALANCE)
            priceX.append(close)
        #if QTY == 0:
        #    print(f"BALANCE: {BALANCE}")


f.write(f"Symbol, QTY, COST, PRICE, BALANCE, PROFIT, avg10, avg30")
f.write("\n")

def selectCompany(conn):
    sql = """SELECT * FROM ticker;"""
    try:
        cur = conn.cursor()
        cur.execute(sql)
        companies = cur.fetchall()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
   
    return companies
     

companies = selectCompany(conn)
companies = [comp[0] for comp in companies if comp[0]=='AMZN']
for ticker in companies:
    #print(ticker)
    #print(f"QTY, COST, PRICE, BALANCE, PROFIT, avg10, avg30" )       
    df = loadDataSql(ticker)
    df['s'] = df[6]
    
    balanceX =[]
    
    if float(df[4].head(1)):
        #df['s'] = df['datetime'].date()
        #df = pd.read_csv("TSLAIntra.csv")
        df.sort_values(by=8, inplace=True)
        BALANCE = 100000
        grp = df.groupby(['s']).mean().reset_index()['s']
        #print(grp)
        i=0
        for g in grp.values:
            i+=1
            
            BUY_QTY = int(BALANCE / float(df[4].head(1)))
            data = df[df.s.eq(g)]
            data = data.values;
            runTrial(i, data)
            print(f"{ticker}, {QTY}, {BALANCE}")
        plt.plot(balanceX, label=ticker)

f.close()


#plt.plot(priceX, label='Price')

plt.show()
