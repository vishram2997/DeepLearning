import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import numpy as np
import os


from keras.callbacks import ModelCheckpoint

root_path = 'gdrive/My Drive/Colab Notebooks/Stock/'  #change dir to your project folder


os.remove("Log.csv")
f = open("Log.csv", "a")

def loadDataSql(symbol):
    conn = psycopg2.connect(host="73.170.217.34",database="Trade", user="postgres", password="hGNIS@123PG")
    sql = """SELECT ticker, open, low, high, close, datetime,  date_part('day',datetime) as s, 
            volume  FROM stockdailyintra
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
    

def runTrial(day, data):
    #print(data[:10])
    global LAST_BAL, BALANCE, BUY_QTY, AVAIL_BAL
    for row in range(40,len(data)):
        open,  low , high , close = data[row][1], data[row][2], data[row][3], data[row][4]
        
        lastMin =  min(data[row-10:row, 4])
        lastMax =  max(data[row-10:row, 4])
        avg_30 = np.mean(data[row-30:row, 4])
        avg_20 = np.mean(data[row-20:row, 4])
        avg_40 = np.mean(data[row-40:row, 4])
        avg_10 = np.mean(data[row-10:row, 4])
        avg_10_L = np.mean(data[row-10-1:row-1, 4])
        avg_5 = np.mean(data[row-5:row, 4])
        
      
        
        if  (avg_10 > avg_5):
            if QTY < 0:
                Cover(QTY, close) 
            if QTY==0:
                Buy(BUY_QTY, close)
            if (QTY >0) and (QTY <(2*BUY_QTY)) and (COST > close):
                Buy(BUY_QTY, close)
        
            
            
        
        if  (avg_10 < avg_5):
            if QTY >0: 
                Sell(QTY, close)
            if QTY==0:    
                Short(BUY_QTY, close)

            if (QTY <0) and (COST < close):
                Cover(QTY, close)
                
        
        
        if row > (len(data)-10):
            if QTY >0:
                Sell(QTY, close)
            if QTY <0:
                Cover(QTY, close)
        
       
      
      
        PROFIT =   BALANCE -LAST_BAL
        LAST_BAL = BALANCE
        #print(f"{QTY}, {COST}, {close}, {BALANCE} , {PROFIT}, {avg_10}, {avg_30}" )
        f.write(f"{QTY}, {COST}, {close}, {BALANCE} , {PROFIT}, {avg_10}, {avg_30}")
        f.write("\n")
        if PROFIT != 0:
            balanceX.append(BALANCE)
            priceX.append(close)
        #if QTY == 0:
        #    print(f"BALANCE: {BALANCE}")


f.write(f"QTY, COST, PRICE, BALANCE, PROFIT, avg10, avg30")
f.write("\n")



print(f"QTY, COST, PRICE, BALANCE, PROFIT, avg10, avg30" )       
#df = loadDataSql('TSLA')
#df['s'] = df[6]
#df['s'] = df['datetime'].date()
df = pd.read_csv("TSLAIntra.csv")
df.sort_values(by='s', inplace=True)

grp = df.groupby(['s']).mean().reset_index()['s']
print(grp)
i=0
for g in grp.values:
    i+=1
    data = df[df.s.eq(g)]
    data = data.values;
    runTrial(i, data)
    print(f"QTY: {QTY} BALANCE: {BALANCE}")

f.close()


#plt.plot(priceX, label='Price')
plt.plot(balanceX, label='Balance')
plt.show()
