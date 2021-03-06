import psycopg2	

import pandas as pd
import requests
import numpy as np
import os
import csv
import json

conn = psycopg2.connect(host="73.170.217.34",database="Trade", user="postgres", password="hGNIS@123P")


def insertCompany(conn):
    companyList = [['ROKU','ROKU']]
    company = np.array(companyList)

    
    sql = """INSERT INTO ticker(code, name)
                VALUES(%s, %s) RETURNING code;"""
    try:
        cur = conn.cursor()
        cur.executemany(sql, (company))
        conn.commit()
        print(cur.rowcount)
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
  

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
     
     


apikey = "3HJSYM7HFM6OT87F"
URL = "https://www.alphavantage.co/query?function="
function = "TIME_SERIES_DAILY"
outputsize="full"
datatype = "json"
import time

def updateStockData(conn,symbol ):
    start_time = time.time()
    
    dataURL = f"{URL}{function}&symbol={symbol}&apikey={apikey}&outputsize={outputsize}&datatype={datatype}"
    download = requests.get(dataURL)
    download = json.loads(download.content)
    data =pd.read_json(json.dumps(download['Time Series (Daily)'])).transpose().reset_index()
    data['symbol'] = symbol
    
    #print(data.values.tolist())
    
    sql = """INSERT INTO public.stockdaily(
                    date, open,  high, low, close, volume,ticker)
                VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING ticker;"""
    try:
        cur = conn.cursor()
        cur.executemany(sql, (data[1:].values.tolist()))
        conn.commit()
        print(cur.rowcount)
        cur.close()
        diff = time.time() - start_time
        if(diff < 12):
            time.sleep(12-diff)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    
        


def updateStockDataDaily(conn):
    companyList =selectCompany(conn)
    companyList = [comp[0] for comp in companyList]

    company = np.array(companyList)
    for symbol in company[31:]:
        start_time = time.time()
        
        dataURL = f"{URL}{function}&symbol={symbol}&apikey={apikey}&outputsize={outputsize}&datatype={datatype}"
        download = requests.get(dataURL)
        download = json.loads(download.content)
        data =pd.read_json(json.dumps(download['Time Series (Daily)'])).transpose().reset_index()
        data['symbol'] = symbol
        
       #print(data.values.tolist())
        
        sql = """INSERT INTO public.stockdaily(
                     date, open,  high, low, close, volume,ticker)
                    VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING ticker;"""
        try:
            cur = conn.cursor()
            cur.executemany(sql, (data.values.tolist()))
            conn.commit()
            print(cur.rowcount)
            cur.close()
            diff = time.time() - start_time
            if(diff < 12):
                time.sleep(12-diff)
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
       

#insertCompany(conn)
updateStockData(conn,'ROKU')