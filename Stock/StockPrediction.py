import pandas as pd
import requests
import numpy as np
import os
import csv
import time

apikey = "3HJSYM7HFM6OT87F"
URL = "https://www.alphavantage.co/query?function="
function = "TIME_SERIES_DAILY"
outputsize="full"
datatype = "csv"

def downloadDataFile():
    companyList = pd.read_csv("Top100.csv")
    companyList = companyList['Symbol']

    company = np.array(companyList)


    for symbol in company:
        if not os.path.exists(symbol+'.csv'):
            dataURL = f"{URL}{function}&symbol={symbol}&apikey={apikey}&outputsize={outputsize}&datatype={datatype}"
            download = requests.get(dataURL)
            
            with open(symbol+'.csv','wb') as f:
                print(f'writing file {symbol}.csv')
                f.write(download.content)
                time.sleep(12)





 
