from turtle import clear
from clickhouse_driver import Client
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split 
import matplotlib.dates as mdates
from datetime import datetime
import math

# prende una data di inizio e una data di fine (stringhe) e restituisce una lista di tutti i giorni in mezzo
def fill_dates(date_start,date_end):
    # filling data_format
    dates = pd.date_range(date_start,date_end)
    ret = []
    for x in dates:
        ret.append(x.date())
    return ret


def to_day_of_the_year(l):
    l1 = []
    for val in l:
        l1.append(val.timetuple().tm_yday)
    return l1

# mette degli 0 nella date in cui non ci sono state vendite
def fill_dataframe(full_calendar,mydays,datafr,ragione_sociale,colonne_df):
    j = 0
    for curr_date in full_calendar:
        # print(str(curr_date).split(" ")[0]+"   "+ str(mydays[j])+" j: "+str(j))
        # Se la data corrente e' una di quelle in cui ho fatto una vendita
        if(str(curr_date).split(" ")[0] == str(mydays[j])):
            # inserisci il venduto
            j = j + 1
        else:
            # altrimenti metti 0
            new_row = [ragione_sociale,curr_date,0,0,0]
            dff = pd.DataFrame([new_row],columns=colonne_df)
            datafr = pd.concat([datafr,dff])
            # datafr = datafr.append(new_row,ignore_index=True)
    
    datafr["data_format_date"] = pd.to_datetime(datafr["data_format_date"])
    datafr = datafr.sort_values(by="data_format_date")

    return datafr

def convert_rag_soc(datafr):
    converted_values = []
    for elem in datafr['rag_soc']:
        print("current elem:")
        print(elem)
        category = elem.split(" ")[0]
        number = elem.split(" ")[1]
        print("category:")
        print(category)
        print("number:")
        print(number)
        if(category == "SUPERSTORE"):
            converted_values.append(float(str(1)+str(number)))
        if(category == "SUPERMARKET"):
            converted_values.append(float(str(2)+str(number)))
        if(category == "IPERSTORE"):
            converted_values.append(float(str(3)+str(number)))
    datafr.drop(axis=1,columns='rag_soc')
    datafr['rag_soc'] = converted_values
    



client = Client(
                host='127.0.0.1',
                user='savato',
                password='savato',
                database='chpv',
                )
client.execute("use chpv")

sql = "Select rag_soc,data_format_date,qta,val,flag_off from dump2 where cod_prod='163854067' order by data_format_date"
query_result = client.execute(sql, settings = {'max_execution_time' : 3600})

cols = ['rag_soc','data_format_date','qta','val','flag_off']
df = pd.DataFrame(query_result)
df.columns = cols

date_query = df['data_format_date'].to_numpy()
# print(date_query)

date_complete = fill_dates(str(date_query[0]),str(date_query[len(date_query)-1]))
# print(date_complete)

df2 = fill_dataframe(date_complete,date_query,df,'SUPERSTORE 01',cols)

convert_rag_soc(df2)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df2)

# print(type(df2['data_format_date'][0]))
# print(type(df2['data_format_date'][660]))
# date_indici = to_day_of_the_year(date_complete)

# prezzi = df['val'].to_numpy()

# flag_offerta = df['flag_off'].to_numpy()

# qta_venduto = df['qta'].to_numpy()

# y = df2['qta'].to_numpy()

# df2 = df2.drop(axis=1,columns='qta')
# X = df2.to_numpy()







# # X, y = make_regression(n_samples=200, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
# regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
# regr.predict(X_test[:2])
# regr.score(X_test, y_test)

