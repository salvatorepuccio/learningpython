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
import warnings
warnings.filterwarnings("ignore")

# prende una data di inizio e una data di fine (stringhe) e restituisce una lista di tutti i giorni in mezzo
def fill_dates(date_start,date_end):
    # filling data_format
    dates = pd.date_range(date_start,date_end)
    ret = []
    for x in dates:
        ret.append(x.date())
    return ret


def get_day_of_the_year(l):
    l1 = []
    for val in l:
        l1.append(val.timetuple().tm_yday)
    return l1

# mette degli 0 nella date in cui non ci sono state vendite
def fill_dataframe(df,colonne_df):
    full_calendar = fill_dates(str(df['data_format_date'][0]),str(df['data_format_date'][len(df)-1]))
    j = 0
    N_date_mancanti = 0
    for curr_date in full_calendar:
        # Se la data corrente e' una di quelle in cui ho fatto una vendita
        if(str(curr_date).split(" ")[0] == str(df['data_format_date'][j])):
            # print("Presente "+str(curr_date))
            # non fare nulla
            j = j + 1
        else:
            # altrimenti metti una "vendita" 0
            N_date_mancanti += 1
            # print("Manca "+str(curr_date))
            new_row = [curr_date,0,0,0]
            dff = pd.DataFrame([new_row],columns=colonne_df)
            df = pd.concat([df,dff])
    
    df["data_format_date"] = pd.to_datetime(df["data_format_date"])
    df = df.sort_values(by="data_format_date")
    print("Mancavano "+str(N_date_mancanti)+" date")
    df = df.reset_index(drop=True)
    return df

def somma_duplicate(df):
    data_prec = "NULL"
    i = 0
    eliminate = 0
    end = len(df)
    for i in range(0,end):
        if(str(df['data_format_date'][i]) == data_prec):
            eliminate+=1
            df['qta'][i] += df['qta'][i-1]
            df['val'][i] += df['val'][i-1]
            df = df.drop(i-1)
            end -= 1

        data_prec = str(df['data_format_date'][i])
    print("Eliminate "+str(eliminate)+" duplicazioni")
    df = df.reset_index(drop=True)
    return df
        

def myplot2(x,y,y1):
    fig, ax = plt.subplots()
    ax = plt.gca()
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.WeekdayLocator(interval=4)
    ax.xaxis.set_major_locator(locator)
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    ax.plot(x,y,linewidth=1)
    ax.plot(x[-len(y1):],y1,color='red',linewidth=1)
    plt.show()

def myplot(x,y):
    fig, ax = plt.subplots()
    ax = plt.gca()
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.WeekdayLocator(interval=4)
    ax.xaxis.set_major_locator(locator)
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    ax.plot(x,y)
    plt.show()

def print_all(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

def get_days(df_col):
    ret = []
    for x in df_col:
        ret.append(x.weekday())
    return ret

def get_week(df_col):
    ret = []
    for x in df_col:
        ret.append(x.isocalendar()[1])
    return ret

def get_unit_price(df_qta_val):
    ret = []
    append_me=0
    for i in range(0,len(df_qta_val)):
        if(df_qta_val['val'][i]==0):
            append_me = 0
        else:
            append_me = df_qta_val['val'][i] / df_qta_val['qta'][i]
        ret.append(append_me)
        # print("qta "+str(df_qta_val['qta'][i])+" val "+str(df_qta_val['val'][i])+" prezzo unitario "+str(append_me))
    return ret

    

client = Client(
                host='127.0.0.1',
                user='savato',
                password='savato',
                database='chpv',
                )
client.execute("use chpv")

# | 000249036     PARMAREG.L'ABC MERENDA IVA 10% │
# │ 000249037     PARMAREG.L'ABC MERENDA IVA 22% │
# │ 005068383     CALZ.TREND SPORT BICOLORE 2NNN │
# │ 005158930     GAMB.MICRO 50 DEN G.LADY 110K  │
# │ 005195001     GAMB.STRETCH 15 DEN G.LADY x 2 │
# │ 005195029     COLLANT BODY FORM GOLDEN LADY  │
# │ 005233311     GAMB.TREND MICRO 30 D          │
# │ 005300679     COLLANT LEDA FILANCA G.LADY x2 │
# │ 005544076     COLLANT MY BEAUTY 50           │
# │ 005809311     COLLANT SILHOUETTE 30          │
# │ 005818542     COLLANT BENESS.COMPR.MEDIA 70D │
# │ 066216584     OMB.MINI UOMO U/LIG.AUT.TEC.   │
# │ 080112701     PRORASO CREMA PREB.ML.100 VASO │
# │ 080112702     PRORASO CIOTOLA BARBA ML.150   │
# │ 080112703     PRORASO SCH.BARBA RINFR.ML400# │
# │ 080112707     PRORASO SCH.BARBA PROTET.400ML │
# │ 080112709     PRORASO SCH-BARBA P.SENS.ML300 │
# │ 080119200     MASCARA THE ROCKET VERY BLACK  │
# │ 080123754     FIGARO SCH.BARBA SENSITIVE 400 │
# │ 080124800     I PROVENZALI SAP.LIQ MAND.DOL  |
# | 163854067     FINDUS 30 BASTONCINI MERL G750 |
# 148520655     VITASNELLA ACQUA CL.50
# 148520098     S.BENEDETTO ACQUA LT.2
# 148520189     ULIVETO ACQUA EFFERV.LT.1.5

# ┌─rag_soc────────┐
# │ SUPERSTORE 01  │
# │ SUPERSTORE 02  │
# │ SUPERMARKET 01 │
# │ SUPERSTORE 05  │
# │ IPERSTORE 01   │
# │ IPERSTORE 02   │
# │ SUPERSTORE 06  │
# │ SUPERMARKET 02 │
# │ SUPERSTORE 03  │
# │ SUPERMARKET 03 │
# │ SUPERMARKET 04 │
# │ SUPERMARKET 05 │
# │ SUPERMARKET 06 │
# │ SUPERSTORE 04  │
# │ SUPERSTORE 07  │
# │ IPERSTORE 03   │
# │ IPERSTORE 04   │
# │ SUPERMARKET 07 │
# │ SUPERSTORE 08  │
# └────────────────┘
# ┌─rag_soc────────┐
# │ SUPERMARKET 08 │
# │ SUPERMARKET 09 │
# │ SUPERMARKET 10 │
# │ SUPERSTORE 09  │
# │ SUPERSTORE 10  │
# │ SUPERMARKET 11 │
# │ SUPERMARKET 12 │
# │ SUPERSTORE 11  │
# │ SUPERMARKET 13 │
# │ SUPERSTORE 12  │
# │ SUPERSTORE 13  │
# └────────────────┘
# ┌─rag_soc────────┐
# │ SUPERSTORE 14  │
# │ SUPERMARKET 14 │
# │ SUPERSTORE 15  │
# └────────────────┘

sql = "Select data_format_date,qta,val,flag_off from dump2 where cod_prod='148520189' and rag_soc = 'IPERSTORE 03' group by data_format_date,val,flag_off,qta order by data_format_date"
query_result = client.execute(sql, settings = {'max_execution_time' : 3600})

cols = ['data_format_date','qta','val','flag_off']
df = pd.DataFrame(query_result)
df.columns = cols

# Somma le righe duplicate, ovvero con la stessa data
df = somma_duplicate(df)

# Aggiunge una riga per ogni data mancante all'interno del range della query, con qta = 0
df = fill_dataframe(df,cols)
print(len(df))

# Plot delle vendite (0 compresi) nel periodo della query
# myplot(df['data_format_date'],df['qta'])

# Estrapolazione dalle date di num settimana e giorno della settimana
day_of_week = get_days(df['data_format_date'])# num [0,6] lun,mar,mer....dom
print(len(day_of_week))
week_n = get_week(df['data_format_date'])
unit_price = get_unit_price(df[['qta','val']])
day_of_year = get_day_of_the_year(df['data_format_date'])

# Inserimento nuove colonne
df.insert(0, 'day_of_week', day_of_week, True)
df.insert(0, 'week_n', week_n, True)
df.insert(0,'unit_price', unit_price, True)
df.insert(0,'day_of_year', day_of_year, True)

# print_all(df)

y = df['qta'].to_numpy()
X = df[['day_of_year','unit_price','week_n','day_of_week','flag_off']].to_numpy()

# Setto le dimensioni di train e test (80,20)
total_size = len(X)
train_size = math.floor(total_size / 100 * 80)
test_size = total_size - train_size
# X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
print("Total "+str(len(X))+" train size "+str(train_size)+" test size "+str(test_size))

# Divido i dati in train e test
X_train = X[:train_size]
X_test = X[train_size:total_size]

y_train = y[:train_size]
y_test = y[train_size:total_size]

# Alleno il modello
regr = MLPRegressor(random_state=1, max_iter=5000).fit(X_train, y_train)

# Predico su X_test
y_pred = regr.predict(X_test)

# Stampo l'accuracy
print(regr.score(X_test, y_test))

# Tronco i valori predetti
y_pred = np.trunc(y_pred)
# print(y_test)
# print(y_pred)

# Mostro i risultati
myplot2(df['data_format_date'],y,y_pred)