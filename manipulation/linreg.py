from re import I
from turtle import clear
from clickhouse_driver import Client
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates
import datetime
import math

def to_integer(dt_time):
    return 10000*dt_time.year + 100*dt_time.month + dt_time.day

# prende una lista di date e restituisce una lista di indici che corrispondono
# ai giorni dell'anno (1 gennaio = 1... l'anno non conta)
def to_day_of_the_year(l):
    l1 = []
    for val in l:
        l1.append(val.timetuple().tm_yday)
    return l1

def my_to_numpy(mylist):
    d = pd.DataFrame(mylist,columns=['col'])
    return d['col'].to_numpy()

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


def myplot2(x,y,y1):
    fig, ax = plt.subplots()
    ax = plt.gca()
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.WeekdayLocator(interval=4)
    ax.xaxis.set_major_locator(locator)
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    ax.plot(x,y)
    ax.plot(x[-len(y1):],y1,color='red')
    plt.show()

# prende una data di inizio e una data di fine (stringhe) e restituisce una lista di tutti i giorni in mezzo
def fill_dates(date_start,date_end):
    # filling data_format
    # data_format_full = np.arange(data_format[0], data_format[len(data_format)-1])
    start = datetime.datetime.strptime(date_start, "%Y-%m-%d")
    end = datetime.datetime.strptime(date_end, "%Y-%m-%d")
    return pd.date_range(start, end)
    
# mette degli 0 nella date in cui non ci sono state vendite
def fill_sells(full_calendar,mydays):
    sells_full = []
    j = 0
    for curr_date in full_calendar:
        # Se la data corrente e' una di quelle in cui ho fatto una vendita
        if(str(curr_date).split(" ")[0] == str(mydays[j])):
            # inserisci il venduto
            sells_full.append(sold[j])
            j = j + 1
        else:
            # altrimenti metti 0
            sells_full.append(0)

    return sells_full




client = Client(
                host='127.0.0.1',
                user='savato',
                password='savato',
                database='chpv',
                )
client.execute("use chpv")




# prodotto = 'WURSTEL'
# data_inizio = '2021-11-01'
# data_fine = '2022-01-31'


# sql = "Select data_format_date,sum(qta) from dump2 where data_format_date>'"+data_inizio+"' and data_format_date<'"+data_fine+"' and descr_prod like '%"+prodotto+"%' and flag_off=0 group by data_format_date limit 50000000"
# # sql = "Select data_doc,sum(qta) from dump2 where cod_prod = '145644015' and flag_off=0 group by data_doc"
# print(sql)
# query_result = client.execute(sql, settings = {'max_execution_time' : 3600})

# cols = ['data','sum '+prodotto+' offerta']
# df = pd.DataFrame(query_result)
# df.columns = cols
# # print(df)


# print("Query result lines: "+str(len(df.index)))


# if len(df.index) > 0:

#     ax=df.plot(x=cols[0], y=cols[1],figsize=(20,20))
#     # ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
#     # plt.grid(True)
#     # plt.gcf().autofmt_xdate() 
#     plt.title("SOMMA VENDITE PRODOTTI CONTENGONO '"+prodotto+"' IN DESCRIZIONE\nDAL "+data_inizio+" A "+data_fine+" ",fontsize=24)
#     x = df[cols[0]].to_numpy()
#     y = df[cols[1]].to_numpy()
#     ax=plt.plot(x, y, 'o', color='black')
#     ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
#     plt.grid(True)
#     plt.gcf().autofmt_xdate() 
#     plt.show()



sql = "Select data_format_date,data_doc,sum(qta) from dump2 where cod_prod = '081220123' and flag_off=0 group by data_format_date,data_doc order by data_doc"
# print(sql)#BAULI PANETTONE TRADIZION. KG1 145644015 
query_result = client.execute(sql, settings = {'max_execution_time' : 3600})

cols = ['data_format_date','data_doc','sum']
df = pd.DataFrame(query_result)
df.columns = cols

# preparare il dataframe
# devo inserire nelle date mancanti, un valore di vendita = 0
data_format = df['data_format_date'].to_numpy()
data_doc = df['data_doc'].to_numpy()
sold = df['sum'].to_numpy()

# fill dates
data_format_full = fill_dates(str(data_format[0]),str(data_format[len(data_format)-1]))
# filling data_doc
data_index_full = to_day_of_the_year(data_format_full)
# print(data_index_full)

# fill sells
sells_full = fill_sells(data_format_full,data_format)

# myplot(data_format_full,sells_full)



# # converting to numpy arrays
# X = np.array(data_index_full).reshape((-1,1))
# # print(len(data_index_full))
# y = np.array(sells_full)
# # print(len(y))
# myplot(data_format_full,y)

# #training model
# model = LinearRegression().fit(X,y)

# # print(type(datetime.date(2022,1,1)))
# # print(datetime.date(2022,1,1))

# data_format_pred = fill_dates("2022-01-01","2022-01-31")
# print(len(data_format_pred))
# data_index_pred = to_day_of_the_year(data_format_pred)
# print(len(data_index_pred))
# X_pred = np.array(data_index_pred).reshape((-1,1))
# print(len(X_pred))
# y_pred = model.predict(X_pred)

# #clean y_pred
# clean_y_pred =  np.array( [ math.floor(num) for num in y_pred ] )

# print(len(clean_y_pred))
# print(clean_y_pred)

# myplot(data_format_pred,y_pred)
# # myplot2(data_format_full,y,y_pred)