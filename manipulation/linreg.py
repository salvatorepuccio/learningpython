from re import I
from turtle import clear
from clickhouse_driver import Client
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates
import datetime

def to_integer(dt_time):
    return 10000*dt_time.year + 100*dt_time.month + dt_time.day

def date_to_int(l):
    l1 = []
    for val in l:
        ymd = str(val).split("-")
        l1.append(int(ymd[0]+ymd[1]+ymd[2]))
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
    ax.plot(x, y)
    plt.show()

client = Client(
                host='127.0.0.1',
                user='savato',
                password='savato',
                database='chpv',
                )
client.execute("use chpv")


# sql = "Select data_format_date,sum(qta) from dump2 where data_format_date between '2020-01-01' and '2020-01-31' and flag_off=0 group by data_format_date"
# # cols = ['id','cod_cli_for', 'rag_soc', 'cod_prod', 'descr_prod', 'data_doc', 'data_format_date', 'qta_offerta', 'qta_non_offerta', 'val_off', 'val_non_off']
# columns = ['data','somma_venduto_non_offerta']
# query_result = client.execute(sql, settings = {'max_execution_time' : 3600})
# df = pd.DataFrame(query_result, columns = columns)

# df.plot(x ='data', y='somma_venduto_non_offerta', kind = 'line')
# plt.show()





# sql = "Select data_format_date,sum(qta) from dump2 where cod_prod = '163850127' group by data_format_date"
# # cols = ['id','cod_cli_for', 'rag_soc', 'cod_prod', 'descr_prod', 'data_doc', 'data_format_date', 'qta_offerta', 'qta_non_offerta', 'val_off', 'val_non_off']
# columns = ['data','sum(venduto bastoncini)']
# query_result = client.execute(sql, settings = {'max_execution_time' : 3600})
# df = pd.DataFrame(query_result, columns = columns)

# df.plot(x ='data', y='sum(venduto bastoncini)',figsize=(21,14))
# plt.show()




# sql = "Select data_format_date,sum(qta) from dump2 where cod_prod = '163850127' and flag_off=1 group by data_format_date"
# columns = ['data','sum(venduto bastoncini offerta)']
# query_result = client.execute(sql, settings = {'max_execution_time' : 3600})
# df = pd.DataFrame(query_result, columns = columns)

# df.plot(x ='data', y='sum(venduto bastoncini offerta)',figsize=(21,14))
# plt.show()






# sql = "Select data_format_date,sum(qta_non_offerta) from dump where cod_prod = '145644012' group by data_format_date"
# # cols = ['id','cod_cli_for', 'rag_soc', 'cod_prod', 'descr_prod', 'data_doc', 'data_format_date', 'qta_offerta', 'qta_non_offerta', 'val_off', 'val_non_off']
# prodotto = 'pandoro non'
# columns = ['data','sum(venduto '+prodotto+' offerta)']
# query_result = client.execute(sql, settings = {'max_execution_time' : 3600})
# df = pd.DataFrame(query_result, columns = columns)

# df.plot(x ='data', y='sum(venduto '+prodotto+' offerta)',figsize=(21,14))
# plt.show()





# sql = "Select data_format_date,sum(qta) from dump2 where cod_prod = '145644012' and data_doc>20210101 and data_doc<20211231 and flag_off=1 group by data_format_date"
# # cols = ['id','cod_cli_for', 'rag_soc', 'cod_prod', 'descr_prod', 'data_doc', 'data_format_date', 'qta_offerta', 'qta_non_offerta', 'val_off', 'val_non_off']
# prodotto = 'pandoro'
# columns = ['data','sum(venduto '+prodotto+' offerta)']
# query_result = client.execute(sql, settings = {'max_execution_time' : 3600})
# df = pd.DataFrame(query_result, columns = columns)

# df.plot(x ='data', y='sum(venduto '+prodotto+' offerta)',figsize=(21,14))
# plt.show()





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



sql = "Select data_format_date,data_doc,sum(qta) from dump2 where cod_prod = '145644015' and flag_off=0 group by data_format_date,data_doc order by data_doc"
# print(sql)#BAULI PANETTONE TRADIZION. KG1
query_result = client.execute(sql, settings = {'max_execution_time' : 3600})

cols = ['data_format_date','data_doc','sum']
df = pd.DataFrame(query_result)
df.columns = cols

# preparare il dataframe
# devo inserire nelle date mancanti, un valore di vendita = 0
data_format = df['data_format_date'].to_numpy()
data_doc = df['data_doc'].to_numpy()
sold = df['sum'].to_numpy()

# filling data_format
# data_format_full = np.arange(data_format[0], data_format[len(data_format)-1])
data_format_full = np.arange(0, 364)
# filling data_doc
data_doc_full = date_to_int(data_format_full)

# filling sold
sold_full = []
j = 0
for curr_date in data_format_full:
    if(curr_date == data_format[j]):
        sold_full.append(sold[j])
        j = j + 1
    else:
        sold_full.append(0)
# myplot(data_format_full,sold_full)




# converting to numpy arrays
X = np.array(data_doc_full).reshape((-1,1))
y = np.array(sold_full)
# print(y)
# myplot(data_format_full,y)

#training model
model = LinearRegression().fit(X,y)

# print(type(datetime.date(2022,1,1)))
# print(datetime.date(2022,1,1))

data_format_pred = np.arange(datetime.date(2022,1,1),datetime.date(2022,12,31))
data_doc_pred = date_to_int(data_format_pred)
X_pred = np.array(data_doc_pred).reshape((-1,1))
y_pred = model.predict(X)


myplot(data_format_pred,y_pred)






# start_date = data_format[0]
# print(start_date)

# number_of_days = 5

# date_list = []
# for day in range(number_of_days):
#   a_date = (start_date + datetime.timedelta(days = day)).isoformat()
#   date_list.append(a_date)

# print(my_to_numpy(date_list))

# print(data_format)

# ax=df.plot(x=cols[0], y=cols[2],figsize=(28,21))
# ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
# plt.grid(True)
# plt.gcf().autofmt_xdate()

# y = df[cols[2]].to_numpy() #somma venduto
# x = df[cols[1]].to_numpy().reshape((-1,1)) #date numeriche
# x_format = df[cols[0]].to_numpy()

# plt.plot(x_format, y, 'o', color='black')
# plt.show()


# model = LinearRegression().fit(x,y)

# r_sq = model.score(x, y)
# print(f"coefficient of determination: {r_sq}")
# print(f"intercept: {model.intercept_}")
# print(f"slope: {model.coef_}")


# xPred_format = pd.date_range(start="2022-01-01",end="2022-12-31")


# xPred = []
# for a in xPred_format:
#     xPred.append(to_integer(a))

# df2 = pd.DataFrame(xPred, columns=['data_doc'])
# xPred_numpy = df2['data_doc'].to_numpy().reshape((-1,1))
# xPred_format_numpy = df2['data_format_date'].to_numpy()

# print(x)
# print(xPred_numpy)

# y_pred = model.predict(xPred_numpy)
# print(f"predicted response:\n{y_pred}")

# plt.plot(xPred_numpy, y_pred, 'o', color='black')
# plt.show()
