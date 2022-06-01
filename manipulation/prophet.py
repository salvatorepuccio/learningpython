from turtle import clear
from clickhouse_driver import Client
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates
import datetime
import math
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly

# prende una data di inizio e una data di fine (stringhe) e restituisce una lista di tutti i giorni in mezzo
def fill_dates(date_start,date_end):
    # filling data_format
    # data_format_full = np.arange(data_format[0], data_format[len(data_format)-1])
    start = datetime.datetime.strptime(date_start, "%Y-%m-%d")
    end = datetime.datetime.strptime(date_end, "%Y-%m-%d")
    return pd.date_range(start, end)

def to_day_of_the_year(l):
    l1 = []
    for val in l:
        l1.append(val.timetuple().tm_yday)
    return l1


# mette degli 0 nella date in cui non ci sono state vendite
def fill_sells(full_calendar,mydays):
    sells_full = []
    j = 0
    for curr_date in full_calendar:
        # print(str(curr_date).split(" ")[0]+"   "+ str(mydays[j])+" j: "+str(j))
        # Se la data corrente e' una di quelle in cui ho fatto una vendita
        if(str(curr_date).split(" ")[0] == str(mydays[j])):
            # inserisci il venduto
            sells_full.append(sold[j])
            j = j + 1
        else:
            # altrimenti metti 0
            sells_full.append(0)

    return sells_full


# https://www.diva-portal.org/smash/get/diva2:1108597/FULLTEXT01.pdf

client = Client(
                host='127.0.0.1',
                user='savato',
                password='savato',
                database='chpv',
                )
client.execute("use chpv")

# prod = "163854067 FINDUS 30 BASTONCINI MERL G750"
# prod = "BAULI PANETTONE TRADIZION. KG1 145644015"
# prod = "148520189 ULIVETO ACQUA EFFERV.LT.1.5"
prod = "133207998     SCOTTEX CARTA IGIENICA X10"
sql = "Select data_format_date,sum(qta) from dump2 where cod_prod = '133207998' and flag_off=0 group by data_format_date order by data_format_date"
query_result = client.execute(sql, settings = {'max_execution_time' : 3600})

# cols = ['cod_cli_for', 'rag_soc', 'data_format_date', 'flag_off', 'sum']
# df = pd.DataFrame(query_result)

# df.columns = cols
# df = df[["data_format_date","sum"]]
# df = df.rename(columns = {"data_format_date":"ds","sum":"y"})
# # df['ds'] = fill_dates(df['ds'][0],df['ds'][len(df)-1])
# ds = pd.date_range(df['ds'][0], df['ds'][len(df)-1])

# m = Prophet(yearly_seasonality = True) # the Prophet class (model)
# m.fit(df) # fit the model using all data


# future = m.make_future_dataframe(periods=365) #we need to specify the number of days in future
# forecast = m.predict(future)
# # forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# fig1 = m.plot(forecast)
# plt.show()
# fig2 = m.plot_components(forecast)
# plot_plotly(m, forecast).show()

# plot_components_plotly(m, forecast)


# m.plot(forecast)
# plot_components_plotly(m, forecast)
# plt.title("forecast of "+prod+" sells using the Prophet")
# plt.xlabel("Date")
# plt.ylabel("Close Stock Price")
# plt.show()






cols = ['data_format_date','sum']
df = pd.DataFrame(query_result)
df.columns = cols
# devo inserire nelle date mancanti, un valore di vendita = 0
data_format = df['data_format_date'].to_numpy()
# print(data_format)
# data_doc = df['data_doc'].to_numpy()
sold = df['sum'].to_numpy()

# fill dates
data_format_full = fill_dates(str(data_format[0]),str(data_format[len(data_format)-1]))
# filling data_doc
# data_index_full = to_day_of_the_year(data_format_full)

# fill sells
sells_full = fill_sells(data_format_full,data_format)

# myplot(data_format_full,sells_full)

data = pd.DataFrame(list(zip(data_format_full, sells_full)),columns=["ds","y"])

m = Prophet(yearly_seasonality = True) # the Prophet class (model)
m.fit(data) # fit the model using all data

future = m.make_future_dataframe(periods=365) #we need to specify the number of days in future
forecast = m.predict(future)
# # forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
plt.show()
fig2 = m.plot_components(forecast)
plt.show()
# plot_plotly(m, forecast).show()