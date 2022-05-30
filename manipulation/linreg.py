from clickhouse_driver import Client
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates




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





prodotto = 'WURSTEL'
data_inizio = '2021-11-01'
data_fine = '2022-01-31'


sql = "Select data_format_date,sum(qta) from dump2 where data_format_date>'"+data_inizio+"' and data_format_date<'"+data_fine+"' and descr_prod like '%"+prodotto+"%' and flag_off=0 group by data_format_date limit 50000000"
# sql = "Select data_doc,sum(qta) from dump2 where cod_prod = '145644015' and flag_off=0 group by data_doc"
print(sql)
query_result = client.execute(sql, settings = {'max_execution_time' : 3600})

cols = ['data','sum '+prodotto+' offerta']
df = pd.DataFrame(query_result)
df.columns = cols
# print(df)


print("Query result lines: "+str(len(df.index)))


if len(df.index) > 0:

    ax=df.plot(x=cols[0], y=cols[1],figsize=(20,20))
    # ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    # plt.grid(True)
    # plt.gcf().autofmt_xdate() 
    plt.title("SOMMA VENDITE PRODOTTI CONTENGONO '"+prodotto+"' IN DESCRIZIONE\nDAL "+data_inizio+" A "+data_fine+" ",fontsize=24)
    x = df[cols[0]].to_numpy()
    y = df[cols[1]].to_numpy()
    ax=plt.plot(x, y, 'o', color='black')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.grid(True)
    plt.gcf().autofmt_xdate() 
    plt.show()





# sql = "Select data_doc,sum(qta) from dump2 where cod_prod = '145644015' and flag_off=0 group by data_doc"
# print(sql)#BAULI PANETTONE TRADIZION. KG1
# query_result = client.execute(sql, settings = {'max_execution_time' : 3600})

# cols = ['data_doc','sum '+prodotto+' offerta']
# df = pd.DataFrame(query_result)
# df.columns = cols;

# ax=df.plot(x =cols[0], y=cols[1],figsize=(28,21))
# ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
# plt.grid(True)
# plt.gcf().autofmt_xdate()
# x = pd.to_datetime(df[cols[0]], format='%Y%m%d')
# # print(x)
# # df.set_index(cols[0])#data
# # x = df[cols[0]].to_numpy().reshape((-1,1))#TRASFOMRA LE DATE IN INDICI, MA NON VA BENE
# y = df[cols[1]].to_numpy()#somma venduto (da prevedere)

# plt.plot(x, y, 'o', color='black');
# # plt.show()

# # print(x)
# # print(y)








# model = LinearRegression().fit(x,y)
# r_sq = model.score(x, y)
# print(f"coefficient of determination: {r_sq}")

# print(f"intercept: {model.intercept_}")
# intercept: 5.633333333333329

# print(f"slope: {model.coef_}")
    
# new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
# print(f"intercept: {new_model.intercept_}")

# print(f"slope: {new_model.coef_}")







# y_pred = model.predict(x_new)
# data_arrays = np.array(x_new,y_pred)
# df_pred = pd.DataFrame(data = data_arrays, columns = cols)
# print(df_pred)
# print(f"predicted response:\n{y_pred}")