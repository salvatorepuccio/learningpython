from audioop import avg
from calendar import week
from logging.handlers import DEFAULT_UDP_LOGGING_PORT
from turtle import clear
from attr import attributes
from clickhouse_driver import Client
import cupshelpers
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split 
import matplotlib.dates as mdates
import time
import warnings
from functions import *
import math
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
# prende una data di inizio e una data di fine (stringhe) e restituisce una lista di tutti i giorni in mezzo

client = Client(
                host='127.0.0.1',
                user='savato',
                password='savato',
                database='chpv',
                )
client.execute("use chpv")

#  ______________________________________________
# | 000249036     PARMAREG.L'ABC MERENDA IVA 10% │
# │ 000249037     PARMAREG.L'ABC MERENDA IVA 22% │
# │ 080112709     PRORASO SCH-BARBA P.SENS.ML300 │ [OK]
# │ 080119200     MASCARA THE ROCKET VERY BLACK  │
# │ 080123754     FIGARO SCH.BARBA SENSITIVE 400 │
# │ 080124800     I PROVENZALI SAP.LIQ MAND.DOL  |
# | 163854067     FINDUS 30 BASTONCINI MERL G750 |
# | 148520655     VITASNELLA ACQUA CL.50         |
# | 148520098     S.BENEDETTO ACQUA LT.2         |
# | 148520189     ULIVETO ACQUA EFFERV.LT.1.5    | [OK]
# | 155205436     BARILLA PENNE RIG.5 CER.GR.400 | [OK]
# | 163858005     AMADORI BASTONCINI POLLO G.280 | 
# | 145645009     PALUANI PANDORO CR.CIOC.GR.750 |
# | 145644117     MAINA PANDORO CONF.MOMENTI MAG │
# │ 145644127     MAINA PANDORO CIOCCOLOTT.GR750 |
# | 125621076     DIESSE PIATTI FONDI GR.700     |
# | 140310420     KINDER CARDS LATTE/CAC. GR.128 |    
# │ 133300503     AMUCHINA GEL MANI XGERM ML.80  │  
# | 148500405     PERONI BIRRA CL.33X3           |
#  ----------------------------------------------


t = time.time()

data_start = '2020-01-01'
data_stop = '2021-12-31'
s = '\
133300503     AMUCHINA GEL MANI XGERM ML.80'
s_split = s.split("   ")
cod_prod = s_split[0]
descr_prod  = s_split[1]

sql_prod = "Select data_format_date,sum(qta_offerta+qta_non_offerta),sum(val_off+val_non_off) from dump where cod_prod='"+cod_prod+"' and data_format_date>='"+data_start+"' and data_format_date<='"+data_stop+"' group by data_format_date order by data_format_date"
sql_tutti = "Select data_format_date,sum(qta_offerta+qta_non_offerta),sum(val_off+val_non_off) from dump where data_format_date>='"+data_start+"' and data_format_date<='"+data_stop+"' group by data_format_date order by data_format_date"

ricerca_carni = "(descr_prod like '%BISTECCA%' or descr_prod like '%SCOTTONA%')"
ricerca_carbonella = "(descr_prod like '%CARBONE SACCO%' or descr_prod like '%CARBONELLA%' or descr_prod like '%CARBONE x BBQ FOCHISTA%' or descr_prod like '%EAZY BBQ CARBONELLA DI QUERCIA%')"
ricerca_chips = "(descr_prod like '%PATATINE%' and descr_prod not like '%PATATINE FRITTE%')"
ricerca_cola = "(descr_prod like '%COCA COLA PET LT.1,5%' or descr_prod like '%COCA COLA LT.2#%' or descr_prod like '%COCA COLA PET LT.1.35#%' or descr_prod like '%COCA COLA PET LT.1 #%' or descr_prod like '%COCA COLA LT.1,5x2%')"

sql_descr = "SELECT data_format_date,sum(qta_offerta+qta_non_offerta),sum(val_off+val_non_off) from dump where "+ricerca_chips+" and data_format_date>='"+data_start+"' and data_format_date<='"+data_stop+"' group by data_format_date order by data_format_date"
sql_descr2 = "Select data_format_date,sum(qta_offerta+qta_non_offerta),sum(val_off+val_non_off) from dump where "+ricerca_cola+" and data_format_date>='"+data_start+"' and data_format_date<='"+data_stop+"' group by data_format_date order by data_format_date"


sql = sql_descr #<-----scegliere
sql2 = sql_descr2

print("query1: ",sql)
print("query2: ",sql2)
query_result = client.execute(sql, settings = {'max_execution_time' : 3600})
print("Righe",len(query_result))
if len(query_result) == 0:
    print("Query vuota!")
    stop()

query_result2 = client.execute(sql2, settings = {'max_execution_time' : 3600})
print("Righe",len(query_result2))
if len(query_result2) == 0:
    print("Query vuota!")
    stop()



cols = ['data_format_date','qta','val']
df = pd.DataFrame(query_result)
df2 = pd.DataFrame(query_result2)
df.columns = cols
df2.columns = cols

# Somma le righe duplicate, ovvero con la stessa data
df = somma_duplicate(df)
df2 = somma_duplicate(df2)



# Aggiunge una riga per ogni data mancante all'interno del range della query, con qta = 0
df = fill_dataframe_visual(df,data_start,data_stop,[0,0])
df2 = fill_dataframe_visual(df2,data_start,data_stop,[0,0])

# trunc = lambda x: math.trunc(1000 * x) / 1000
# df['val'].applymap(trunc)

df['val'] = ((df['val']*1000).astype(int).astype(float))/1000
df2['val'] = ((df2['val']*1000).astype(int).astype(float))/1000

df['qta'] = ((df['qta']*1000).astype(int).astype(float))/1000
df2['qta'] = ((df2['qta']*1000).astype(int).astype(float))/1000

day_of_week = get_day_in_week(df['data_format_date']) # num [0,6] lun,mar,mer....dom
week_n = get_week_in_year(df['data_format_date']) # num [0,52] settimana dell'anno
df.insert(len(df.columns), 'day_of_week', day_of_week, True)
df.insert(len(df.columns), 'week_n', week_n, True)

myplot_multi(df['data_format_date'],df['qta'],df2['qta'],"chips","cola",True)   
myplot2(df["data_format_date"],df['qta'],df2['qta'],"blu: chips , rosso: cola")               