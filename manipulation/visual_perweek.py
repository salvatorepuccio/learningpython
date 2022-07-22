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
# | 133208508     TENDERLY CARTA IG.KILOMETR.X4  |
# 148510080     BIRRA FUTTITINNI CL.33
# 148500585     BIRRA CERES CL.33X4
# 148502298     PORETTI BIRRA 4LUPPOLI CL.66X6
#148500404     PERONI BIRRA CL.33X6 
# 148500405     PERONI BIRRA CL.33X3
# 148500025     MESSINA BIRRA CL.33X3          │
# │ 148500027     ICHNUSA BIRRA N.F CL.50
# 148500793     BIRRA BECK'S CL.33   |
#  ----------------------------------------------


t = time.time()

data_start = '2020-01-01'
data_stop = '2020-12-31'
s = '\
148500405     PERONI BIRRA CL.33X3'
s_split = s.split("   ")
cod_prod = s_split[0]
descr_prod  = s_split[1]


sql_prod = "Select data_format_date,sum(qta_offerta+qta_non_offerta),sum(val_off+val_non_off) from dump where cod_prod='"+cod_prod+"' and data_format_date>='"+data_start+"' and data_format_date<='"+data_stop+"' group by data_format_date order by data_format_date"
sql_tutti = "Select data_format_date,sum(qta_offerta+qta_non_offerta),sum(val_off+val_non_off) from dump where data_format_date>='"+data_start+"' and data_format_date<='"+data_stop+"' group by data_format_date order by data_format_date"
sql_descr = "Select data_format_date,sum(qta_offerta+qta_non_offerta),sum(val_off+val_non_off) from dump where descr_prod like '%BIRRAligh%' and data_format_date>='"+data_start+"' and data_format_date<='"+data_stop+"' group by data_format_date order by data_format_date"


sql = sql_tutti #<-----scegliere

print("query: ",sql)
query_result = client.execute(sql, settings = {'max_execution_time' : 3600})
print("Righe",len(query_result))
if len(query_result) == 0:
    print("Query vuota!")
    stop()

cols = ['data_format_date','qta','val']
df = pd.DataFrame(query_result)
df.columns = cols



# Somma le righe duplicate, ovvero con la stessa data
df = somma_duplicate(df)
righe_non_nulle = len(df)


# Aggiunge una riga per ogni data mancante all'interno del range della query, con qta = 0
df = fill_dataframe_visual(df,data_start,data_stop,[0,0])
righe_totali = len(df)

# trunc = lambda x: math.trunc(1000 * x) / 1000
# df['val'].applymap(trunc)

df['val'] = ((df['val']*1000).astype(int).astype(float))/1000

day_of_week = get_day_in_week(df['data_format_date']) # num [0,6] lun,mar,mer....dom
week_n = get_week_in_year(df['data_format_date']) # num [0,52] settimana dell'anno
df.insert(len(df.columns), 'day_of_week', day_of_week, True)
df.insert(len(df.columns), 'week_n', week_n, True)

print_all(df)

fig, ax = plt.subplots()
cmap0 = LinearSegmentedColormap.from_list('', ['white', 'darkblue'])
ax.scatter(week_n,day_of_week,c=df['val'],s=400,cmap=cmap0,marker='s')
# ax.tricontourf(day_of_wee k,week_n,df['val'],256)
# plt.grid(linewidth=0.01)
plt.yticks(day_of_week)
plt.xticks(week_n)
ax.set_xlim((0,54))
ax.set_ylim((0,8))
x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()
titolo = "Da "+data_start+" a "+data_stop+" "
if sql == sql_tutti:
    titolo+= "TUTTI i prodotti"
if sql == sql_prod:
    titolo+= "Prodotto: "+descr_prod
plt.title(titolo)
fig.subplots_adjust(bottom=0.67)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()                    