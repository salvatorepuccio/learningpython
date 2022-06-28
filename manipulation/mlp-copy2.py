from audioop import avg
from logging.handlers import DEFAULT_UDP_LOGGING_PORT
from turtle import clear
from attr import attributes
from clickhouse_driver import Client
import cupshelpers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split 
import matplotlib.dates as mdates
from datetime import datetime
import math
import time
from sklearn import metrics
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning
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
def fill_dataframe(df,df_columns):
    full_calendar = fill_dates(str(df['data_format_date'][0]),str(df['data_format_date'][len(df)-1]))
    j = 0
    N_date_mancanti = 0
    for curr_date in full_calendar:
        
        if(str(curr_date).split(" ")[0] == str(df['data_format_date'][j])):
            # Se la data corrente e' una di quelle in cui ho fatto una vendita, vai avanti
            j = j + 1
        else:
            # altrimenti metti una riga che indice che non ho venduto nulla
            new_row = [curr_date,0,0,-1]#['data_format_date','qta','val','flag_off']
            dff = pd.DataFrame([new_row],columns=df_columns)
            df = pd.concat([df,dff])
            N_date_mancanti += 1
    
    df["data_format_date"] = pd.to_datetime(df["data_format_date"])
    df = df.sort_values(by="data_format_date")
    # print("Mancavano "+str(N_date_mancanti)+" date")
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
    # print("Eliminate "+str(eliminate)+" duplicazioni")
    df = df.reset_index(drop=True)
    return df
        
def myplot2(x,y,y1,title):
    fig, ax = plt.subplots()
    ax = plt.gca() 
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.WeekdayLocator(interval=4)
    ax.xaxis.set_major_locator(locator)
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.title(title)
    ax.plot(x,y,linewidth=1)
    ax.plot(x[-len(y1):],y1,color='red',linewidth=1)
    
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
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
    ax.plot(x,y,linewidth=1)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
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

def get_avg_price(df_qta_val):
    sum = 0
    avg = 0
    n = 0
    for i in range(0,len(df_qta_val)):
        if(df_qta_val['qta'][i]>0):
            n+=1
            sum += df_qta_val['val'][i] / df_qta_val['qta'][i]
    # print("Somma dei prezzi unitari: "+str(sum))

    avg = sum / n
    # print("Media dei prezzi unitari: "+str(avg))
    return avg

def get_unit_price(df_qta_val):
    avg_price = get_avg_price(df_qta_val)
    ret = []
    unit_price=0
    for i in range(0,len(df_qta_val)):
        if(df_qta_val['qta'][i]==0):
            unit_price = 0
        else:
            unit_price = df_qta_val['val'][i] / df_qta_val['qta'][i]
        ret.append(unit_price)
        # print("qta "+str(df_qta_val['qta'][i])+" val "+str(df_qta_val['val'][i])+" prezzo unitario "+str(append_me))
    return ret

def get_day_of_month(df_date):
    ret = []
    for x in df_date:
        ret.append(int(x.strftime("%d")))
    return ret

def convert_flag_to_bool(df_flag):
    ret = []
    for x in df_flag:
        if(str(x)== str(0)):
            ret.append(False)
        elif(str(x) == str(1)):
            ret.append(True)
        else:
            print("Flag: "+str(x))
    return ret

def stophere():
    raise SystemExit(0)  

client = Client(
                host='127.0.0.1',
                user='savato',
                password='savato',
                database='chpv',
                )
client.execute("use chpv")

#  ______________________________________________
# | 000249036     PARMAREG.L'ABC MERENDA IVA 10% │
# │ 000249037     PARMAREG.L'ABC MERENDA IVA 22% ││
# │ 080112709     PRORASO SCH-BARBA P.SENS.ML300    [OK] │
# │ 080119200     MASCARA THE ROCKET VERY BLACK  │
# │ 080123754     FIGARO SCH.BARBA SENSITIVE 400 │
# │ 080124800     I PROVENZALI SAP.LIQ MAND.DOL  |
# | 163854067     FINDUS 30 BASTONCINI MERL G750 |
# | 148520655     VITASNELLA ACQUA CL.50         |
# | 148520098     S.BENEDETTO ACQUA LT.2         |
# | 148520189     ULIVETO ACQUA EFFERV.LT.1.5   [OK] |
# | 155205436     BARILLA PENNE RIG.5 CER.GR.400    [OK]
# 163858005     AMADORI BASTONCINI POLLO G.280 
# 145645009     PALUANI PANDORO CR.CIOC.GR.750
# 145644117     MAINA PANDORO CONF.MOMENTI MAG │
#│ 145644127     MAINA PANDORO CIOCCOLOTT.GR750
# 125621076     DIESSE PIATTI FONDI GR.700|
#140310420     KINDER CARDS LATTE/CAC. GR.128
#148500405     PERONI BIRRA CL.33X3
#  ----------------------------------------------
iperstores = ['IPERSTORE 01','IPERSTORE 02', 'IPERSTORE 03', 'IPERSTORE 04']
superstores = ['SUPERSTORE 01','SUPERSTORE 02','SUPERSTORE 03','SUPERSTORE 04','SUPERSTORE 05','SUPERSTORE 06','SUPERSTORE 07','SUPERSTORE 08','SUPERSTORE 09','SUPERSTORE 10','SUPERSTORE 11','SUPERSTORE 12','SUPERSTORE 13','SUPERSTORE 14','SUPERSTORE 15']
supermarkets = ['SUPERMARKET 01','SUPERMARKET 02','SUPERMARKET 03','SUPERMARKET 04','SUPERMARKET 05','SUPERMARKET 06','SUPERMARKET 07','SUPERMARKET 08','SUPERMARKET 09','SUPERMARKET 10','SUPERMARKET 11','SUPERMARKET 12','SUPERMARKET 13','SUPERMARKET 14']
categories = [iperstores,superstores,supermarkets]
prodotto = "PERONI BIRRA CL.33X3"
t = time.time()
for category in categories:

    scores = []
    n_stores = 0
    for current_store in category:
        print("")
        print(current_store)
        
        sql = "Select data_format_date,qta,val,flag_off from dump2 where cod_prod='148500405' and rag_soc = '"+current_store+"' group by data_format_date,val,flag_off,qta order by data_format_date"
        query_result = client.execute(sql, settings = {'max_execution_time' : 3600})
        
        if(len(query_result)>0):
            
            n_stores+=1
            cols = ['data_format_date','qta','val','flag_off']
            df = pd.DataFrame(query_result)
            df.columns = cols

            # Somma le righe duplicate, ovvero con la stessa data
            df = somma_duplicate(df)
            righe_non_nulle = len(df)
            
            # Aggiunge una riga per ogni data mancante all'interno del range della query, con qta = 0
            df = fill_dataframe(df,cols)
            righe_totali = len(df)

            # Plot delle vendite (0 compresi) nel periodo della query
            # myplot(df['data_format_date'],df['qta'])
            
            # Estrazione nuove info dai dati esistenti
            day_of_week = get_days(df['data_format_date'])# num [0,6] lun,mar,mer....dom
            week_n = get_week(df['data_format_date'])# num [0,52] settimana dell'anno
            unit_price = get_unit_price(df[['qta','val']])# prezzo unitario del prodotto
            day_of_year = get_day_of_the_year(df['data_format_date'])# num [0,365] num giorno nell'anno
            day_of_month = get_day_of_month(df['data_format_date'])# num [0,30] num giorno del mese
            # flag_off_bool = convert_flag_to_bool(df['flag_off'])# flag offerta True, False invece di 1,0

            # Inserimento nuove colonne
            df.insert(0, 'day_of_month', day_of_month, True)
            df.insert(0, 'day_of_week', day_of_week, True)
            df.insert(0,'day_of_year', day_of_year, True)
            df.insert(0, 'week_n', week_n, True)
            df.insert(0,'unit_price', unit_price, True)
            # df.insert(0, 'flag_off_bool', flag_off_bool, True)

            y = df['qta'].to_numpy()
            X = df[['day_of_month','day_of_week','day_of_year','week_n','unit_price','flag_off']].to_numpy()
            # X = df[['day_of_year','unit_price','flag_off']].to_numpy()
        
            X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=False,test_size=0.20)
            # print("Righe valide in input: ",righe_non_nulle)
            # print(str(tuple))

            activators = ['logistic', 'tanh', 'relu']
            solvers = ['lbfgs','sgd', 'adam']
            learning_rates = ['constant', 'invscaling', 'adaptive']

            mlp_gs = MLPRegressor()
            parameter_space = {
                'hidden_layer_sizes': range(10,100,10),
                # [(10,),(20,),(30,),(40,),(50,),(60,),(70,),(80,),(90,),(100,)],
                'activation': activators,
                'solver': solvers,
                'learning_rate': learning_rates,
                'max_iter' : range(100,300,50),
                'random_state' : [35] 
            }
            from sklearn.model_selection import GridSearchCV
            t = time.time()
            clf = GridSearchCV(mlp_gs,parameter_space, n_jobs=-1, cv=5,refit=True,scoring='neg_median_absolute_error')
            clf.fit(X_train, y_train) # X is train samples and y is the corresponding labels
            y_pred = clf.predict(X_test)
            y_pred = y_pred.clip(min=0)
            score = clf.score(X_test,y_test)

            print('\n\n\n\nBest parameters found:\n', clf.best_params_)
            print("neg_median_absolute_error")
            print("Score: ",score)
            input("Tempo %.2f s\n" % (time.time() - t))

            # regr = MLPRegressor(
            #     activation=clf.best_params_['activation'],
            #     solver=clf.best_params_['solver'],
            #     hidden_layer_sizes=clf.best_params_['hidden_layer_sizes'],
            #     max_iter=clf.best_params_['max_iter'],
            #     random_state=35,
            #     learning_rate=clf.best_params_['learning_rate']
            #     ).fit(X_train, y_train)

            # y_pred = regr.predict(X_test)
            # y_pred = y_pred.clip(min=0)

            # Calcolo score
            # score = regr.score(X_test, y_test)

             # Imposto titolo
            title = "Prod: "+prodotto+" Store: "+current_store+" score: "+str(score)
            # if(score>0):
            #     scores.append(score)
            # print("Score: ",np.round(score, 2))

            # Mostro i risultati
            myplot2(df['data_format_date'],y,y_pred,title)
            print("\n") 

            # means = clf.cv_results_['mean_test_score']
            # stds = clf.cv_results_['std_test_score']
            # # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            # #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

            # y_test, y_pred = y_test , clf.predict(X_test)
            # from sklearn.metrics import classification_report
            # print('Results on the test set:')
            # print(classification_report(y_test, y_pred))

            
            # print("PESI: "+str(regr.coefs_[0][0]))

            # Predico su X_test
            # y_pred = regr.predict(X_test)
            # y_pred = y_pred.clip(min=0)

            # score = regr.score(X_test, y_test)
            # print("Score: ",score)
                    
            # # Imposto titolo
            # title = "Prod: "+prodotto+" Store: "+current_store+" score: "+str(score)
            # if(best_score_for_iter>0):
            #     scores.append(best_score_for_iter)
            # # print("Score: ",np.round(score, 2))

            # # Mostro i risultati
            # myplot2(df['data_format_date'],y,y_pred,title)
            print("\n")                   

        else:
            print("\n"+current_store+" non vende "+prodotto+"\n")                    


            # print("Hidden Layers: "+str(hidden_layers)+
            # " Neurons per layer: "+str(neurons_per_layer)+
            # " activator "+activator+"\nScore: "+score)

            # Tronco i valori predetti
            # y_pred = np.trunc(y_pred)
            # y_test = np.trunc(y_test)
            # print(y_test)
            # print(y_pred)
            
                        
        
print("Media score: "+str(sum(scores) / n_stores))
print("Tempo %.2f s\n" % (time.time() - t))