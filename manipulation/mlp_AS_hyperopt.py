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
import time
import warnings
from functions import *
warnings.filterwarnings("ignore")
# prende una data di inizio e una data di fine (stringhe) e restituisce una lista di tutti i giorni in mezzo

def objective(n_estimators):
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_features='sqrt',
                                   random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return {'loss': -accuracy, 'status': STATUS_OK}




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


iperstores = [
    'IPERSTORE 01',
    'IPERSTORE 02', 
    'IPERSTORE 03', 
    'IPERSTORE 04'
    ]
superstores = [
    'SUPERSTORE 01',
    'SUPERSTORE 02',
    'SUPERSTORE 03',
    'SUPERSTORE 04',
    'SUPERSTORE 05',
    'SUPERSTORE 06',
    'SUPERSTORE 07',
    'SUPERSTORE 08',
    'SUPERSTORE 09',
    'SUPERSTORE 10',
    'SUPERSTORE 11',
    'SUPERSTORE 12',
    'SUPERSTORE 13',
    'SUPERSTORE 14',
    'SUPERSTORE 15'
    ]
supermarkets = [
    'SUPERMARKET 01',
    'SUPERMARKET 02',
    'SUPERMARKET 03',
    'SUPERMARKET 04',
    'SUPERMARKET 05',
    'SUPERMARKET 06',
    'SUPERMARKET 07',
    'SUPERMARKET 08',
    'SUPERMARKET 09',
    'SUPERMARKET 10',
    'SUPERMARKET 11',
    'SUPERMARKET 12',
    'SUPERMARKET 13',
    'SUPERMARKET 14'
    ]
categories = [
    iperstores,
    superstores,
    supermarkets
    ]
prodotto = "PERONI BIRRA CL.33X3"
t = time.time()
for category in categories:

    scores = []
    n_stores = 0
    for current_store in category:
        print("")
        print(current_store)
        
        sql = "Select data_format_date,rag_soc,qta,val,flag_off from dump2 where cod_prod='148500405' group by data_format_date,rag_soc,qta,val,flag_off order by data_format_date"
        query_result = client.execute(sql, settings = {'max_execution_time' : 3600})
        
        if(len(query_result)>0):
            
            n_stores+=1
            cols = ['data_format_date','rag_soc','qta','val','flag_off']
            df = pd.DataFrame(query_result)
            df.columns = cols

            # Somma le righe duplicate, ovvero con la stessa data
            df = somma_duplicate(df)
            righe_non_nulle = len(df)
           
            
            # Aggiunge una riga per ogni data mancante all'interno del range della query, con qta = 0
            df = fill_dataframe_all_stores(df,cols)
            righe_totali = len(df)

            # Plot delle vendite (0 compresi) nel periodo della query
            # myplot(df['data_format_date'],df['qta'])
            
            # Estrazione nuove info dai dati esistenti
            day_of_week = get_day_in_week(df['data_format_date'])# num [0,6] lun,mar,mer....dom
            week_n = get_week_in_year(df['data_format_date'])# num [0,52] settimana dell'anno
            unit_price = get_unit_price(df[['qta','val']])# prezzo unitario del prodotto
            day_of_year = get_day_in_year(df['data_format_date'])# num [0,365] num giorno nell'anno
            day_of_month = get_day_in_month(df['data_format_date'])# num [0,30] num giorno del mese
            # rag_soc_conv = convert_and_get_rag_soc_to_int(df['rag_soc'])
            # flag_off_bool = convert_flag_to_bool(df['flag_off'])# flag offerta True, False invece di 1,0
            
            # df.drop('rag_soc',inplace=True, axis=1)

            # Inserimento nuove colonne
            df.insert(len(df.columns), 'day_of_month', day_of_month, True)
            df.insert(len(df.columns), 'day_of_week', day_of_week, True)
            df.insert(len(df.columns),'day_of_year', day_of_year, True)
            df.insert(len(df.columns), 'week_n', week_n, True)
            df.insert(len(df.columns),'unit_price', unit_price, True)
            # df.insert(len(df.columns), 'rag_soc', rag_soc_conv, True)
           
            # df.insert(len(df.columns), 'flag_off_bool', flag_off_bool, True)

            y = df['qta'].to_numpy()
            X = df[['day_of_month','day_of_week','day_of_year','week_n','unit_price','flag_off']].to_numpy()
            # X = df[['day_of_year','unit_price','flag_off']].to_numpy()
        
            X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=False,test_size=0.20)
            print("Train size: ",len(X_train))
            print("Test size: ",len(X_test))
            
            # print("Righe valide in input: ",righe_non_nulle)
            # print(str(tuple))

            activators = ['logistic', 'tanh', 'relu']
            solvers = ['lbfgs','sgd', 'adam']
            learning_rates = ['constant', 'invscaling', 'adaptive']

            # Se utilizzo lbfgs ho biosgno di scalare i dati
            # X_train = scala(X_train)

            mlp_gs = MLPRegressor(early_stopping=True,validation_fraction=0.3)
            parameter_space = {
                'hidden_layer_sizes': range(10,200,10),
                'activation': activators,
                'solver': ['adam'],
                'max_iter' : [2000],
                'random_state' : [42],
                "alpha" : [0.0001]
            }
            from sklearn.model_selection import GridSearchCV
            t = time.time()
            clf = GridSearchCV(mlp_gs,parameter_space, cv=15, n_jobs=-1,refit=True,scoring='neg_mean_squared_error')
            clf.fit(X_train, y_train)
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
            title = "Prod: "+prodotto+" Store: TUTTI "+" score: "+str(score)
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

        stop()
            
                        
        
print("Media score: "+str(sum(scores) / n_stores))
print("Tempo %.2f s\n" % (time.time() - t))