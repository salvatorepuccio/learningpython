# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from clickhouse_driver import Client
from functions import *
from sklearn.model_selection import train_test_split 
import lightgbm as lgb

client = Client(
                host='127.0.0.1',
                user='savato',
                password='savato',
                database='chpv',
                )
client.execute("use chpv")

data_start = '2020-01-01'
data_stop = '2021-12-31'
s = '\
148500405     PERONI BIRRA CL.33X3'
s_split = s.split("   ")
cod_prod = s_split[0]
descr_prod  = s_split[1]

print('Loading data...')
# load or create your dataset
sql_prod = "Select data_format_date,sum(qta_offerta+qta_non_offerta),sum(val_off+val_non_off) from dump where cod_prod='"+cod_prod+"' and data_format_date>='"+data_start+"' and data_format_date<='"+data_stop+"' group by data_format_date order by data_format_date"
sql_tutti = "Select data_format_date,sum(qta_offerta+qta_non_offerta),sum(val_off+val_non_off),cod_prod from dump where data_format_date>='"+data_start+"' and data_format_date<='"+data_stop+"' group by data_format_date order by data_format_date,cod_prod"
sql_descr = "Select data_format_date,sum(qta_offerta+qta_non_offerta),sum(val_off+val_non_off) from dump where descr_prod like '%BIRRAligh%' and data_format_date>='"+data_start+"' and data_format_date<='"+data_stop+"' group by data_format_date order by data_format_date"
sql = sql_prod #<-----scegliere
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
df = fill_dataframe_visual(df,cols,data_start,data_stop)
righe_totali = len(df)  

# trunc = lambda x: math.trunc(1000 * x) / 1000
# df['val'].applymap(trunc)

df['val'] = ((df['val']*1000).astype(int).astype(float))/1000

day_of_week = get_day_in_week(df['data_format_date'])# num [0,6] lun,mar,mer....dom
week_n = get_week_in_year(df['data_format_date'])# num [0,52] settimana dell'anno
unit_price = get_unit_price(df[['qta','val']])# prezzo unitario del prodotto
day_of_year = get_day_in_year(df['data_format_date'])# num [0,365] num giorno nell'anno
day_of_month = get_day_in_month(df['data_format_date'])# num [0,30] num giorno del mese

df.insert(len(df.columns), 'day_of_month', day_of_month, True)
df.insert(len(df.columns), 'day_of_week', day_of_week, True)
df.insert(len(df.columns),'day_of_year', day_of_year, True)
df.insert(len(df.columns), 'week_n', week_n, True)
df.insert(len(df.columns),'unit_price', unit_price, True)

y = df['qta'].to_numpy()
X = df[['day_of_month','day_of_week','day_of_year','week_n','unit_price']].to_numpy()
# X = df[['day_of_year','unit_price','flag_off']].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=False,test_size=0.20)
print("Train size: ",len(X_train))
print("Test size: ",len(X_test))

# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------

print('\nStarting training...')
# train
gbm = lgb.LGBMRegressor(num_leaves=31, #31
                        learning_rate=0.05, #0.05
                        n_estimators=20) #20

gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        callbacks=[lgb.early_stopping(5)])

print('\nStarting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# eval
rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
print(f'The RMSE of prediction is: {rmse_test}')

# feature importances
print(f'Feature importances: {list(gbm.feature_importances_)}')

print("\nPlotting...")
myplot2(df['data_format_date'],y,y_pred,"")


# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------


# self-defined eval metric
# f(y_true: array, y_pred: array) -> name: str, eval_result: float, is_higher_better: bool
# Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_true, y_pred):
    return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False


print('\nStarting training with custom eval function...')
# train
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=rmsle,
        callbacks=[lgb.early_stopping(5)])

print('\nStarting predicting...')
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

print("\nPlotting...")
myplot2(df['data_format_date'],y,y_pred,"")


# another self-defined eval metric
# f(y_true: array, y_pred: array) -> name: str, eval_result: float, is_higher_better: bool
# Relative Absolute Error (RAE)
def rae(y_true, y_pred):
    return 'RAE', np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(np.mean(y_true) - y_true)), False


print('\nStarting training with multiple custom eval functions...')
# train
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=[rmsle, rae],
        callbacks=[lgb.early_stopping(5)])

print('\nStarting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# eval
rmsle_test = rmsle(y_test, y_pred)[1]
rae_test = rae(y_test, y_pred)[1]
print(f'The RMSLE of prediction is: {rmsle_test}')
print(f'The RAE of prediction is: {rae_test}')

print("\nPlotting...")
myplot2(df['data_format_date'],y,y_pred,"")

# other scikit-learn modules
estimator = lgb.LGBMRegressor(num_leaves=31)

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1, 1],
    'n_estimators': [20, 40, 60, 80, 100]
}

gbm = GridSearchCV(estimator, param_grid, cv=3)
gbm.fit(X_train, y_train)

print(f'Best parameters found by grid search are: {gbm.best_params_}')

print("\nPlotting...")
myplot2(df['data_format_date'],y,y_pred,"")

# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------
