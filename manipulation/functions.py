from turtle import clear
from attr import attributes
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from enum import Enum

from sqlalchemy import column

def get_calendar(date_start,date_end):
    # filling data_format
    dates = pd.date_range(date_start,date_end)
    ret = []
    for x in dates:
        ret.append(x.date())
    return ret

def get_day_in_year(l):
    l1 = []
    for val in l:
        l1.append(val.timetuple().tm_yday)
    return l1

# mette degli 0 nella date in cui non ci sono state vendite
def fill_dataframe(df,df_columns):
    full_calendar = get_calendar(str(df['data_format_date'][0]),str(df['data_format_date'][len(df)-1]))
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
    print("Mancavano "+str(N_date_mancanti)+" date")
    df = df.reset_index(drop=True)
    return df

def fill_dataframe_all_stores(df,df_columns):
    full_calendar = get_calendar(str(df['data_format_date'][0]),str(df['data_format_date'][len(df)-1]))
    j = 0
    N_date_mancanti = 0
    for curr_date in full_calendar:
        
        if(str(curr_date).split(" ")[0] == str(df['data_format_date'][j])):
            # Se la data corrente e' una di quelle in cui ho fatto una vendita, vai avanti
            j = j + 1
        else:
            # altrimenti metti una riga che indice che non ho venduto nulla
            new_row = [curr_date,999,0,0,-1]#['data_format_date','rag_soc','qta','val','flag_off']
            dff = pd.DataFrame([new_row],columns=df_columns)
            df = pd.concat([df,dff])
            N_date_mancanti += 1
    
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

def myplot(x,y,fullscreen: bool):
    fig, ax = plt.subplots()
    ax = plt.gca()
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.WeekdayLocator(interval=4)
    ax.xaxis.set_major_locator(locator)
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    ax.plot(x,y,linewidth=1)
    if(fullscreen):
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
    plt.show()

def myplot_multi(x,y1,y2,title1: str, title2: str,fullscreen: bool):
    fig, axs = plt.subplots(2)

    formatter = mdates.DateFormatter("%Y-%m-%d")
    axs[0].xaxis.set_major_formatter(formatter)
    axs[1].xaxis.set_major_formatter(formatter)

    locator = mdates.WeekdayLocator(interval=4)
    axs[0].xaxis.set_major_locator(locator)
    axs[1].xaxis.set_major_locator(locator)

    axs[0].grid(True)
    axs[1].grid(True)
    plt.gcf().autofmt_xdate()

    axs[0].plot(x,y1,linewidth=1)
    axs[1].plot(x,y2,linewidth=1)

    axs[0].set_title(title1)
    axs[1].set_title(title2)

    if(fullscreen):
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()

    plt.show()

def print_all(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

def get_day_in_week(df_col):
    ret = []
    for x in df_col:
        ret.append((x.weekday())+1)
    return ret

def get_week_in_year(df_col):
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

def get_day_in_month(df_date):
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

def stop():
    raise SystemExit(0)  

def scala(X_train):
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(X_train)
    return scaler.transform(X_train)

def convert_and_get_rag_soc_to_int(df_rag_soc):
    ret  = []
    for x in df_rag_soc:
        if type(x) == str:
            # print("Current elment: ",x)
            elems = x.split(" ")
            appendme = 0
            if(elems[0]=='IPERSTORE'):
                appendme = 100
            if(elems[0]=='SUPERSTORE'):
                appendme = 200
            if(elems[0]=='SUPERMARKET'):
                appendme = 300
            appendme+=int(elems[1])
        # print("Converted: ",appendme)
        ret.append(appendme)
    return ret
        
def fill_dataframe_visual(df,start,stop,row):
    # full_calendar = fill_dates(str(df['data_format_date'][0]),str(df['data_format_date'][len(df)-1]))
    full_calendar = get_calendar(start,stop)
    j = 0
    date_mancanti = []
    N_date_mancanti = 0
    for curr_date in full_calendar:
        
        if(str(curr_date) == str(df['data_format_date'].values[j])):
            # Se la data corrente e' una di quelle in cui ho fatto una vendita, vai avanti
            # print("La data e' presente "+str(curr_date))
            j = j + 1
        else:
            # altrimenti metti una riga che indice che non ho venduto nulla
            date_mancanti.append(curr_date)
            new_row = [curr_date] + row
            # new_row = [curr_date,0,0]#['data_format_date','qta','val','flag_off']
            dff = pd.DataFrame([new_row],columns=list(df.columns))
            df = pd.concat([df,dff])
            N_date_mancanti += 1
    
    df["data_format_date"] = pd.to_datetime(df["data_format_date"])
    df = df.sort_values(by="data_format_date")
    print("Mancavano "+str(N_date_mancanti)+" date")
    print("Date mancanti:")
    for x in date_mancanti:
        print(str(x))
    df = df.reset_index(drop=True)
    return df