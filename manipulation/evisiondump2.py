from matplotlib import offsetbox
import pandas as pd
import time
import numpy as np
import sys

def modify(df):
    # aggiungere nomi alle colonne
    df.columns = ['cod_cli_for', 'rag_soc', 'cod_prod', 'descr_prod', 'data_doc', 'data_format_date', 'qta', 'qta_non_offerta', 'val', 'val_non_off']
    # modificare le colonne
    df['flag_off'] = df['qta'] > 0
    df['flag_off'] = df['flag_off'].astype(int)
    df['qta'] = df['qta'] + df['qta_non_offerta']
    df['val'] = df['val'] + df['val_non_off']
    # eliminare colonne
    df.drop('val_non_off', axis=1, inplace=True)
    df.drop('qta_non_offerta', axis=1, inplace=True)
    # eliminazione righe che contenfono quantita' negative
    idxs = df[ df['qta'] <= 0 ].index
    df.drop(idxs, inplace = True)
    print("Righe scartate "+str(len(idxs))+" righe da scrivere "+str(len(df.index)))
    
    return len(df.index)


# START ---------------------------------------------------------------------------

# primo_indice = int(sys.argv[2])

righe_da_leggere = 10000000

n = int(sys.argv[1])


if n >= 0 and n <= 8:
    t = time.time()
    print("Salto le prime "+str(righe_da_leggere*n)+" di righe e ne leggo "+str(righe_da_leggere))
    df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',skiprows=(righe_da_leggere*n),nrows=righe_da_leggere)
    print("Comletata! %.2f s\n" % (time.time() - t))

    len = modify(df)

    t = time.time()
    if n == 0:
        df.to_csv("~/CSVs/newdump2.tsv",sep="\t",index=False,header=True)
    else:
        df.to_csv("~/CSVs/newdump2.tsv",sep="\t",index=False,header=False,mode='a')
    print("Comletata! %.2f s\n" % (time.time() - t))


if n == 9:
    t = time.time()
    print("Salto le prime "+str(righe_da_leggere*n)+" di righe e leggo le rimanenti")
    df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',skiprows=80000000)
    print("Comletata! %.2f s\n" % (time.time() - t))
    print(df)

    len = modify(df)

    t = time.time()
    df.to_csv("~/CSVs/newdump2.tsv",sep="\t",index=False,header=False,mode='a')
    print("Comletata! %.2f s\n" % (time.time() - t))  
    

