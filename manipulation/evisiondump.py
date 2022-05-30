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
    # idxs = df[ df['qta'] < 0 ].index
    # print(df[ df['qta'] < 0 ])
    # df.drop(idxs, inplace = True)
    # print("Scartate "+str(len(idxs))+" righe")
    
    # return len(idxs)


# START ---------------------------------------------------------------------------

# righe_scritte = 0

if sys.argv[1] == str(1):
    t = time.time()
    print("Lettura file...")
    df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',nrows=10000000)
    print("Comletata! %.2f s\n" % (time.time() - t))

    len = modify(df)
    # righe_scritte = righe_scritte + (10000000 - len)

    t = time.time()
    # print("Scrittura "+str(righe_scritte)+" righe nel file...")
    df.to_csv("~/CSVs/newdump.tsv",sep="\t",index_label='id')
    print("Comletata! %.2f s\n" % (time.time() - t))



if sys.argv[1] == str(2):
    t = time.time()
    print("Lettura file...")
    df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',skiprows=10000000,nrows=10000000)
    print("Comletata! %.2f s\n" % (time.time() - t))

    len = modify(df)
    # righe_scritte = righe_scritte + (10000000 - len)

    t = time.time()
    df.index=np.arange(10000000,20000000)
    # print("Scrittura "+str(righe_scritte)+" righe nel file...")
    df.to_csv("~/CSVs/newdump.tsv",sep="\t",header=False,mode='a')
    print("Comletata! %.2f s\n" % (time.time() - t))



if sys.argv[1] == str(3):
    t = time.time()
    print("Lettura file...")
    df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',skiprows=20000000,nrows=10000000)
    print("Comletata! %.2f s\n" % (time.time() - t))

    len = modify(df)
    # righe_scritte = righe_scritte + (10000000 - len)

    t = time.time()
    df.index=np.arange(20000000,30000000)
    # print("Scrittura "+str(righe_scritte)+" righe nel file...")
    df.to_csv("~/CSVs/newdump.tsv",sep="\t",header=False,mode='a')
    print("Comletata! %.2f s\n" % (time.time() - t))



if sys.argv[1] == str(4):
    t = time.time()
    print("Lettura file...")
    df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',skiprows=30000000,nrows=10000000)
    print("Comletata! %.2f s\n" % (time.time() - t))

    len = modify(df)
    # righe_scritte = righe_scritte + (10000000 - len)

    t = time.time()
    df.index=np.arange(30000000,40000000)
    # print("Scrittura "+str(righe_scritte)+" righe nel file...")
    df.to_csv("~/CSVs/newdump.tsv",sep="\t",header=False,mode='a')
    print("Comletata! %.2f s\n" % (time.time() - t))



if sys.argv[1] == str(5):
    t = time.time()
    print("Lettura file...")
    df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',skiprows=40000000,nrows=10000000)
    print("Comletata! %.2f s\n" % (time.time() - t))

    len = modify(df)
    # righe_scritte = righe_scritte + (10000000 - len)

    t = time.time()
    df.index=np.arange(40000000,50000000)
    # print("Scrittura "+str(righe_scritte)+" righe nel file...")
    df.to_csv("~/CSVs/newdump.tsv",sep="\t",header=False,mode='a')
    print("Comletata! %.2f s\n" % (time.time() - t))



if sys.argv[1] == str(6):
    t = time.time()
    print("Lettura file...")
    df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',skiprows=50000000,nrows=10000000)
    print("Comletata! %.2f s\n" % (time.time() - t))

    len = modify(df)
    # righe_scritte = righe_scritte + (10000000 - len)

    t = time.time()
    df.index=np.arange(50000000,60000000)
    # print("Scrittura "+str(righe_scritte)+" righe nel file...")
    df.to_csv("~/CSVs/newdump.tsv",sep="\t",header=False,mode='a')
    print("Comletata! %.2f s\n" % (time.time() - t))



if sys.argv[1] == str(7):
    t = time.time()
    print("Lettura file...")
    df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',skiprows=60000000,nrows=10000000)
    print("Comletata! %.2f s\n" % (time.time() - t))

    modify(df)

    t = time.time()
    df.index=np.arange(60000000,70000000)
    # print("Scrittura "+str(righe_scritte)+" righe nel file...")
    df.to_csv("~/CSVs/newdump.tsv",sep="\t",header=False,mode='a')
    print("Comletata! %.2f s\n" % (time.time() - t))



if sys.argv[1] == str(8):
    t = time.time()
    print("Lettura file...")
    df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',skiprows=70000000,nrows=10000000)
    print("Comletata! %.2f s\n" % (time.time() - t))

    len = modify(df)
    # righe_scritte = righe_scritte + (10000000 - len)

    t = time.time()
    df.index=np.arange(70000000,80000000)
    # print("Scrittura "+str(righe_scritte)+" righe nel file...")
    df.to_csv("~/CSVs/newdump.tsv",sep="\t",header=False,mode='a')
    print("Comletata! %.2f s\n" % (time.time() - t))



if sys.argv[1] == str(9):
    t = time.time()
    print("Lettura file...")
    df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',skiprows=80000000)
    print("Comletata! %.2f s\n" % (time.time() - t))

    len = modify(df)
    # righe_scritte = righe_scritte + (10000000 - len)

    t = time.time()
    df.index=np.arange(80000000,80547100)
    # print("Scrittura "+str(righe_scritte)+" righe nel file...")
    df.to_csv("~/CSVs/newdump.tsv",sep="\t",header=False,mode='a')
    print("Comletata! %.2f s\n" % (time.time() - t))  
    
    # 80547100