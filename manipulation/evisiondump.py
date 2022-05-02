import pandas as pd
import numpy as np
import time

# t = time.time()
# print("Lettura file...")
# df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',nrows=10000000)
# print("Comletata! %.2f s\n" % (time.time() - t))

# df.columns = ['cod_cli_for', 'rag_soc', 'cod_prod', 'descr_prod', 'data_doc', 'data_format_date', 'qta_offerta', 'qta_non_offerta', 'val_off', 'val_non_off']
# t = time.time()
# print("Scrittura file...")
# df.to_csv("~/CSVs/newdump.tsv",sep="\t",index_label='id')
# print("Comletata! %.2f s\n" % (time.time() - t))



# t = time.time()
# print("Lettura file...")
# df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',skiprows=10000000,nrows=10000000)
# print("Comletata! %.2f s\n" % (time.time() - t))

# df.index=np.arange(10000000,20000000)
# print("Scrittura file...")
# df.to_csv("~/CSVs/newdump.tsv",sep="\t",header=False,mode='a')
# print("Comletata! %.2f s\n" % (time.time() - t))


# t = time.time()
# print("Lettura file...")
# df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',skiprows=20000000,nrows=10000000)
# print("Comletata! %.2f s\n" % (time.time() - t))

# df.index=np.arange(20000000,30000000)
# print("Scrittura file...")
# df.to_csv("~/CSVs/newdump.tsv",sep="\t",header=False,mode='a')
# print("Comletata! %.2f s\n" % (time.time() - t))
# # exit()


# t = time.time()
# print("Lettura file...")
# df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',skiprows=30000000,nrows=10000000)
# print("Comletata! %.2f s\n" % (time.time() - t))

# df.index=np.arange(30000000,40000000)
# print("Scrittura file...")
# df.to_csv("~/CSVs/newdump.tsv",sep="\t",header=False,mode='a')
# print("Comletata! %.2f s\n" % (time.time() - t))
# # exit()


# t = time.time()
# print("Lettura file...")
# df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',skiprows=40000000,nrows=10000000)
# print("Comletata! %.2f s\n" % (time.time() - t))

# df.index=np.arange(40000000,50000000)
# print("Scrittura file...")
# df.to_csv("~/CSVs/newdump.tsv",sep="\t",header=False,mode='a')
# print("Comletata! %.2f s\n" % (time.time() - t))
# # exit()


# t = time.time()
# print("Lettura file...")
# df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',skiprows=50000000,nrows=10000000)
# print("Comletata! %.2f s\n" % (time.time() - t))

# df.index=np.arange(50000000,60000000)
# print("Scrittura file...")
# df.to_csv("~/CSVs/newdump.tsv",sep="\t",header=False,mode='a')
# print("Comletata! %.2f s\n" % (time.time() - t))


# t = time.time()
# print("Lettura file...")
# df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',skiprows=60000000,nrows=10000000)
# print("Comletata! %.2f s\n" % (time.time() - t))

# df.index=np.arange(60000000,70000000)
# print("Scrittura file...")
# df.to_csv("~/CSVs/newdump.tsv",sep="\t",header=False,mode='a')
# print("Comletata! %.2f s\n" % (time.time() - t))


# t = time.time()
# print("Lettura file...")
# df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',skiprows=70000000,nrows=10000000)
# print("Comletata! %.2f s\n" % (time.time() - t))

# df.index=np.arange(70000000,80000000)
# print("Scrittura file...")
# df.to_csv("~/CSVs/newdump.tsv",sep="\t",header=False,mode='a')
# print("Comletata! %.2f s\n" % (time.time() - t))


t = time.time()
print("Lettura file...")
df = pd.read_csv("~/CSVs/dump.tsv",sep='\t',skiprows=80000000)
print("Comletata! %.2f s\n" % (time.time() - t))

df.index=np.arange(80000000,80547100)
print("Scrittura file...")
df.to_csv("~/CSVs/newdump.tsv",sep="\t",header=False,mode='a')
print("Comletata! %.2f s\n" % (time.time() - t))