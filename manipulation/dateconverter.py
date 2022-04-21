import pandas as pd
import time


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# 2 colonne
# 100 righe :  0.00s
# 1k righe :   0.06s
# 5k righe :   0.30s
# 10k righe :  0.59s
# 50k righe :  2.98s
# 100k righe : 7.03s
# 500k righe : 30.08s
# 1M righe :   59.33 s
# 1,5M righe : 101.54 s
# 2M righe :   123.62 s
 

file_path = '~/CSVs/'
nome_file = 'sales-1M'
ext = '.csv'
output_path = file_path


print("Lettura file "+nome_file+ext)
time0 = time.time()
df = pd.read_csv(file_path+nome_file+ext)
print("File letto in %.2f s\n"% (time.time() - time0))


time1 = time.time()
print("Conversione Order Date...")
df["Order Date"]= pd.to_datetime(df["Order Date"])
print("Order Date completata! %.2f s\n" % (time.time() - time1) )

time2 = time.time()
print("Conversione Order Date...")
df["Ship Date"]= pd.to_datetime(df["Ship Date"])
print("Ship Date comletata! %.2f s\n" % (time.time() - time2))

time3 = time.time()
print("Scrittura su "+'edit-'+nome_file+ext+' ...')
df.to_csv(output_path+'edit-'+nome_file+ext)
print("File scritto in %.2f s\n" % (time.time() - time3))

print("Tempo totale: %.2f s" % (time.time() - time0))

# print(df.head(7))
