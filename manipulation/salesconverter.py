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
# 1M righe :   59.33s
# 1,5M righe : 101.54s
# 2M righe :   123.62s
# 5M righe :   336.64s


headers = [
    'Id',
    'Region',
	'Country',
	'Item Type',
	'Sales Channel',
	'Order Priority',
	'Order Date',
	'Order ID',
	'Ship Date',
	'Units Sold',
	'Unit Price',
	'Unit Cost',
	'Total Revenue',
	'Total Cost',
	'Total Profit'
	]
 

input_path = '~/CSVs/sales/'
input_file_name = 'sales-100'
ext = '.csv'
input_file_absolute_path = input_path+input_file_name+ext
output_path = input_path+'cleaned/'


print("Lettura file "+input_file_absolute_path)
time0 = time.time()
df = pd.read_csv(input_file_absolute_path)
print("File letto in %.2f s\n"% (time.time() - time0))

time1 = time.time()
print("Conversione Order Date...")
df["Order Date"]= pd.to_datetime(df["Order Date"])
print("Date completata! %.2f s\n" % (time.time() - time1) )

time2 = time.time()
print("Conversione Ship Date...")
df["Ship Date"]= pd.to_datetime(df["Ship Date"])
print("Ship Date comletata! %.2f s\n" % (time.time() - time2))

# Cambiare nome ad una colonna
# df=df.rename(columns={"":"Id"})
# df.columns.values[0] = "Region2"

time3 = time.time()
print("Scrittura su "+'edit-'+input_file_name+ext+' ...')
df.to_csv(output_path+'edit-'+input_file_name+ext,index=False) #index=False per evitare che scriva la colonna degli indici
print("File scritto in %.2f s\n" % (time.time() - time3))

print("Tempo totale: %.2f s" % (time.time() - time0))
print(df.head(10).to_string(index=False)+'\n')

# print(df.head(7))

