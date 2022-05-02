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


# headers = [
#     'Id',
#     'Region',
# 	'Country',
# 	'Item Type',
# 	'Sales Channel',
# 	'Order Priority',
# 	'Order Date',
# 	'Order ID',
# 	'Ship Date',
# 	'Units Sold',
# 	'Unit Price',
# 	'Unit Cost',
# 	'Total Revenue',
# 	'Total Cost',
# 	'Total Profit'
# 	]
 

input_path = '~/CSVs/sales/'
input_file_name = 'sales-2M'
extcsv = '.csv'
exttsv = '.tsv'
input_file_absolute_path = input_path+input_file_name+extcsv
output_path = input_path+'cleaned/'


print("Lettura file "+input_file_absolute_path)
# answ = input("Continuare?[Y/n]")

# while answ != 'y':
# 	if answ == 'n':
# 		exit
# 	answ = input("Continuare?[Y/n]")



time0 = time.time()
df = pd.read_csv(input_file_absolute_path)
print("File letto in %.2f s\n"% (time.time() - time0))

df=df.rename(columns={"Item Type":"ItemType"})
df=df.rename(columns={"Sales Channel":"SalesChannel"})
df=df.rename(columns={"Order Priority":"OrderPriority"})
df=df.rename(columns={"Order Date":"OrderDate"})
df=df.rename(columns={"Order ID":"OrderID"})
df=df.rename(columns={"Ship Date":"ShipDate"})
df=df.rename(columns={"Units Sold":"UnitsSold"})
df=df.rename(columns={"Unit Price":"UnitPrice"})
df=df.rename(columns={"Unit Cost":"UnitCost"})
df=df.rename(columns={"Total Revenue":"TotalRevenue"})
df=df.rename(columns={"Total Cost":"TotalCost"})
df=df.rename(columns={"Total Profit":"TotalProfit"})

time1 = time.time()
print("Conversione Order Date...")
df["OrderDate"]= pd.to_datetime(df["OrderDate"])
print("Date completata! %.2f s\n" % (time.time() - time1) )

time2 = time.time()
print("Conversione Ship Date...")
df["ShipDate"]= pd.to_datetime(df["ShipDate"])
print("Ship Date comletata! %.2f s\n" % (time.time() - time2))

# Cambiare nome ad una colonna



# df.columns.values[0] = "cod_cli_for"
# df.columns.values[1] = "rag_soc"
# df.columns.values[2] = "cod_prod"
# df.columns.values[3] = "descr_prod"
# df.columns.values[4] = "data_doc"
# df.columns.values[5] = "datra_format_date"
# df.columns.values[6] = "qta_offerta"
# df.columns.values[7] = "qta_non_offerta"
# df.columns.values[8] = "val_off"
# df.columns.values[9] = "val_non_off"

time3 = time.time()
print("Scrittura su "+'edit-'+input_file_name+extcsv+' ...')
# df.to_csv(output_path+'edit-'+input_file_name+ext,index=False) #index=False per evitare che scriva la colonna degli indici
df.to_csv(output_path+'edit-'+input_file_name+exttsv,sep='\t',index=False)
# df.to_csv(output_path+'edit-'+input_file_name+exttsv,sep='\t')
print("File scritto in %.2f s\n" % (time.time() - time3))

print("Tempo totale: %.2f s" % (time.time() - time0))
print(df.head(10).to_string(index=False)+'\n')

# print(df.head(7))

