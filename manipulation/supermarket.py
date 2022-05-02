import pandas as pd
import time


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


input_path = '~/CSVs/sales/'
input_file_name = 'supermarket_sales'
extcsv = '.csv'
exttsv = '.tsv'
input_file_absolute_path = input_path+input_file_name+extcsv
output_path = input_path+'cleaned/'


print("Lettura file "+input_file_absolute_path)
time0 = time.time()
df = pd.read_csv(input_file_absolute_path)
print("File letto in %.2f s\n"% (time.time() - time0))

time1 = time.time()
print("Conversione Date...")
df["Date"]=pd.to_datetime(df["Date"])
print("Date completata! %.2f s\n" % (time.time() - time1) )

df.drop('Rating', inplace=True, axis=1)
df.drop("gross margin percentage", inplace=True, axis=1)
df.drop('Branch', inplace=True, axis=1)
df.drop('Payment', inplace=True, axis=1)
df.drop("Customer type", inplace=True, axis=1)
df.drop('City', inplace=True, axis=1)
df.drop("gross income", inplace=True, axis=1)
df.drop("cogs", inplace=True, axis=1)
df.drop('Tax 5%', inplace=True, axis=1)
df.drop('Time', inplace=True, axis=1)
df.drop('Total', inplace=True, axis=1)






time3 = time.time()
print("Scrittura su "+'edit-'+input_file_name+extcsv+' ...')
# df.to_csv(output_path+'edit-'+input_file_name+ext,index=False) #index=False per evitare che scriva la colonna degli indici
df.to_csv(output_path+'edit-'+input_file_name+exttsv,sep='\t',index=False)
print("File scritto in %.2f s\n" % (time.time() - time3))

print("Tempo totale: %.2f s" % (time.time() - time0))
print(df.head(10).to_string(index=False)+'\n')

# print(df.head(7))

