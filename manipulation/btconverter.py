import pandas as pd
import time
import os
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
    'Date',
    'Description',
    'Deposits',
    'Withdrawls',
    'Balance'
	]
 

input_folder = '~/CSVs/bt/'
input_file_name = 'bt5M'
ext = '.csv'
input_file_absolute_path = input_folder+input_file_name+ext
output_folder = input_folder+'cleaned/'


print("Lettura file "+input_file_absolute_path)
time0 = time.time()
df = pd.read_csv(input_file_absolute_path)
print("File letto in %.2f s\n"% (time.time() - time0))

time1 = time.time()
print("Conversione Date...")
df["Date"] = pd.to_datetime(df["Date"])
print("Date completata! %.2f s\n" % (time.time() - time1))

time2 = time.time()
print("Conversione Deposits...")
# df = df.astype({"Deposits": 'str', "Withdrawls": 'str',"Balance": "str"})
# print(df.dtypes)
# df['Deposits'] = df['Deposits'].astype('str')
df['Deposits'] = df['Deposits'].astype('str')
df['Deposits'] = df['Deposits'].str.replace(',','')
# df['Deposits'] = '"'+str(df['Deposits'])+'"'



df['Withdrawls'] = df['Withdrawls'].astype('str')
df['Withdrawls'] = df['Withdrawls'].str.replace(',', '')

df['Balance'] = df['Balance'].astype('str')
df['Balance'] = df['Balance'].str.replace(',', '')
# print(df.dtypes)
# df.loc[df.Deposits  , "Deposits"] = str("'"+df.Deposits+"'")
# df['Deposits']= df['Deposits'].astype(str)
# print("Deposits comletata! %.2f s\n" % (time.time() - time2))

# time3 = time.time()
# print("Conversione Withdrawls...")
# df['Withdrawls']= df['Withdrawls'].astype(str)
# print("Withdrawls comletata! %.2f s\n" % (time.time() - time3))

# time4 = time.time()
# print("Conversione Balance...")
# df['Balance']= df['Balance'].astype(str)
# print("Balance comletata! %.2f s\n" % (time.time() - time4))



# Cambiare nome ad una colonna
# df=df.rename(columns={"":"Id"})
# df.columns.values[0] = "Region2"

timen = time.time()
print("Scrittura su "+'edit-'+input_file_name+ext+' ...')
# df.to_csv(output_folder+'edit-'+input_file_name+ext,index=False) #index=False per evitare che scriva la colonna degli indici
df.to_csv(output_folder+'edit-'+input_file_name+'.tsv', sep='\t', encoding='utf-8', index=False)
print("File scritto in %.2f s\n" % (time.time() - timen))

print("Tempo totale: %.2f s" % (time.time() - time0))
time.sleep(1)
#os.system("glogg ~/CSVs/bt/cleaned/edit-bt100.tsv")


#print(df.head(10).to_string(index=False)+'\n')

# print(df.head(7))

