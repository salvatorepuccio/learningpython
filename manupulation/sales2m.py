import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

attr = ['Region',
	'Country',
	'ItemType',
	'SalesChannel',
	'OrderPriority',
	'OrderDate',
	'OrderID',
	'ShipDate',
	'UnitsSold',
	'UnitPrice',
	'UnitCost',
	'TotalRevenue',
	'TotalCost',
	'TotalProfit'
	]

# dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

# df = pd.read_csv('mycsv.csv', parse_dates=['DATE'], date_parser=dateparse)


# df = pd.read_csv('~/Downloads/sales-100-1.csv',low_memory=False,delimiter=",")
# convertdate = pd.to_datetime(df['OrderDate'])
# print(convertdate)

df = pd.read_csv('~/Downloads/sales-100-1.csv', parse_dates=["OrderDate"])
df["OrderDate"]= pd.to_datetime(df["OrderDate"])
df["ShipDate"]= pd.to_datetime(df["ShipDate"])
print(df.head(10))




#print(df.head(10))
