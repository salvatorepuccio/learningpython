import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# attr = ['Region',
# 	'Country',
# 	'ItemType',
# 	'SalesChannel',
# 	'OrderPriority',
# 	'OrderDate',
# 	'OrderID',
# 	'ShipDate',
# 	'UnitsSold',
# 	'UnitPrice',
# 	'UnitCost',
# 	'TotalRevenue',
# 	'TotalCost',
# 	'TotalProfit'
# 	]


df = pd.read_csv('~/Downloads/sales-100-1.csv', parse_dates=["OrderDate"])
df["OrderDate"]= pd.to_datetime(df["OrderDate"])
df["ShipDate"]= pd.to_datetime(df["ShipDate"])
#print(df.head(10))
df.to_csv('mod.csv')




#print(df.head(10))
