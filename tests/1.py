from classes import *

g = input("Giorno: ")
m = input("Mese: ")
a = input("Anno: ")
d2 = Day_of_the_year(int(g),int(m),int(a))

giorno = d2.get_day_week()
settimana = d2.get_week()

print(giorno)
print(settimana)