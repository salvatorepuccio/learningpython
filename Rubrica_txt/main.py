from turtle import clear
from classi import *
import os


# def crea_nuova():
#     nome = input("Inerisci il nome: ")
#     r = Rubrica(nome)
#     return r


# i = 'n'

# r = None

# while(i != '9'):
#     print("\n1. crea rubrica")
#     print("2. aggiungi contatto")
#     print("3. leggi\n")
#     print("9. esci")

#     i = input()
#     if i == '1':
#         os.system('clear')
#         r = crea_nuova()
#     elif i == '2':
#         if r == None:
#             os.system('clear')
#             print("Devi ancora creare una rubrica (1)")
#         else:
#             r.crea_contatto()
#     elif i == '3':
#         break

# print("Arrivederci")

r = Rubrica("rubrica")

# r.crea_contatto('0')

# r.crea_contatto('1')

c = r.prendi_contatto()

#print(c)


# print(c['0']['nome'])
# print(c['0']['cognome'])
# print(c['0']['numeri']['casa'])
# print(c['0']['numeri']['cellulare'])

for k,v in c.items():
    print(k,v)
    for k1,v1 in v.items():
        print(k1,v1)
        if k1 == 'numeri':
            for k2,v2 in v1.items():
                print(k2,v2)

#print(c)



