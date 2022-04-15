print("------STRINGHE-----\n")
var = " nome cognome sopraNNome "
print(var.title()) #iniziali maiuscole e il resto minuscolo
print(var.upper()) #tutto maiuscolo
print(var.lower()) #tutto minuscolo
print(var.rstrip()) #toglie eventuali spazi a destra
print(var.lstrip()) #toglie eventuali spazi a sinistra
print(var.strip()) #toglie ecentuali spazi sia a desta che a sinistra
var = 23
print(str(var)) #converte numero in stringa

#operazioni numeriche numeri
# 3 ** 2 = 3^2 = 9
#3 * 2 = 3 per 2 = 6

age = 25
message = "Happy " + str(age) + "rd Birthday!"
print(message)

#-------------LISTE-------------
print("\n------LISTE-----\n")
lista = ['primo','secondo',3,'quarto']
print(lista)
lista[0] = "primo1"
print(lista)
lista.append("quinto")
print(lista)
l = []
l.append("nuovo1")
l.append("nuovo2")
print(l)
l.append(5000) #inserisce in fondo 
print(l)
popped = l.pop() #poppa(rimuove) l'ultimo elemento l.pop(1) rimuove un elemento
print(popped)
print(l)
l.insert(0,"zero") #inserisce nella posizione
print(l)
del l[1] #rimuove un elemento 
print(l)
l.append('4')
l.append('1')
l.append("ciao")
print(l)
#l_copy = l #non crea una nuova lista uguale e' soltanto un puntatore
l.sort() #ordina gli elementi nella lista
#se metti numeri non funziona, se metti i numeri tra '' funziona
#e li ordina pure in modo corretto
#-------------CANCELLARE TUTTO DA LISTA-------------
l.clear() #pulisce la lista
print(l)
#-------------ORDINARE UNA LISTA TEMPORANEAMENTE-------------
l.append("ciao")
l.append("secondo")
l.append("avaia")
print(l)
print(sorted(l))
print(l)

#-------------IF
age = 18
if age == 1:
    print(1)
elif age == 3:
    print(2)
    print(2)
else:
    print(3)

