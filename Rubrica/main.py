from classi import *



i = 'n'

r = Rubrica("rubrica")

while(i != '9'):
   
    print("\n1. aggiungi contatto")
    print("2. vedi contatto")
    print("3. vedi tutta la rubrica")
    print("9. esci\n")
    i = input()

    if i == '1':
        #os.system('clear')
        r.crea_contatto()
    elif i == '2':
        n = input("Nome: ")
        co = input("Cognome: ")
        p = r.prendi_contatto(n,co)
        print(p)
    elif i == '3':
        p = r.get_lista_contatti()
        for k,v in p.items():
            #print(k,v)
            for k1,v1 in v.items():
                if k1 == 'numeri':
                    for k2,v2 in v1.items():
                        print("\t"+k2+":",v2)
                else:
                    print(k1+":",v1)
            print()
    else:
        break

        

print("Arrivederci")



