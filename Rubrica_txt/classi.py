import json

class Contatto():
    def __init__(self,nome,cognome,numeri) -> None:
        self.nome = nome
        self.cognome = cognome
        self.numeri = numeri #dizionario
        self.n_numeri = len(numeri)



class Rubrica():
    def __init__(self,nomefile) -> None:
        nomefile = nomefile.rstrip()+".json"
        self.file = nomefile
        self.rubrica = {}
        try:
            with open(nomefile,'r+') as f:
                f.read()
        except FileNotFoundError:
            with open(nomefile,"w+") as f:
                print("file not found")
                f.read()
        else:
            print(self.file)



    def crea_contatto(self,id):
        nome = input("Nome: ")
        cognome = input("Cognome: ")
        numeri = {}
        res = 's'
        i=-1
        while res == 's':
            res = 'x'
            i+=1
            numero = input("Numero: ")
            etichetta = input("Etichetta: ")
            numeri[etichetta] = numero

            while res != 's' and res != 'n':
                res = input("Vuoi inserire un altro numero? [s/n]")
        contatto = Contatto(nome,cognome,numeri)
        Rubrica.scrivi_contatto(self,id,contatto)



    def apri_rubrica(self):#return lista
        try:
            with open(self.file,'r') as f:
                ob = json.load(f)
                self.rubrica = json.loads(ob)
        except json.JSONDecodeError:
            print("La rubrica e' vuota")
            self.rubrica = {}
            return self.rubrica
        except FileNotFoundError:
            print("Creare prima il file")
            self.rubrica = {}
            return self.rubrica
        else:
            return self.rubrica



    def scrivi_contatto(self,id,contatto):
        self.rubrica = Rubrica.apri_rubrica(self)
        self.rubrica[id] = contatto
        try:
            with open(self.file,'w+') as f:
                o = json.dumps(self.rubrica,default = vars,indent=True)
                json.dump(o,f)

        except FileNotFoundError:
            print("Un file chiamato "+ self.file +" non e' stato trovato.")
        else:
            print("Contatto inserito")


        
    def prendi_contatto(self):
        self.rubrica = Rubrica.apri_rubrica(self)
        #implementare...
        if self.rubrica == {}:
            return "Nessun contatto"
        else:
            return self.rubrica