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
        self.file_rubrica = nomefile
        self.rubrica = {}
        try:
            with open(self.file_rubrica,'r+') as f:
                f.read()
        except FileNotFoundError:
            with open(self.file_rubrica,'w+') as f:
                print("file created")
                f.read()



    def crea_contatto(self):
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
        Rubrica.scrivi_contatto(self,contatto)



    def get_lista_contatti(self):#return lista
        try:
            with open(self.file_rubrica,'r') as f:
                ob = json.load(f)
                self.rubrica = json.loads(ob)
        except json.JSONDecodeError:
            self.rubrica = {}
            return self.rubrica
        else:
            return self.rubrica



    def scrivi_contatto(self,contatto):
        self.rubrica = Rubrica.get_lista_contatti(self)
        id = len(self.rubrica)
        self.rubrica[id] = contatto
        try:
            with open(self.file_rubrica,'w+') as f:
                o = json.dumps(self.rubrica,default = vars,indent=True)
                json.dump(o,f)
        except FileNotFoundError:
            print("Un file chiamato "+ self.file_rubrica +" non e' stato trovato.")
        else:
            print("Contatto inserito")


        

    def prendi_contatto(self,nome,cognome):
        self.rubrica = Rubrica.get_lista_contatti(self)
        #implementare...
        if self.rubrica == {}:
            return "Nessun contatto"
        
        #cerca contatto
        for id,contact_obj in self.rubrica.items():
            if contact_obj['nome'] == nome.strip() and contact_obj['cognome'] == cognome.strip():
                return contact_obj
        return "Nessun contatto"