class Day_of_the_year():

    """costruttore con parametri"""
    def __init__(self,day,month,year):
        self.day = day
        self.month = month
        self.year = year

    def get_days_in_month(month):
        if month == 2:
            return 28
        if month == 11 or month == 4 or month == 6 or month == 9:
            return 30
        else:
            return 31    

    def get_week(self):
        import math
        sum_day = 0
        for i in range (1,self.month+1):
            sum_day += Day_of_the_year.get_days_in_month(i)
        # day = Day_of_the_year.get_day_week(self)
        # week = {'Domenica':6,'Lunedi':0,'Martedi':1,'Mercoledi':2,'Giovedi':3,'Venerdi':4,'Sabato':5}
        # day_n = week[day]
        #sum_day = sum_day - day_n + 1
        da_togliere = Day_of_the_year.get_days_in_month(self.month) - self.day
        sum_day = sum_day - da_togliere
        return math.floor(sum_day/7)
        
    def get_day_week(self):
        y1 = str(self.year)
        y2 = y1[2:5]
        c1 = y1[0:2]
        c = int(c1)
        m = self.month -2
        if m < 1: 
            m = m + 12
        y = int(y2)
        if self.month == 1 or self.month == 2:
            y = y -1
        k = self.day 
        import math
        w = (k + math.floor(2.6*m - 0.2) -2*c + y + math.floor(y/4) + math.floor(c/4))%7
        return['Domenica','Lunedi','Martedi','Mercoledi','Giovedi','Venerdi','Sabato'][w]

