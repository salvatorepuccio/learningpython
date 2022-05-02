# # from operator import invert
# # num=input("inserisci un numero: ")
# # var = [int(x) for x in str(num)]
# # print(var)
# # var.reverse()
# # print(var)
# # #print(var.sort(reverse=True))

# favorite_languages = {
#        'jen': 'python',
#        'sarah': 'c',
#        'edward': 'ruby',
#        'phil': 'python',
#        }
# # for n, l in favorite_languages.items():
# #     print(n.title() + "'s favorite language is " + 
# #     l.title() + ".")

# for x in favorite_languages.items():
#     print(x)

# def bmi(weight, height):
#     #your code here
#     bmi = weight/(height**2)
#     if bmi <= 18.5:
#         return "Underweight"
#     elif bmi <= 25:
#         return "Normal"
#     elif bmi <= 30:
#         return "Overweight"
#     else:
#         return "Obese"

# alt = input("Altezza")
# peso = input("peso")
# print(bmi(int(peso),int(alt)))

# ret = ['1', '2', '3', '4'][0]
# a = True + True + True + 1
# print(a)
# b=True
# if b == 1:
#     print("si")
# else:
#     print("no")

# def invert(lst):
#     for i in range(0,len(lst)):
#         lst[i] = lst[i]*-1
#     return lst

# l = [1,2,-3,-4]
# l = invert(l)
# print(l)

# def square_sum(numbers):
#     return sum([x**2 for x in numbers])



# def repeat_str(repeat, string):
#     return ''.join([string for i in range(0,repeat)])

# a = repeat_str(5,"Hello")
# print(a)
#  def first_non_consecutive(arr):
#      return 1

# def first_non_consecutive(arr):
#     for i in range(0,len(arr)-1):
#         if(arr[i+1]!=arr[i]+1):
#             return arr[i+1]
#     return None

# a = [1,2,4,5,6,7]#dovrebbe fare 6 (3*(3+1)/2) ma fa 7, quindi non va bene
# b = first_non_consecutive(a)
# print(b)

# def number(lines):
#     #your code here
#     for i in range(0,len(lines)):
#         #print("for: " + str(i) + str(lines[i]))
#         appendme = str(i+1) + ": " + lines[i]
#         lines.pop(i)
#         lines.insert(i,appendme)
#         print(appendme)
#     return lines

# def number(lines):
#     return ['%d: %s' % v for v in enumerate(lines, 1)]

# def sort_by_length(arr):
#     return sorted(arr,key=len)

# def find(n):
#     res = 0
#     for x in range(3,n+1):
#         if x%3 == 0:
#             res += x
#             continue
#         if x%5 == 0:
#             res += x
#     return res

# def find(n):
#     return sum(e for e in range(1, n+1) if e % 3 == 0 or e % 5 == 0)




# a = find(15)
# #print(a)

# #def first_non_consecutive(arr):
# #    return arr[ sum( [True for i in range(0,len(arr)-1) if arr[i]+1==arr[i+1] ] )]

# def first_non_consecutive(arr):
#     return arr[ sum( [True for i in arr[:] if arr[i]+1==arr[i+1] ] )]

# a = [1,2,3,4,5,6,7,9,10,11,12,15,98,80000]
# b = first_non_consecutive(a)
#print(b)

# def arithmetic(a,b,operator):
#     return [a+b,a-b,a*b,a/b][0 if operator == "add" 1 elif operator == "subtract"  2 elif operator == "multiply" 3 elif operator == "divide"]

def arithmetic(a,b,operator):
    return [a+b,a-b,a*b,a/b][0 if operator == "add" else 1 if operator == "subtract" else 2 if operator == "multiply" else 3 if operator == "divide" else -1]

q = 7
w = 3

a = arithmetic(q,w,"add")
print(str(q)+" + "+str(w), a)

a = arithmetic(q,w,"subtract")
print(str(q)+" - "+str(w), a)

a = arithmetic(q,w,"multiply")
print(str(q)+" * "+str(w), a)

a = arithmetic(q,w,"divide")
print(str(q)+" / "+str(w), a)

def pari_dispari(n):
    if(n%2 == 0):
        return "pari"
    else:
        return "dispari"




# # class Dog():

# #     def __init__(self,name,age):
# #         self.name = name
# #         self.age = age
# #         self.collare = True
    
# #     def get_name(self):
# #         return self.name
    
# #     def get_age(self):
# #         return self.age

# # dog = Dog("Eika",7)
# # #dog.collare = True

# # print(dog.name + " e' un cane di " + str(dog.age) + " anni e ", end ="")

# # if dog.collare is True:
# #     print("HA il collare.")
# # else:
# #     print("NON HA il collare.")


# # class Doggo(Dog):
# #     def __init__(self, name, age):
# #         super(Doggo,self).__init__(name, age)
# #         self.rampage=False

# # doggo = Doggo("Cane",89)
# # doggo.rampage = True

# # print(doggo.name,doggo.age,doggo.collare,doggo.rampage)

# def open_or_senior(data):
#     res = []
#     i = 0
#     for e in data:
#         if e[0]>54 and e[1]>7:
#             res.append("Senior")
#         else:
#             res.append("Open")
#     return res

# def openOrSenior(data):
#   return ["Senior" if age >= 55 and handicap >= 8 else "Open" for (age, handicap) in data]

# def series_num(n):
#     if n == 0:
#         return "0.00"
#     res = 1
#     den = 4
#     for i in range(1,n):
#         res+=1/(den)
#         den = den+3
    
#     return str(format(res,".2f"))

# def series_sum(n):
#     return '{:.2f}'.format(sum(1.0/(3 * i + 1) for i in range(n)))


# # a = series_num(2)
# # print(a)

# def xo(s):
#     xs = 0
#     os = 0
#     for x in s:
#         if x == 'x' or x == 'X':
#             xs += 1
#         if x == 'o' or x == 'O':
#             os += 1
#     return xs == os


# a = xo("xosxOo")
# #print(a)

from es import pari_dispari as pd

s = pd(11)
print(s)

