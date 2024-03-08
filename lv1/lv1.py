
#PRIMJERI...

#1
x = 1
if x < 10:
    print(" x je manji od 10 ")
else :
    print (" x je veci ili jednak od 10 ")


#2
i = 5
while i > 0:
    print ( i )
    i -= 1
print ("Petlja\tgotova")
for i in range (0 , 5 ):
    print ( i )


#3
lstEmpty = [ ]
lstFriend = ["Marko", "Luka", "Pero", "Dobriša", "Guzman", "Prdica"]
lstFriend.append("Ivan")
print ( lstFriend [0])
print ( lstFriend [0:5:3]) #[start:end:step]
print ( lstFriend [ :2])
print ( lstFriend [1: ])
print ( lstFriend [1:3])


#4
a = [1 , 2 , 3]
b = [4 , 5 , 6]
c = a + b
print(c)
print(max(c))
c[0] = 7
c.pop()
for number in c:
    print(" List number ", number )

for i in range(0, len(c)):
    print(f"Number index {i} : ", c[i]) #with f-string you can print things in {}
print(" Done ! ")


#5 strings...
fruit = "banana"
index = 0
count = 0
# while index < len(fruit):
#     letter = fruit[index]
#     if letter == 'a':
#         count = count + 1
#     print(letter)
#     index += 1

for c in fruit:
    if c == 'a': count += 1 
    print(c)
print ( count )
print ( fruit [0:3])
print ( fruit [0: ])
print ( fruit [2:6:1])
print ( fruit [0:-1]) #[0:-x] ispiše od početka do x slova od kraja (npr. str = "martin", str[0,-2] je "mart")


#6
line = "Dobrodosli u nas grad"
if line.startswith("Dobrodosli"):
    print("Prva rijec je Dobrodosli")
elif line.startswith("dobrodosli"):
    print("Prva rijec je dobrodosli")
print(line.lower())
data = "From : pero@yahoo.com"
atpos = data.find("@")
print(atpos)


#7
letters = ("a ", "b ", "c ", "d ", "e ")
numbers = (1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11)
mixed = (1 , " Hello ", 3.14 )
print(letters[0])
print(letters[1:4])
for letter in letters:
    print(letter)


#8
hr_num = {"jedan" : 1 , "dva" : 2 , "tri" : 3}
print(hr_num)
print(hr_num["dva"])
hr_num["cetiri"] = 4
print(hr_num)


#9 koristenje modula...
import random
import math
for i in range(20): 
    x = random.randint(0, 2)
    y = math.cos(x * math.pi)
    print("Broj : ", x , " Sin(broj) : ", y )


#10 functions
    
def print_hello():
    print("Hello world")
print_hello()


#11 otvaranje text dat...

fhand = open ("lv1.py")
for line in fhand:
    line = line.rstrip()
    print(line)
    words = line.split()
fhand.close()


#12 proba
txt = "     banana,,,,,ssqqqww....."
x = txt.rstrip(",.qsw")
x = x.lstrip()
x = x.lstrip("ab") #stripa sve do slova koje ne može stripat i onda odustaje
#imamo strip, left strip, right strip ako kao argument predamo
# .strip()
# .lstrip()
# .rstrip()

print("of all fruits", x, "is my favorite")


# #1. zadatak
# def total_euro(pay_per_hour, working_hours):
#     return pay_per_hour * working_hours

# print("1. zadatak")
# radni_sati = int(input("Unesi br radnih sati: "))
# placa_po_satu = int(input("Unesi placu po satu: "))
# print("Radni sati: ", radni_sati)
# print("eura/h: ", placa_po_satu)
# print("Ukupno: ", total_euro(placa_po_satu, radni_sati), " eura")

# #2. zadatak
# print("2. zadatak")
# mark = float(input("Unesi ocjenu: "))
# if not type(mark) is float:
#     raise TypeError("Only floats or whole numbers are alowed")
# if mark >= 0.9 and mark <= 1.0:
#     print("A")
# elif mark >= 0.8:
#     print("B")
# elif mark >= 0.7:
#     print("C")
# elif mark >= 0.6:
#     print("D")
# elif mark < 0.6:
#     print("F")
# else: print("Mark is not in range")

# import statistics

# #3. zadatak
# print("3. zadatak")
# lista_ = [0.0]
# while True:
#     input_ = input("Unesi broj: ")
#     if input_ == "Done": 
#         break
#     if type(input_) is float or int: 
#         lista_.append(float(input_))
#     else:
#         print("Input is not a num")
# print(f"Unešeno je {len(input_)} brojeva.")
# print(f"Srednja vrijednost je {statistics.mean(lista_)}")
# print(f"Min vrijednost je {min(lista_)}")
# print(f"Max vrijednost je {max(lista_)}")
# lista_.sort()
# for num in lista_: 
#     print(num)

# #4. zadatak
# words = {}
# print("4. zadatak")
# fhand = open("song.txt")
# for line in fhand:
#     for word in line.rstrip().split():
#         if word not in words:
#             words[word] = 1
#         else:
#             words[word] += 1
# fhand.close()
# print(words)
# num_of_once_words_ = 0
# for word in words:
#     if words[word] == 1:
#         num_of_once_words_ += 1
# print(f"Broj rijeci koje se pojavljuju samo jednom je: {num_of_once_words_}")
# for word in words:
#     if words[word] == 1:
#         print(word)

#5. zadatak
print("5. zadatak")
word_count_ = 0
line_count = 0
file = open("SMSSpamCollection.txt", encoding="utf8")
for line in file:
    if line.rstrip().startswith("ham"):
        word_count_ += len(line.rstrip().lstrip("ham ").split())
        line_count += 1
file.close()
print(f"Avg num of ham words = {word_count_ / line_count}")
word_count_ = 0
line_count = 0
file = open("SMSSpamCollection.txt", encoding="utf8")
for line in file:
    if line.rstrip().startswith("spam"):
        word_count_ += len(line.rstrip().lstrip("spam ").split())
        line_count += 1
file.close()
file = open("SMSSpamCollection.txt", encoding="utf8")
exclamation_count = 0
for line in file:
    if line.rstrip().startswith("spam")
        exclamation_count += 1
file.close()
print(f"Num of ! : {exclamation_count}")
