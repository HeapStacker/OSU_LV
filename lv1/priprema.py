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

# fhand = open ("lv1.py")
# for line in fhand:
#     line = line.rstrip()
#     print(line)
#     words = line.split()
# fhand.close()


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