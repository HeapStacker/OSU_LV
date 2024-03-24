import numpy as np
a = np.array([6, 2, 9])
print(type(a)) #vraća klasifikaciju objekta
print(a.ndim) #vraća dimenziju polja, za npr. matrice je 2
print(a.shape) #vraća tuple s dužinom, širinom polja u našem slučaju (3,) jer array ima samo 3 broja
print(a.dtype) #vraća tip podataka u array-u
print(a[0], a[1], a[2])
a[1] = 5
print(a)
print(a[1:2])
print(a[1:-1])
b = np.array(
        [
            [3, 7, 1], 
            [4, 5, 6],
            [33, 11, 24],
            [16, 11, 10],
            [7, 7, 7],
            [0, 0, 0], 
            [11, 2, 18]
        ]
    ) #napravi dvodimenzionalno polje (matricu)
print(b.shape)
print(b)
print(b[0, 2], b[0, 1], b[1, 1])
print()
print(b[0:6, 0:2]) #izdvajanje (ispisujemo prvih 6 redova ali ispisujemo tako da ih ispišemo do 3. stupca (0:2 ne ispisujemo treći))
print()
print(b[:, 1]) #ispisujemo svaki red ali samo 2. stupac
print()
print(b[:, 0]) #ispisujemo samo prvi član svakog reda
#kad se radi o ispisivanju samo jednog stupca matrice ispisuje se kao array

#inicijalizacija polja (matrica) s određenim brojevima...
c = np.zeros((4, 2))
d = np.ones((3, 2))
e = np.full((3, 3), 5) #3x3 matrica ispunjena brojem 5
f = np.eye(5) #jedinicna matrica dimenzije 5
g = np.array([1, 2, 3], np.float32) #specifikacija tipa podataka (ide iza samih podataka)
print(g)
g = np.array([1, 2, 3], np.int8) 
print(g)
duljina = len(g)
print(duljina)
h = g.tolist()
print(h)
c = b.transpose() #ne izgleda dobro kad koristimo jednodimenzionalne nizove (npr h = g.transpose() ispadne isto ko i g)
print(c)
d = np.concatenate((a, g)).tolist()
d = a + g #za razliku od normalnih lista, + ovdje zbraja članove
d.sort()
print(d)
print()

#primjer 2.2
print("primjer 2.2")
a = np.array([3, 1, 5], float)
b = np.array([2, 4, 8], float)
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print()
print(a.min()) #ispiše najmanji el
print(a.argmin()) #ispiše mjesto na kojemu se nalazi najmanji el
print(a.max())
print(a.argmax())
print(a.sum())
print(a.mean())
print()
print(np.mean(a))
print(np.max(a)) #isto, np ima funkciju argmax, argmin..
print(np.min(a))
print(np.sum(a))

#primjer 2.3
import random
import statistics
random.seed(1) # za razliku od numpy-a je thread_safe (kad odrediš seed onda će u svakom izvodu programa generacija biti jednaka)
randList = []
for i in range(0, 10):
    randList.append(random.randint(0, 2)) #appenda se borj koji mize biti 0, 1 ili 2 (ne 3)
print(randList)
print(statistics.mean(randList))
print()

print("numpy rand list generation...")
np.random.seed(1) #nije threadsafe (kolko cujem)
randList = []
for i in range(0, 10):
    randList.append(np.random.randint(0, 3)) #appenda se broj koji može biti 0, 1 ili 2 (ne 3)
randList = np.random.rand(10) # na ovaj nacin mozemo brže napraviti listu od 10 random brojeva (bez for loopa)
print(randList)
print(randList.mean())

#primjer 2.4
print()
import matplotlib.pyplot as plt
x = np.linspace(0, 6, num=10) # num je broj generiranih tocaka
print(x.tolist()) # ak ne stavimo tolist bit će u numpy formatu (kao lista ali bez zareza)
y = np.sin(x)
plt.plot(x, y, "b", linewidth = 2, marker="d", markersize=10)
plt.axis([0, 6, -2, 10]) # apscisa od 0 do 6 a ordinata od -2 do 10 
plt.xlabel("x")
plt.ylabel("vrijednost funkcije")
plt.title("Sinus f-ija")
plt.show()

#OVO SU MARKERI KOJE MOŽEŠ KORISTITI U PLOTANJU..
# '.': 'point',
# ',': 'pixel',
# 'o': 'circle',
# 'v': 'triangle_down',
# '^': 'triangle_up',
# '<': 'triangle_left',
# '>': 'triangle_right',
# '1': 'tri_down',
# '2': 'tri_up',
# '3': 'tri_left',
# '4': 'tri_right',
# '8': 'octagon',
# 's': 'square',
# 'p': 'pentagon',
# '*': 'star',
# 'h': 'hexagon1',
# 'H': 'hexagon2',
# '+': 'plus',
# 'x': 'x',
# 'D': 'diamond',
# 'd': 'thin_diamond',
# '|': 'vline',
# '_': 'hline',
# 'P': 'plus_filled',
# 'X': 'x_filled',
# TICKLEFT: 'tickleft',
# TICKRIGHT: 'tickright',
# TICKUP: 'tickup',
# TICKDOWN: 'tickdown',
# CARETLEFT: 'caretleft',
# CARETRIGHT: 'caretright',
# CARETUP: 'caretup',
# CARETDOWN: 'caretdown',
# CARETLEFTBASE: 'caretleftbase',
# CARETRIGHTBASE: 'caretrightbase',
# CARETUPBASE: 'caretupbase',
# CARETDOWNBASE: 'caretdownbase',
# "None": 'nothing',
# "none": 'nothing',
# ' ': 'nothing',
# '': 'nothing'

print()
#primjer 2.5
img = plt.imread("lv2/road.jpg")
img = img[:,:,0].copy()
print(img.shape)
print(img.dtype)
plt.figure()
plt.imshow(img, cmap="gray")
plt.show()

