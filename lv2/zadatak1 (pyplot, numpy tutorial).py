import numpy as np
import matplotlib.pyplot as plt

# Definirajte x vrijednosti
#zadnji broj je koliko točaka linspace vrati
x = np.linspace(0, 10, 100)
print(type(x))
print(x.shape)

# Definirajte y vrijednosti za svaku funkciju
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)

# Nacrtajte funkcije
plt.figure()
plt.plot(x, y1, color='blue', linewidth=2)
plt.plot(x, y2, color='red', linewidth=1, linestyle='--')
plt.plot(x, y3, color='green', linewidth=1, linestyle='-.')

# Postavke prikaza
plt.title('Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(["sin(x)", "cos(x)", "sin(x) * cos(x)"])

#prikaz trapezoida
plt.figure()
x = np.array([1, 2, 3, 3, 1])
y = np.array([1, 2, 2, 1, 1])
plt.plot(x , y , 'g' , linewidth = 2 , marker = "*" , markersize = 10)
plt.axis([0,4,0,4])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Prikaz slike 2.3')

plt.show()