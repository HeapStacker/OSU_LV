import statistics

lista_ = []
while True:
    input_ = input("Unesi broj: ")
    if input_ == "Done": 
        break

    #ovo mogu umjesto try except...
    if type(input_) is float or int: 
        lista_.append(float(input_))
    else:
        print("Input is not a num")

print(f"Unešeno je {len(input_)} brojeva.")
print(f"Srednja vrijednost je {statistics.mean(lista_)}")
print(f"Min vrijednost je {min(lista_)}")
print(f"Max vrijednost je {max(lista_)}")
lista_.sort()

for num in lista_: 
    print(num)