word_count_ = 0
line_count = 0
exclamation_count = 0

file = open("lv1/SMSSpamCollection.txt", encoding="utf8")
for line in file:
    if line.strip().startswith("ham"):
        #word_count_ += len(line.rstrip().lstrip("ham ").split())
        #ovo je bolje stip miće sve razmake (lijeve i desne), split stvara array i [1:] uzima sve u arrayu osim prvog ham-a
        word_count_ += len(line.strip().split()[1:]) 
        line_count += 1

print(f"Avg num of ham words = {word_count_ / line_count}")
word_count_ = 0
line_count = 0
file.seek(0)  #stavlja pokazivac file na prvo mjesto da se opet može pročitati s for line in file

for line in file:
    if line.strip().startswith("spam"):
        word_count_ += len(line.strip().split()[1:])
        line_count += 1

print(f"Avg num of ham words = {word_count_ / line_count}")
file.seek(0)

for line in file:
    lin = line.rstrip()
    if lin.startswith("spam"):
        if lin[len(lin) - 1] == "!":
            exclamation_count += 1
            
file.close()

print(f"Num of exclamations in line ends : {exclamation_count}")