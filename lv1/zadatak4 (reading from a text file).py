words = {}
fhand = open("song.txt")
for line in fhand:
    for word in line.rstrip().split():
        if word not in words:
            words[word] = 1
        else:
            words[word] += 1
fhand.close()
print(words)
num_of_once_words_ = 0
for word in words:
    if words[word] == 1:
        num_of_once_words_ += 1
print(f"\nBroj rijeci koje se pojavljuju samo jednom je: {num_of_once_words_}")
for word in words:
    if words[word] == 1:
        print(word)