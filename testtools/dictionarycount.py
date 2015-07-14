from string import ascii_lowercase
from random import choice
from spambayes.Options import get_pathname_option

destination = get_pathname_option("TestDriver", "spam_directories") % 4 + "/"

letterset = {}

for letter in ascii_lowercase:
    letterset[letter] = []

with open("dictionary.txt", 'r') as dict:
    for line in dict:
        letter = line[0]
        letterset[letter].append(line.strip())

x = .3  # Percentage overlap between any two sets (i.e. correlation)
y = 50  # Number of sets to pull

for letter in letterset.keys():
    print "Writing sets for letter " + letter + " ..."

    o_set = letterset[letter]
    b = len(o_set)  # Size of set being pulled from
    a = int(b * x)  # Size of resultant sets

    for i in range(y):
        with open(destination + str(letter) + str(i + 1) + ".txt", 'w') as outfile:
            output = []
            for k in range(a):
                word = choice(o_set)
                if word in output:
                    k -= 1
                    continue
                else:
                    output.append(word)
            for word in output:
                outfile.write(word + "\n")

