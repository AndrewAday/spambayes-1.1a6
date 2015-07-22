from string import ascii_lowercase
from os import listdir, remove
from random import choice
from spambayes.Options import get_pathname_option


def write_dictionary_sets(x=0.3, y=200):

    destination = get_pathname_option("TestDriver", "spam_directories") % 3 + "/"

    letterset = {}  # A dictionary of words: Key = Letter, Value = Words beginning with that letter

    for letter in ascii_lowercase:
        letterset[letter] = []

    with open("dictionary.txt", 'r') as dict:
        for line in dict:
            letter = line[0]
            letterset[letter].append(line.strip())

    for letter in letterset.keys():
        print "Writing sets for letter " + letter + " ..."

        o_set = letterset[letter]   # Size of original set of words beginning with letter
        b = len(o_set)              # Size of set being pulled from
        a = int(b * x)              # Size of resultant sets

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

def reset():

    print "Removing all dictionary sets..."
    dir = get_pathname_option("TestDriver", "spam_directories") % 3 + "/"

    for dictionary in listdir(dir):
        print "Removing " + dir + dictionary
        remove(dir + dictionary)

def main():
    reset()
    write_dictionary_sets(x=0.3, y=500)

if __name__ == "__main__":
    main()