from string import ascii_lowercase

def random_words(number):

    words = []

    for letter1 in ascii_lowercase:
        for letter2 in ascii_lowercase:
            for letter3 in ascii_lowercase:
                for letter4 in ascii_lowercase:
                    word = letter1 + letter2 + letter3 + letter4
                    words.append(word)

    file = open("4letterwords.txt", 'r')
    list = file.read().split()

    for word in words:
        if word in list:
            words.remove(word)

    return words[:number]