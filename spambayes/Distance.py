# import Levenshtein
from math import sqrt
import sys
import os

# l_distance = Levenshtein.distance

match = {"frequency1", "frequency2", "intersection", "inv-match", "sub-match", "sub-match-norm"}

def e_s(x, is_eu):
    if is_eu:
        return x**2

    else:
        return x


def e_f(x, is_eu):
    if is_eu:
        return sqrt(x)

    else:
        return x


def distance(msg1, msg2, opt=None, is_eu=True):
    if opt in match:
        if opt=="frequency1": # msg2 is a dict, where keys are the words and values are the frequencies in the cluster
            distance = 0
            msg1_word_vector = [t[1] for t in msg1.clues]
            for word in msg1_word_vector:
                if word not in msg2:
                    distance += 1.0
                else:
                    distance += 1.0/(msg2[word] + 1.0)
            return distance # 1/(N2+1)+1/(N3+1)+1/(N8+1)

        if opt=="frequency2": # msg2 is a dict here too
            distance = 0
            msg1_word_vector = [t[1] for t in msg1.clues]
            for word in msg1_word_vector:
                if word not in msg2:
                    distance += 1.0
                else:
                    distance += (msg2[word] + 1.0)
            return 1.0/distance # 1/(N2+N3+N8+3)

        if opt=="intersection":
            msg1_word_vector = [t[1] for t in msg1.clues] # 1x150 vector containing most potent words
            msg2_word_vector = [t[1] for t in msg2.clues]

            # print "msg1_word_vector: ", msg1_word_vector
            # print "msg2_word_vector: ", msg2_word_vector

            # now return the cardinality of their intersection
            return len(set(msg1_word_vector) & set(msg2_word_vector)) # NOTE: can convert lists to sets because all words are unqiue

        msg1.clues.sort() # contains (supposedly) all words in email
        msg2.clues.sort()
        words_1 = msg1.clues
        words_2 = msg2.clues

        i = 0
        j = 0
        counter = 0

        



        while i < len(words_1) and j < len(words_2):
            clue_1 = words_1[i]
            clue_2 = words_2[j]
            if clue_1[0] < clue_2[0]:
                i += 1

            elif clue_1[0] > clue_2[0]:
                j += 1

            else:
                counter += 1
                i += 1
                j += 1
        


        if opt == "inv-match":
            if counter == 0:
                return 2

            else:
                return float(1) / float(counter)

        elif opt == "sub-match":
            return max(len(msg1.clues), len(msg2.clues)) - counter

        elif opt == "sub-match-norm":
            denominator = max(len(msg1.clues), len(msg2.clues))
            try:
                return float(denominator - counter) / denominator

            except ZeroDivisionError:
                if len(msg1.clues) == 0 & len(msg2.clues) == 0:
                    return 0

                else:
                    raise AssertionError

    if opt is None:
        s = 0
        for i in range(min(len(msg1.clues), len(msg2.clues))):
            s += e_s(l_distance(msg1.clues[len(msg1.clues) - 1 - i][1], msg2.clues[len(msg2.clues) - 1 - i][1]), is_eu)

        # This basically adds for overlap; we can try removing this to see
        # if distance is a bit more accurate
        if len(msg1.clues) != len(msg2.clues):
            if len(msg1.clues) > len(msg2.clues):
                for i in range(len(msg1.clues) - len(msg2.clues)):
                    s += e_s(l_distance(msg1.clues[i][1], ""), is_eu)
            else:
                for i in range(len(msg2.clues) - len(msg1.clues)):
                    s += e_s(l_distance(msg2.clues[i][1], ""), is_eu)
        return e_f(s, is_eu)

    if opt == "ac":
        s = 0
        for i in range(min(len(msg1.allclues), len(msg2.allclues))):
            s += e_s(l_distance(msg1.allclues[len(msg1.clues) - 1 - i][1], msg2.allclues[len(msg2.clues) - 1 - i][1]),
                     is_eu)
        if len(msg1.allclues) != len(msg2.allclues):
            if len(msg1.allclues) > len(msg2.allclues):
                for i in range(len(msg1.allclues) - len(msg2.allclues)):
                    s += e_s(l_distance(msg1.allclues[i][1], ""), is_eu)
            else:
                for i in range(len(msg2.allclues) - len(msg1.allclues)):
                    s += e_s(l_distance(msg2.allclues[i][1], ""), is_eu)
        return e_f(s, is_eu)

    if opt == "ac-trunc":
        s = 0
        for i in range(min(len(msg1.allclues), len(msg2.allclues))):
            s += e_s(l_distance(msg1.allclues[i][1], msg2.allclues[i][1]), is_eu)
        return e_f(s, is_eu)

    if opt == "trunc":
        s = 0
        for i in range(min(len(msg1.clues), len(msg2.clues))):
            s += e_s(l_distance(msg1.clues[i][1], msg2.clues[i][1]), is_eu)
        return e_f(s, is_eu)

    if opt == "extreme":
        msg1.clues.sort()
        msg2.clues.sort()
        s = 0

        if len(msg1.clues) >= len(msg2.clues):
            i = 0
            j = len(msg2.clues) - 1

            while i < j:
                s += e_s(l_distance(msg1.clues[i][1], msg2.clues[i][1]), is_eu)
                s += e_s(l_distance(msg1.clues[j - len(msg2.clues) + len(msg1.clues)][1], msg2.clues[j][1]), is_eu)
                i += 1
                j -= 1

            if i == j:
                s += e_s(l_distance(msg1.clues[i][1], msg2.clues[i][1]), is_eu)
                i += 1
                j -= 1

            for i in range(i, j - len(msg2.clues) + len(msg1.clues) + 1):
                s += e_s(l_distance(msg1.clues[i][1], ""), is_eu)

        else:
            i = 0
            j = len(msg1.clues) - 1

            while i < j:
                s += e_s(l_distance(msg2.clues[i][1], msg1.clues[i][1]), is_eu)
                s += e_s(l_distance(msg2.clues[j - len(msg1.clues) + len(msg2.clues)][1], msg1.clues[j][1]), is_eu)
                i += 1
                j -= 1

            if i == j:
                s += e_s(l_distance(msg1.clues[i][1], msg2.clues[i][1]), is_eu)

            for i in range(i, j - len(msg1.clues) + len(msg2.clues) + 1):
                s += e_s(l_distance(msg2.clues[i][1], ""), is_eu)
        return e_f(s, is_eu)

    if opt == "extreme-trunc":
        msg1.allclues.sort()
        msg2.allclues.sort()
        s = 0

        if len(msg1.clues) >= len(msg2.clues):
            i = 0
            j = len(msg2.clues) - 1

            while i < j:
                s += e_s(l_distance(msg1.clues[i][1], msg2.clues[i][1]), is_eu)
                s += e_s(l_distance(msg1.clues[j - len(msg2.clues) + len(msg1.clues)][1], msg2.clues[j][1]), is_eu)
                i += 1
                j -= 1

            if i == j:
                s += e_s(l_distance(msg1.clues[i][1], msg2.clues[i][1]), is_eu)
                i += 1
                j -= 1

        else:
            i = 0
            j = len(msg1.clues) - 1

            while i < j:
                s += e_s(l_distance(msg2.clues[i][1], msg1.clues[i][1]), is_eu)
                s += e_s(l_distance(msg2.clues[j - len(msg1.clues) + len(msg2.clues)][1], msg1.clues[j][1]), is_eu)
                i += 1
                j -= 1

            if i == j:
                s += e_s(l_distance(msg1.clues[i][1], msg2.clues[i][1]), is_eu)
                i += 1
                j -= 1

        return e_f(s, is_eu)

    if opt == "ac-extreme":
        msg1.allclues.sort()
        msg2.allclues.sort()
        s = 0

        if len(msg1.allclues) >= len(msg2.allclues):
            i = 0
            j = len(msg2.allclues) - 1

            while i < j:
                s += e_s(l_distance(msg1.allclues[i][1], msg2.allclues[i][1]), is_eu)
                s += e_s(l_distance(msg1.allclues[j - len(msg2.allclues) + len(msg1.allclues)][1],
                                    msg2.allclues[j][1]), is_eu)
                i += 1
                j -= 1

            if i == j:
                s += e_s(l_distance(msg1.allclues[i][1], msg2.allclues[i][1]), is_eu)
                i += 1
                j -= 1

            for i in range(i, j - len(msg2.allclues) + len(msg1.allclues) + 1):
                s += e_s(l_distance(msg1.allclues[i][1], ""), is_eu)

        else:
            i = 0
            j = len(msg1.allclues) - 1

            while i < j:
                s += e_s(l_distance(msg2.allclues[i][1], msg1.allclues[i][1]), is_eu)
                s += e_s(l_distance(msg2.allclues[j - len(msg1.allclues) + len(msg2.allclues)][1],
                                    msg1.allclues[j][1]), is_eu)
                i += 1
                j -= 1

            if i == j:
                s += e_s(l_distance(msg1.allclues[i][1], msg2.allclues[i][1]), is_eu)

            for i in range(i, j - len(msg1.allclues) + len(msg2.allclues) + 1):
                s += e_s(l_distance(msg2.allclues[i][1], ""), is_eu)
        return e_f(s, is_eu)
