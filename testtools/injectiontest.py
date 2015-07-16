import os
import sys

sys.path.insert(-1, os.getcwd())
sys.path.insert(-1, os.path.dirname(os.getcwd()))

from spambayes.Options import get_pathname_option
from spambayes import msgs, TestDriver
from InjectionPollution import InjectionPolluter
from tabulate import tabulate
from random import choice

def main():

    ham = [get_pathname_option("TestDriver", "ham_directories") % i for i in range(1, 4)]
    spam = [get_pathname_option("TestDriver", "spam_directories") % i for i in range(1, 4)]
    injected = get_pathname_option("TestDriver", "spam_directories") % 3

    sizes = [150, 1000, 10000]
    #for i in range(1000, 15001, 1000):
     #   sizes.append(i)

    d = TestDriver.Driver()
    d.new_classifier()
    d.train(msgs.HamStream(ham[0], [ham[0]]),
            msgs.SpamStream(spam[0], [spam[0]]))

    detection_rates = []
    clues_list = []
    for size in sizes:
        injector = InjectionPolluter(6000, size, 3)
        injector.injectfeatures()

        d.train(msgs.HamStream(ham[2], [ham[2]]),
                msgs.SpamStream(spam[2], [spam[2]]))
        d.test(msgs.HamStream(ham[1], [ham[1]]),
               msgs.SpamStream(spam[1], [spam[1]]))

        msg_list = []
        for msg in msgs.SpamStream(spam[2], [spam[2]]):
            msg_list.append(msg)

        clues = d.classifier._getclues(msg_list[0])
        clues_list.append(clues)

        detection_rate = d.tester.correct_classification_rate()
        detection_rates.append(detection_rate)

        d.untrain(msgs.HamStream(ham[2], [ham[2]]),
                  msgs.SpamStream(spam[2], [spam[2]]))

        injector.reset()

    with open("/Users/AlexYang/Desktop/clues.txt", 'w') as outfile:

        probs_list = []
        words_list = []

        for clues in clues_list:
            probs = []
            words = []
            for clue in clues:
                probs.append(clue[0])
                words.append(clue[1])

            probs_list.append(probs)
            words_list.append(words)

        table = {}

        for i in range(len(sizes)):
            table[str(sizes[i]) + " Words"] = words_list[i]
            table[str(sizes[i]) + " Probs"] = probs_list[i]

        outfile.write(tabulate(table, headers="keys"))

    with open("/Users/AlexYang/Desktop/clusterstats.txt", 'w') as outfile:

        outfile.write(tabulate({"# of Injected Words": sizes,
                                "Detection Rates": detection_rates},
                               headers="keys", tablefmt="plain"))

if __name__ == "__main__":
    main()
