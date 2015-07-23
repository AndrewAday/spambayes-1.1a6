from dictionarycount import write_dictionary_sets, reset
from spambayes import TestDriver, msgs
from spambayes.Options import get_pathname_option
from tabulate import tabulate
from mislabeledfilemover import MislabeledFileMover

def test():

    y = []

    for i in range(0, 100, 501, 100):
        y.append(i)
    y += [750, 1000]

    hamdirs = [get_pathname_option("TestDriver", "ham_directories") % i for i in range(1, 4)]
    spamdirs = [get_pathname_option("TestDriver", "spam_directories") % i for i in range(1, 4)]

    d = TestDriver.Driver()
    d.new_classifier()
    d.train(msgs.HamStream(hamdirs[0], [hamdirs[0]]),
            msgs.SpamStream(spamdirs[0], [spamdirs[0]]))

    detection_rates = []
    for y_val in y:
        reset()
        write_dictionary_sets(x=0.3, y=y_val)

        d.train(msgs.HamStream(hamdirs[2], [hamdirs[2]]),
                msgs.SpamStream(spamdirs[2], [spamdirs[2]]))
        d.test(msgs.HamStream(hamdirs[1], [hamdirs[1]]),
               msgs.SpamStream(spamdirs[1], [spamdirs[1]]))

        rate = d.tester.correct_classification_rate()
        detection_rates.append(rate)

        d.untrain(msgs.HamStream(hamdirs[2], [hamdirs[2]]),
                  msgs.SpamStream(spamdirs[2], [spamdirs[2]]))

        reset()
    outfile = open("rates.txt", 'w')
    outfile.write(tabulate({"# of Sets": y, "Detection Rate": detection_rates},
                           headers="keys"))

def main():

    test()

if __name__ == "__main__":
    test()