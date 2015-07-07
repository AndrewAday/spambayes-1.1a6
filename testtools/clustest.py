import os
import sys
from random import choice

sys.path.insert(-1, os.getcwd())
sys.path.insert(-1, os.path.dirname(os.getcwd()))

from spambayes import ActiveUnlearnDriver
from spambayes.Options import get_pathname_option
from spambayes import msgs
from dictionarywriter import DictionaryWriter
from InjectionPollution import InjectionPolluter
from mislabeledfilemover import MislabeledFileMover

def main():

    ham = [get_pathname_option("TestDriver", "ham_directories") % i for i in range(1, 4)]
    spam = [get_pathname_option("TestDriver", "spam_directories") % i for i in range(1, 4)]

    au = ActiveUnlearnDriver.ActiveUnlearner([msgs.HamStream(ham[0], [ham[0]]), msgs.HamStream(ham[2], [ham[2]])],          #ham
                                             [msgs.SpamStream(spam[0], [spam[0]]), msgs.SpamStream(spam[2], [spam[2]])],    #spam
                                             msgs.HamStream(ham[1], [ham[1]]),                                             #real ham
                                             msgs.SpamStream(spam[1], [spam[1]]),                                          #real spam
                                             "extreme")                                                                    #opt

    msg = choice(au.driver.tester.train_examples[3])    # Randomly chosen from Ham Set3
    original_rate = au.driver.tester.correct_classification_rate()
    detection_rates = []
    target_cluster_rates = []

    for size in range(300, 6005, 300):
        cluster = ActiveUnlearnDriver.Cluster(msg, size)
        print "Clustering with size " + str(cluster.size) + "..."
        detection_rates.append(au.detect_rate(cluster))
        target_cluster_rates.append(float(au.target_set3(cluster)) / float(cluster.size))

    with open("/Users/AlexYang/Desktop/clusterstats.txt", 'w') as outfile:
        outfile.write(msg.tag + "\n")
        outfile.write("Original Rate: " + str(original_rate))
        outfile.write("\nDetection Rates: \n")
        for rate in l:
            outfile.write(str(rate) + "\n")
        outfile.write("\nPercentage of Injected Polluted Clustered:\n")
        for count in c_l:
            outfile.write(str(count) + "\n")
        for msg in cl:
            outfile.write("\n" + msg.tag + "\n")

if __name__ == "__main__":
    main()
