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
from tabulate import tabulate

def main():

    ham = [get_pathname_option("TestDriver", "ham_directories") % i for i in range(1, 4)]
    spam = [get_pathname_option("TestDriver", "spam_directories") % i for i in range(1, 4)]

    i = InjectionPolluter(1200)
    i.injectfeatures()

    au = ActiveUnlearnDriver.ActiveUnlearner([msgs.HamStream(ham[0], [ham[0]]), msgs.HamStream(ham[2], [ham[2]])],
                                             [msgs.SpamStream(spam[0], [spam[0]]), msgs.SpamStream(spam[2], [spam[2]])],
                                             msgs.HamStream(ham[1], [ham[1]]), msgs.SpamStream(spam[1], [spam[1]]))

    msg = choice(au.driver.tester.train_examples[3])    # Randomly chosen from Ham Set3
    original_rate = au.driver.tester.correct_classification_rate()
    cluster_sizes = []
    detection_rates = []
    target_cluster_rates = []

    for size in range(5, 100, 5):
        cluster = ActiveUnlearnDriver.Cluster(msg, size, au, "extreme")
        print "Clustering with size " + str(cluster.size) + "..."
        cluster_sizes.append(size)
        detection_rates.append(au.detect_rate(cluster))
        target_cluster_rates.append(float(cluster.target_set3()) / float(cluster.size))

    with open("/Users/AlexYang/Desktop/clusterstats.txt", 'w') as outfile:

        outfile.write("Clustered around: " + msg.tag)
        outfile.write("\nOriginal Rate: " + str(original_rate) + "\n")

        outfile.write(tabulate({"Cluster Sizes": cluster_sizes,
                                "Detection Rates": detection_rates,
                                "% of Targets Clustered": target_cluster_rates},
                               headers="keys", tablefmt="plain"))

if __name__ == "__main__":
    main()
