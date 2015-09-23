__author__ = 'Alex'

import os
import sys
import time

sys.path.insert(-1, os.getcwd())
sys.path.insert(-1, os.path.dirname(os.getcwd()))

from spambayes import ActiveUnlearnDriver
from spambayes.Options import options # Imports global options variable from spambayes/Options.py
from spambayes import msgs
from testtools import data_sets as ds # File manager for test/training data
from testtools.io_locations import dest

# Set options global for spambayes
options["TestDriver", "show_histograms"] = False

# Reassign the functions in ds
dir_enumerate = ds.dir_enumerate
seterize = ds.seterize
seconds_to_english = ds.seconds_to_english

# variables contain all ham data and all spam data
hams = ds.hams
spams = ds.spams
# Schema:
#[
    # [
    #   '/Users/andrewaday/Downloads/Data Sets/DictionarySets-1.1/Ham/Set1', 
    #   '/Users/andrewaday/Downloads/Data Sets/DictionarySets-1.1/Ham/Set2', 
    #   '/Users/andrewaday/Downloads/Data Sets/DictionarySets-1.1/Ham/Set3'
    # ], ...
#]

set_dirs = ds.set_dirs # Array containing names of all parent data directories

pollution_set3 = True #True if Set3 file contains polluted data


def unlearn_stats(au, outfile, data_set, train, test, polluted, total_polluted, total_unpolluted,
                  train_time, clusters=False, vanilla=None, noisy_clusters=False):
        """Runs an unlearn algorithm on an ActiveUnlearner and prints out the resultant stats."""
        outfile.write("---------------------------\n")
        outfile.write("Data Set: " + data_set + "\n")
        outfile.write("Vanilla Training: " + str(train[0]) + " ham and " + str(train[1]) + " spam.\n")
        outfile.write("Testing: " + str(test[0]) + " ham and " + str(test[1]) + " spam.\n")
        outfile.write("Pollution Training: " + str(polluted[0]) + " ham and " + str(polluted[1]) +
                      " spam.\n")
        if vanilla is not None:
            outfile.write("Vanilla Detection Rate: " + str(vanilla[0]) + ".\n")
        outfile.write("---------------------------\n")
        outfile.write("\n\n")
        outfile.write("CLUSTER AND RATE COUNTS:\n")
        outfile.write("---------------------------\n")

        original_detection_rate = au.driver.tester.correct_classification_rate()

        outfile.write("0: " + str(original_detection_rate) + "\n")

        time_start = time.time()
        cluster_list = au.impact_active_unlearn(outfile, test=True, pollution_set3=pollution_set3, gold=True)
        time_end = time.time()
        unlearn_time = seconds_to_english(time_end - time_start)
        total_polluted_unlearned = 0
        total_unlearned = 0
        total_unpolluted_unlearned = 0
        total_noisy_unlearned = 0
        final_detection_rate = au.current_detection_rate
        noise = []

        if vanilla is not None:
            noise = noisy_data_check(find_pure_clusters(cluster_list, ps_3=pollution_set3),
                                     vanilla[1])

        print "\nTallying up final counts...\n"
        for cluster in cluster_list:
            cluster = cluster[1]
            total_unlearned += cluster.size
            total_polluted_unlearned += cluster.target_set3()
            total_unpolluted_unlearned += (cluster.size - cluster.target_set3())
            if cluster in noise:
                total_noisy_unlearned += cluster.size

        outfile.write("\nSTATS\n")
        outfile.write("---------------------------\n")
        outfile.write("Initial Detection Rate: " + str(original_detection_rate) + "\n")
        outfile.write("Final Detection Rate: " + str(final_detection_rate) + "\n")
        outfile.write("Total Unlearned:\n")
        outfile.write(str(total_unlearned) + "\n")
        outfile.write("Polluted Percentage of Unlearned:\n")
        outfile.write(str(float(total_polluted_unlearned) / float(total_unlearned)) + "\n")
        outfile.write("Unpolluted Percentage of Unlearned:\n")
        outfile.write(str(float(total_unpolluted_unlearned) / float(total_unlearned)) + "\n")
        outfile.write("Percentage of Polluted Unlearned:\n")
        outfile.write(str(float(total_polluted_unlearned) / float(total_polluted)) + "\n")
        outfile.write("Percentage of Unpolluted Unlearned:\n")
        outfile.write(str(float(total_unpolluted_unlearned) / float(total_unpolluted)) + "\n")
        if noisy_clusters:
            outfile.write("Percentage of Noisy Data in Unpolluted Unlearned:\n")
            outfile.write(str(float(total_noisy_unlearned) / float(total_unpolluted_unlearned)) + "\n")
        outfile.write("Time for training:\n")
        outfile.write(train_time + "\n")
        outfile.write("Time for unlearning:\n")
        outfile.write(unlearn_time)

        if clusters:
            return cluster_list


def find_pure_clusters(cluster_list, ps_3):
    pure_clusters = []
    for cluster in cluster_list:
        cluster = cluster[1]
        if ps_3:
            if cluster.target_set3() == 0:
                pure_clusters.append(cluster)

        else:
            if cluster.target_set4() == 0:
                pure_clusters.append(cluster)

    return pure_clusters


def noisy_data_check(pure_clusters, v_au):
    noisy_clusters = []
    original_detection_rate = v_au.current_detection_rate
    for cluster in pure_clusters:
        v_au.unlearn(cluster)
        v_au.init_ground(True)
        new_detection_rate = v_au.driver.tester.correct_classification_rate()
        if new_detection_rate > original_detection_rate:
            noisy_clusters.append(cluster)

        v_au.learn(cluster)

    return noisy_clusters


def main():
    sets = [11, 12, 13, 14, 15]

    for i in sets:
        ham = hams[i]
        spam = spams[i]
        data_set = set_dirs[i]

        if i > 10:
            ham_test = ham[1]
            spam_test = spam[1]

            ham_train = ham[0]
            spam_train = spam[0]

        else:
            ham_test = ham[0]
            spam_test = spam[0]

            ham_train = ham[1]
            spam_train = spam[1]

        ham_p = ham[2]
        spam_p = spam[2]

        ham_polluted = dir_enumerate(ham_p)
        spam_polluted = dir_enumerate(spam_p)
        train_ham = dir_enumerate(ham_train)
        train_spam = dir_enumerate(spam_train)
        test_ham = dir_enumerate(ham_test)
        test_spam = dir_enumerate(spam_test)
        total_polluted = ham_polluted + spam_polluted
        total_unpolluted = train_ham + train_spam

        try:
            time_1 = time.time()
            au = ActiveUnlearnDriver.ActiveUnlearner([msgs.HamStream(ham_train, [ham_train]),
                                                      msgs.HamStream(ham_p, [ham_p])],        # Training Ham
                                                     [msgs.SpamStream(spam_train, [spam_train]),
                                                      msgs.SpamStream(spam_p, [spam_p])],     # Training Spam
                                                     msgs.HamStream(ham_test, [ham_test]),          # Testing Ham
                                                     msgs.SpamStream(spam_test, [spam_test]),       # Testing Spam
                                                     distance_opt="inverse", all_opt=True,
                                                     update_opt="hybrid", greedy_opt=False)

            v_au = ActiveUnlearnDriver.ActiveUnlearner([msgs.HamStream(ham_train, [ham_train]), []],
                                                       [msgs.SpamStream(spam_train, [spam_train]), []],
                                                       msgs.HamStream(ham_test, [ham_test]),
                                                       msgs.SpamStream(spam_test, [spam_test]))

            vanilla_detection_rate = v_au.current_detection_rate

            time_2 = time.time()
            train_time = seconds_to_english(time_2 - time_1)
            print "Train time:", train_time, "\n"

            

            with open(dest + data_set + " (unlearn_stats).txt", 'w') as outfile:
                try:
                    unlearn_stats(au, outfile, data_set, [train_ham, train_spam], [test_ham, test_spam],
                                  [ham_polluted, spam_polluted], total_polluted, total_unpolluted,
                                  train_time, vanilla=[vanilla_detection_rate, v_au], noisy_clusters=True)

                except KeyboardInterrupt:
                    outfile.flush()
                    sys.exit()

            # In the hopes of keeping RAM down between iterations
            del au
            del v_au

        except KeyboardInterrupt:
            sys.exit()

if __name__ == "__main__":
    # main()
    print dest
