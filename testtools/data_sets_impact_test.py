__author__ = 'Alex'

import os
import sys
import time

sys.path.insert(-1, os.getcwd())
sys.path.insert(-1, os.path.dirname(os.getcwd()))

from spambayes import ActiveUnlearnDriver
from spambayes.Options import options
from spambayes import msgs
from testtools import data_sets as ds

options["TestDriver", "show_histograms"] = False
dir_enumerate = ds.dir_enumerate
seterize = ds.seterize
seconds_to_english = ds.seconds_to_english
hams = ds.hams
spams = ds.spams
set_dirs = ds.set_dirs


def unlearn_stats(au, outfile, data_set, train, test, polluted, total_polluted, total_unpolluted,
                  train_time, clusters=False):
        outfile.write("---------------------------\n")
        outfile.write("Data Set: " + data_set + "\n")
        outfile.write("Vanilla Training: " + str(train[0]) + " ham and " + str(train[1]) + " spam.\n")
        outfile.write("Testing: " + str(test[0]) + " ham and " + str(test[1]) + " spam.\n")
        outfile.write("Pollution Training: " + str(polluted[0]) + " ham and " + str(polluted[1]) +
                      " spam.\n")
        outfile.write("---------------------------\n")
        outfile.write("\n\n")
        outfile.write("CLUSTER AND RATE COUNTS:\n")
        outfile.write("---------------------------\n")

        original_detection_rate = au.driver.tester.correct_classification_rate()

        outfile.write("0: " + str(original_detection_rate) + "\n")

        time_start = time.time()
        cluster_list = au.impact_active_unlearn(outfile, test=True, pollution_set3=True, gold=True)
        time_end = time.time()
        unlearn_time = seconds_to_english(time_end - time_start)
        total_polluted_unlearned = 0
        total_unlearned = 0
        total_unpolluted_unlearned = 0
        final_detection_rate = au.current_detection_rate

        print "\nTallying up final counts...\n"
        for cluster in cluster_list:
            cluster = cluster[1]
            total_unlearned += cluster.size
            total_polluted_unlearned += cluster.target_set3()
            total_unpolluted_unlearned += (cluster.size - cluster.target_set3())

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
        outfile.write("Time for training:\n")
        outfile.write(train_time + "\n")
        outfile.write("Time for unlearning:\n")
        outfile.write(unlearn_time)

        if clusters:
            return cluster_list


def main():
    num_data_sets = len(hams)
    assert(len(hams) == len(spams))
    sets = [1]

    for i in sets:
        ham = hams[i]
        spam = spams[i]
        data_set = set_dirs[i]

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
                                                     distance_opt="inv-match", all_opt=True,
                                                     update_opt="hybrid", greedy_opt=False)

            time_2 = time.time()
            train_time = seconds_to_english(time_2 - time_1)
            print "Train time:", train_time, "\n"

            with open("C:\\Users\\Alex\\Desktop\\unpollute_stats\\Yang_Data_Sets (lazy impact)\\" + data_set +
                      ".txt", 'w') as outfile:
                try:
                    unlearn_stats(au, outfile, data_set, [train_ham, train_spam], [test_ham, test_spam],
                                  [ham_polluted, spam_polluted], total_polluted, total_unpolluted,
                                  train_time)

                except KeyboardInterrupt:
                    outfile.flush()
                    sys.exit()

        except KeyboardInterrupt:
            sys.exit()

if __name__ == "__main__":
    main()
