__author__ = 'Alex'

import os
import sys
import time

sys.path.insert(-1, os.getcwd())
sys.path.insert(-1, os.path.dirname(os.getcwd()))

from spambayes import ActiveUnlearnDriver
from spambayes.Options import options
from spambayes import msgs

options["TestDriver", "show_histograms"] = False


def seterize(main_dir, sub_dir, is_spam, n):
    if is_spam:
        parent_dir = main_dir + "\\" + sub_dir + "\\" + "Spam" + "\\" + "Set%d"

    else:
        parent_dir = main_dir + "\\" + sub_dir + "\\" + "Ham" + "\\" + "Set%d"

    return [parent_dir % i for i in range(1, n + 1)]


def dir_enumerate(dir_name):
    return len([name for name in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, name))])


def seconds_to_english(seconds):
    seconds_trunc_1 = int(seconds // 60) * 60
    s = seconds - seconds_trunc_1
    seconds -= s
    seconds /= 60
    seconds_trunc_2 = int(seconds // 60) * 60
    m = int(seconds - seconds_trunc_2)
    h = seconds_trunc_2 / 60
    return str(h) + " hours, " + str(m) + " minutes, and " + str(s) + " seconds."


def main():

    data_sets_dir = "C:\\Users\\Alex\\Downloads\\Data Sets"
    set_dirs = ["DictionarySets-1.1", "DictionarySets-1.2", "DictionarySets-2.1", "DictionarySets-2.2",
                "DictionarySets-3.1", "Mislabeled-Big", "Mislabeled-Both-1.1", "Mislabeled-Both-1.2",
                "Mislabeled-Both-2.1", "Mislabeled-Both-2.2", "Mislabeled-Both-3.1", "Mislabeled-HtoS-1.1",
                "Mislabeled-HtoS-1.2", "Mislabeled-HtoS-1.3", "Mislabeled-HtoS-1.4", "Mislabeled-HtoS-1.5",
                "Mislabeled-StoH-1.1", "Mislabeled-StoH-1.2", "Mislabeled-StoH-1.3", "Mislabeled-StoH-2.1",
                "Mislabeled-StoH-2.2"]

    hams = [seterize(data_sets_dir, set_dir, False, 3) for set_dir in set_dirs]
    spams = [seterize(data_sets_dir, set_dir, True, 3) for set_dir in set_dirs]

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
                                                     )

            time_2 = time.time()
            train_time = seconds_to_english(time_2 - time_1)
            print "Train time:", train_time, "\n"

            with open("C:\\Users\\Alex\\Desktop\\unpollute_stats\\Yang_Data_Sets (lazy impact)\\" + data_set +
                      ".txt", 'w') as outfile:
                try:
                    outfile.write("---------------------------\n")
                    outfile.write("Data Set: " + set_dirs[i] + "\n")
                    outfile.write("Vanilla Training: " + str(train_ham) + " ham and " + str(train_spam) + " spam.\n")
                    outfile.write("Testing: " + str(test_ham) + " ham and " + str(test_spam) + " spam.\n")
                    outfile.write("Pollution Training: " + str(ham_polluted) + " ham and " + str(spam_polluted) +
                                  " spam.\n")
                    outfile.write("---------------------------\n")
                    outfile.write("\n\n")
                    outfile.write("CLUSTER AND RATE COUNTS:\n")
                    outfile.write("---------------------------\n")

                    original_detection_rate = au.driver.tester.correct_classification_rate()

                    outfile.write("0: " + str(original_detection_rate) + "\n")

                    time_start = time.time()
                    cluster_list = au.greatest_impact_active_unlearn(outfile, test=True, pollution_set3=True, gold=True,
                                                                     unlearn_method="lazy")
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

                except KeyboardInterrupt:
                    outfile.flush()
                    os.fsync(outfile)
                    sys.exit()

        except KeyboardInterrupt:
            sys.exit()

if __name__ == "__main__":
    main()
