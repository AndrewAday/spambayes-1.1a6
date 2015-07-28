import os
import sys
import time

sys.path.insert(-1, os.getcwd())
sys.path.insert(-1, os.path.dirname(os.getcwd()))

from spambayes import ActiveUnlearnDriver
from spambayes.Options import get_pathname_option
from spambayes import msgs
from testtools import dictionarywriter, mislabeledfilemover


def main():

    ham = [get_pathname_option("TestDriver", "ham_directories") % i for i in range(1, 5)]
    spam = [get_pathname_option("TestDriver", "spam_directories") % i for i in range(1, 5)]

    """
    d = dictionarywriter.DictionaryWriter(600, words=False, wordsEn=False)
    d.write()
    """
    """
    m = mislabeledfilemover.MislabeledFileMover(1500, train_dir=2, test_dir=1, dest_dir=4)
    m.print_filelist()
    m.random_move_file()
    """
    keep_going = True
    trial_number = 1

    try:
        time_1 = time.time()
        au = ActiveUnlearnDriver.ActiveUnlearner([msgs.HamStream(ham[1], [ham[1]]), msgs.HamStream(ham[2], [ham[2]])],       # Training Ham
                                                 [msgs.SpamStream(spam[1], [spam[1]]), msgs.SpamStream(spam[2], [spam[2]])], # Training Spam
                                                 msgs.HamStream(ham[0], [ham[0]]),      # Testing Ham
                                                 msgs.SpamStream(spam[0], [spam[0]]),   # Testing Spam
                                                 )

        time_2 = time.time()
        print "Train time:", time_2 - time_1, "\n"
        while keep_going:
            with open("C:\Users\Alex\Desktop\unpollute_stats\unlearn_stats" + str(trial_number) + ".txt", 'w') as outfile:
                try:
                    outfile.write("CLUSTER AND RATE COUNTS:\n")
                    outfile.write("---------------------------\n")

                    original_detection_rate = au.driver.tester.correct_classification_rate()

                    """
                    answer = raw_input("\nInitial detection rate is " + str(original_detection_rate) +
                                       "; continue (y/n)?\n")
                    valid_input = False

                    while not valid_input:
                        if answer == "n":
                            m.reset()
                            sys.exit()

                        elif answer == "y":
                            valid_input = True

                        else:
                            print "\nPlease enter either y or n.\n"
                    """

                    outfile.write("0: " + str(original_detection_rate) + "\n")

                    time_start = time.time()
                    cluster_list = au.brute_force_active_unlearn(outfile, test=True, center_iteration=False, gold=True)
                    time_end = time.time()
                    brute_force_time = time_end - time_start
                    total_polluted_unlearned = 0
                    total_unlearned = 0
                    total_unpolluted_unlearned = 0
                    final_detection_rate = au.driver.tester.correct_classification_rate()

                    print "\nTallying up final counts...\n"
                    for cluster in cluster_list:
                        total_unlearned += cluster.size + 1
                        total_polluted_unlearned += cluster.target_set3()
                        total_unpolluted_unlearned += cluster.size + 1 - total_polluted_unlearned

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
                    outfile.write(str(float(total_polluted_unlearned) / 1200) + "\n")
                    outfile.write("Time for unlearning:\n")
                    outfile.write(str(brute_force_time))

                except KeyboardInterrupt:
                    outfile.flush()
                    os.fsync(outfile)
                    """
                    m.reset()
                    """
                    sys.exit()

            answer = raw_input("\nKeep going (y/n)? You have performed " + str(trial_number) + " trial(s) so far. ")
            valid_input = False

            while not valid_input:
                if answer == "n":
                    keep_going = False
                    valid_input = True

                elif answer == "y":
                    for cluster in cluster_list:
                        au.learn(cluster)
                    au.init_ground()
                    trial_number += 1
                    valid_input = True

                else:
                    answer = raw_input("Please enter either y or n. ")

    except KeyboardInterrupt:
        """
        m.reset()
        """
        sys.exit()

if __name__ == "__main__":
    main()
