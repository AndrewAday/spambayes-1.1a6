__author__ = 'Alex'


def main():
    import os
    import sys

    sys.path.insert(-1, os.getcwd())
    sys.path.insert(-1, os.path.dirname(os.getcwd()))

    from spambayes import ActiveUnlearnDriver
    from spambayes.Options import get_pathname_option
    from spambayes import msgs

    """
    from dictionarywriter import DictionaryWriter
    """

    ham = [get_pathname_option("TestDriver", "ham_directories") % i for i in range(1, 5)]
    spam = [get_pathname_option("TestDriver", "spam_directories") % i for i in range(1, 5)]
    """
    DictionaryWriter(600).write()
    """

    keep_going = True
    trial_number = 1

    au = ActiveUnlearnDriver.ActiveUnlearner([msgs.HamStream(ham[1], [ham[1]]),
                                              msgs.HamStream(ham[2], [ham[2]])],
                                             [msgs.SpamStream(spam[1], [spam[1]]),
                                              msgs.SpamStream(spam[3], [spam[3]])],
                                             msgs.HamStream(ham[0], [ham[0]]),
                                             msgs.SpamStream(spam[0], [spam[0]]),
                                             )
    while keep_going:
        with open("C:\Users\Alex\Desktop\unlearn_stats" + str(trial_number) + ".txt", 'w') as outfile:
            outfile.write("CLUSTER AND RATE COUNTS:\n")
            outfile.write("---------------------------\n")

            original_detection_rate = au.driver.tester.correct_classification_rate()

            outfile.write("0: " + str(original_detection_rate))

            cluster_list = au.active_unlearn(outfile, True)
            total_polluted_unlearned = 0
            total_unlearned = 0
            total_unpolluted_unlearned = 0
            final_detection_rate = au.driver.tester.correct_classification_rate()

            print "\nTallying up final counts...\n"
            for cluster in cluster_list:
                total_unlearned += cluster.size + 1
                total_polluted_unlearned += cluster.target_set3
                total_unpolluted_unlearned += cluster.size + 1 - total_polluted_unlearned

            outfile.write("STATS\n")
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
            outfile.write(str(float(total_polluted_unlearned) / 1200))

        answer = raw_input("Keep going (y/n)? You have performed " + str(trial_number) + " trial(s) so far. ")
        valid_input = False

        while valid_input:
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
                print "Please enter either y or n."

if __name__ == "__main__":
    main()
