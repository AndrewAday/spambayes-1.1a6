__author__ = 'Alex'


def main():
    import os
    import sys
    from random import choice

    sys.path.insert(-1, os.getcwd())
    sys.path.insert(-1, os.path.dirname(os.getcwd()))

    from spambayes import ActiveUnlearnDriver
    from spambayes.Options import get_pathname_option
    from spambayes import msgs
    from tabulate import tabulate

    """
    from dictionarywriter import DictionaryWriter
    """

    ham = [get_pathname_option("TestDriver", "ham_directories") % i for i in range(1, 5)]
    spam = [get_pathname_option("TestDriver", "spam_directories") % i for i in range(1, 5)]
    """
    DictionaryWriter(600).write()
    """
    au = ActiveUnlearnDriver.ActiveUnlearner([msgs.HamStream(ham[1], [ham[1]]), msgs.HamStream(ham[2], [ham[2]])],
                                             [msgs.SpamStream(spam[1], [spam[1]]), msgs.SpamStream(spam[2], [spam[2]])],
                                             msgs.HamStream(ham[0], [ham[0]]),
                                             msgs.SpamStream(spam[0], [spam[0]]),
                                             )

    au_v = ActiveUnlearnDriver.ActiveUnlearner([msgs.HamStream(ham[1], [ham[1]])],
                                               [msgs.SpamStream(spam[1], [spam[1]])],
                                               msgs.HamStream(ham[0], [ham[0]]),
                                               msgs.SpamStream(spam[0], [spam[0]]),
                                               )

    msg = choice(au_v.driver.tester.train_examples[0])

    cluster_detection_rates = []
    cluster_detection_rates_v = []

    cluster_spam_rates = []
    cluster_spam_rates_v = []

    cluster_sizes = []

    original_rate = au.driver.tester.correct_classification_rate()
    original_rate_v = au_v.driver.tester.correct_classification_rate()

    cluster_size = 5
    cluster_sizes.append(5)
    print "Clustering with size", cluster_size, "..."

    cl = ActiveUnlearnDriver.Cluster(msg, cluster_size, au, "extreme")
    cl_v = ActiveUnlearnDriver.Cluster(msg, cluster_size, au_v, "extreme")

    cluster_spam_rates.append(float(cl.target_spam()) / float(cluster_size))
    cluster_spam_rates_v.append(float(cl_v.target_spam()) / float(cluster_size))

    cluster_detection_rates.append(au.start_detect_rate(cl))
    cluster_detection_rates_v.append(au_v.start_detect_rate(cl_v))

    for i in range(1, 20):
        cluster_size += 5
        cluster_sizes.append(cluster_size)
        print "Clustering with size", cluster_size, "..."

        cluster_detection_rates.append(au.continue_detect_rate(cl, 5))
        cluster_detection_rates_v.append(au_v.continue_detect_rate(cl_v, 5))

        cluster_spam_rates.append(float(cl.target_spam()) / float(cluster_size))
        cluster_spam_rates_v.append(float(cl_v.target_spam()) / float(cluster_size))

    for i in range(1, 30):
        cluster_size += 100
        cluster_sizes.append(cluster_size)
        print "Clustering with size", cluster_size, "..."

        cluster_detection_rates.append(au.continue_detect_rate(cl, 100))
        cluster_detection_rates_v.append(au_v.continue_detect_rate(cl_v, 100))

        cluster_spam_rates.append(float(cl.target_spam()) / float(cluster_size))
        cluster_spam_rates_v.append(float(cl_v.target_spam()) / float(cluster_size))

    with open("C:\Users\Alex\Desktop\clusterstats.txt", 'w') as outfile:
        outfile.write("POLLUTED MACHINE")

        outfile.write("--------------------------\n")

        outfile.write("Clustered around: " + msg.tag)

        outfile.write("--------------------------\n")

        outfile.write("Detection Rates:\n")
        outfile.write(str(original_rate) + "\n")

        for item in cluster_detection_rates:
            outfile.write(str(item) + "\n")

        outfile.write("--------------------------\n")

        outfile.write("Spam Rate:\n")
        for item in cluster_spam_rates:
            outfile.write(str(item) + "\n")

    with open("C:\Users\Alex\Desktop\clusterstats_v.txt", 'w') as outfile:
        outfile.write("VANILLA MACHINE")

        outfile.write("--------------------------\n")

        outfile.write("Clustered around: " + msg.tag)

        outfile.write("--------------------------\n")

        outfile.write("Detection Rates:\n")
        outfile.write(str(original_rate_v) + "\n")

        for item in cluster_detection_rates_v:
            outfile.write(str(item) + "\n")

        outfile.write("--------------------------\n")

        outfile.write("Spam Rate:\n")
        for item in cluster_spam_rates_v:
            outfile.write(str(item) + "\n")

if __name__ == "__main__":
    main()
