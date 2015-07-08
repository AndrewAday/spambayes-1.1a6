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
    from dictionarysplicer import splice_set
    """

    ham = [get_pathname_option("TestDriver", "ham_directories") % i for i in range(1, 5)]
    spam = [get_pathname_option("TestDriver", "spam_directories") % i for i in range(1, 5)]
    """
    DictionaryWriter(200).write()
    splice_set(3)
    """
    au = ActiveUnlearnDriver.ActiveUnlearner([msgs.HamStream(ham[1], [ham[1]]), msgs.HamStream(ham[2], [ham[2]])],
                                             [msgs.SpamStream(spam[1], [spam[1]]), msgs.SpamStream(spam[2], [spam[2]])],
                                             msgs.HamStream(ham[0], [ham[0]]),
                                             msgs.SpamStream(spam[0], [spam[0]]),
                                             )

    msg = choice(au.driver.tester.train_examples[2])
    c_d = []
    c_l = []
    c_s = []
    original_rate = au.driver.tester.correct_classification_rate()

    cluster_size = 5
    c_s.append(5)
    cl = ActiveUnlearnDriver.Cluster(msg, cluster_size, au, "extreme")
    c_l.append(float(cl.target_spam()) / float(cluster_size))
    c_d.append(au.start_detect_rate(cl))

    for i in range(1, 20):
        cluster_size += 5
        c_s.append(cluster_size)
        print "Clustering with size", cluster_size, "..."
        c_d.append(au.continue_detect_rate(cl, 5))
        c_l.append(float(cl.target_spam()) / float(cluster_size))

    for i in range(1, 30):
        cluster_size += 100
        c_s.append(cluster_size)
        print "Clustering with size", cluster_size, "..."
        c_d.append(au.continue_detect_rate(cl, 100))
        c_l.append(float(cl.target_spam()) / float(cluster_size))

    with open("C:\Users\Alex\Desktop\clusterstats.txt", 'w') as outfile:
        outfile.write("Clustered around: " + msg.tag)
        outfile.write("\nOriginal Rate: " + str(original_rate) + "\n")

        outfile.write(tabulate({"Cluster Sizes": c_s,
                                "Detection Rates": c_d,
                                "% of Targets Clustered": c_l},
                               headers="keys", tablefmt="plain"))


if __name__ == "__main__":
    main()
