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

    """
    from dictionarywriter import DictionaryWriter
    from dictionarysplicer import splice_set
    """

    cluster = ActiveUnlearnDriver.Cluster

    ham = [get_pathname_option("TestDriver", "ham_directories") % i for i in range(1, 5)]
    spam = [get_pathname_option("TestDriver", "spam_directories") % i for i in range(1, 5)]
    """
    DictionaryWriter(200).write()
    splice_set(3)
    """
    au = ActiveUnlearnDriver.ActiveUnlearner([msgs.HamStream(ham[1], [ham[1]]), None],
                                             [msgs.SpamStream(spam[1], [spam[1]]), msgs.SpamStream(spam[2], [spam[2]])],
                                             msgs.HamStream(ham[0], [ham[0]]),
                                             msgs.SpamStream(spam[0], [spam[0]]),
                                             )

    msg = choice(au.driver.tester.train_examples[2])
    l = []
    c_l = []
    original_rate = au.driver.tester.correct_classification_rate()

    for i in range(5, 100, 5):
        cluster_size = i

        cl = ActiveUnlearnDriver.Cluster(msg, cluster_size, "extreme")
        c_l.append(float(cl.target_spam()) / float(cluster_size))
        print "Clustering with size", i, "..."
        l.append(au.detect_rate(cl))

    for i in range(100, 3000, 100):
        cluster_size = i

        cl = cluster(msg, cluster_size, "extreme")
        c_l.append(float(cl.target_spam()) / float(cluster_size))
        print "Clustering with size", i, "..."
        l.append(au.detect_rate(cl))

    with open("C:\Users\Alex\Desktop\clusterstats.txt", 'w') as outfile:
        outfile.write(str(original_rate))
        outfile.write(str(l))
        outfile.write(str(c_l))
        outfile.close()

if __name__ == "__main__":
    main()
