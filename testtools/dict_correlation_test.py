from __future__ import generators

import os
import sys

sys.path.insert(-1, os.getcwd())
sys.path.insert(-1, os.path.dirname(os.getcwd()))

from spambayes.Options import options, get_pathname_option
from spambayes import msgs, Distance, ActiveUnlearnDriver
from testtools import benignfilemover, mislabeledfilemover, dictionarywriter, dictionarysplicer
from scipy.stats import pearsonr
from math import sqrt

program = sys.argv[0]


def usage(code, msg=''):
    """Print usage message and sys.exit(code)."""
    if msg:
        print >> sys.stderr, msg
        print >> sys.stderr
    print >> sys.stderr, __doc__ % globals()
    sys.exit(code)


def drive():
    print options.display()

    spam = [get_pathname_option("TestDriver", "spam_directories") % i for i in range(1, 5)]
    ham = [get_pathname_option("TestDriver", "ham_directories") % i for i in range(1, 5)]
    dictionarysplicer.splice_set(3, 4)
    
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
        with open("C:\Users\Alex\Desktop\dict_correlation_stats.txt", 'w') as outfile:
            chosen = set()
            current = au.select_initial()
            cluster = au.determine_cluster(current)
            chosen.add(current)
            au.driver.test(au.testing_ham, au.testing_spam)

            while not cluster:
                current = au.select_initial(chosen)
                cluster = au.determine_cluster(current)
                chosen.add(current)
                au.driver.test(au.testing_ham, au.testing_spam)

            cluster_list = list(cluster.cluster_set)

            dicts = au.driver.tester.train_examples[2]

            data = v_correlation(cluster_list, dicts)

            outfile.write("Trial " + str(trial_number) + " Percentage Overlap (Correlation): " + str(data))
            answer = raw_input("Keep going (y/n)? You have performed " + str(trial_number) + " trial(s) so far. ")

            valid_input = False

            while not valid_input:
                if answer == "n":
                    keep_going = False
                    valid_input = True

                elif answer == "y":
                    au.learn(cluster)
                    au.init_ground()
                    trial_number += 1
                    valid_input = True

                else:
                    print "Please enter either y or n."


def p_correlation(polluted, mislabeled):
    """Uses Pearson's Correlation Coefficient to calculate correlation
     between mislabeled results and initial polluted emails in ground truth"""

    n = min(len(polluted), len(mislabeled))

    x = []
    for i in range(0, n):
        x.append(polluted[i].prob)

    y = []
    for i in range(0, n):
        y.append(mislabeled[i].prob)

    return pearsonr(x, y)


def v_correlation(polluted, mislabeled):

    print "Calculating Polluted Data Clustroid..."

    p_minrowsum = sys.maxint
    p_clustroid = None
    p_avgdistance = 0
    i = 1
    for email in polluted:
        print "Calculating on email " + str(i) + " of " + str(len(polluted))
        rowsum = 0
        for email2 in polluted:
            if email == email2:
                continue
            dist = Distance.distance(email, email2, "extreme")
            rowsum += dist ** 2
        if rowsum < p_minrowsum:
            p_minrowsum = rowsum
            p_clustroid = email
            p_avgdistance = sqrt(rowsum / (len(polluted) - 1))
        i += 1

    print "Calculating Mislabeled Data Clustroid..."

    m_minrowsum = sys.maxint
    m_clustroid = None
    m_avgdistance = 0
    i = 1
    for email in mislabeled:
        print "Calculating on email " + str(i) + " of " + str(len(mislabeled))
        rowsum = 0
        for email2 in mislabeled:
            if email == email2:
                continue
            dist = Distance.distance(email, email2, "extreme")
            rowsum += dist ** 2
        if rowsum < m_minrowsum:
            m_minrowsum = rowsum
            m_clustroid = email
            m_avgdistance = sqrt(rowsum / (len(polluted) - 1))
        i += 1

    print "Calculating Overlap..."

    p_size = 0
    i = 1
    for email in polluted:
        distance = Distance.distance(email, m_clustroid, "extreme")
        print "Scanning Polluted Email " + str(i) + " of " + str(len(polluted)) + " with distance " + str(distance)
        if distance < m_avgdistance:
            p_size += 1
        i += 1
    m_size = 0
    i = 1
    for email in mislabeled:
        distance = Distance.distance(email, p_clustroid, "extreme")
        print "Scanning Mislabeled Email " + str(i) + " of " + str(len(mislabeled)) + " with distance " + str(distance)
        if distance < p_avgdistance:
            m_size += 1
        i += 1

    total_size = len(polluted) + len(mislabeled)

    print "Total Size: " + str(total_size)
    print "Size of Polluted Overlap: " + str(p_size)
    print "Size of Mislabeled Overlap: " + str(m_size)
    print "Polluted average distance: " + str(p_avgdistance)
    print "Mislabeled average distance: " + str(m_avgdistance)

    return (float(p_size) + float(m_size)) / float(total_size)


def main():
    import getopt

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hn:s:',
                                   ['ham-keep=', 'spam-keep='])
    except getopt.error, msg:
        usage(1, msg)

    seed = hamkeep = spamkeep = None
    for opt, arg in opts:
        if opt == '-h':
            usage(0)
        elif opt == '-s':
            seed = int(arg)
        elif opt == '--ham-keep':
            hamkeep = int(arg)
        elif opt == '--spam-keep':
            spamkeep = int(arg)

    if args:
        usage(1, "Positional arguments not supported")

    msgs.setparms(hamkeep, spamkeep, seed=seed)
    drive()

if __name__ == "__main__":
    main()
