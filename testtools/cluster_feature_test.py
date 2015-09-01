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
from testtools import data_sets_impact_test as d_test
from testtools import feature_test as f_t


options["TestDriver", "show_histograms"] = False

hams, spams = ds.hams, ds.spams
set_dirs, dir_enumerate = ds.set_dirs, ds.dir_enumerate
stats = d_test.unlearn_stats
seconds_to_english = ds.seconds_to_english
feature_trunc = f_t.feature_trunc
feature_print = f_t.feature_print
feature_lists = f_t.feature_lists
extract_features = f_t.extract_features
au_sig_words = f_t.au_sig_words


def cluster_sig_words(au, cluster):
    c = au.driver.tester.classifier
    features = []
    r_features = []
    words = set(word for msg in cluster.cluster_set for word in msg)

    assert(len(words) > 0)
    for word in words:
        record = c.wordinfo.get(word)
        if record is not None:
            prob = c.probability(record)
            features.append((prob, word))
            r_features.append((word, prob))
    assert(len(features) > 0), c.wordinfo
    features.sort()
    return [features, dict(r_features)]


def cluster_feature_matrix(v_au, p_au, cluster_list):
    machines = [v_au, p_au]
    words = set().union(machine.driver.tester.classifier.wordinfo.keys() for machine in machines)
    au_features = [au_sig_words(machine, words) for machine in machines]
    cluster_features = [cluster_sig_words(v_au, cluster) for cluster in cluster_list]
    sigs = extract_features(au_features + cluster_features, sep_sigs_only=True)
    feature_matrix = feature_lists(sigs, len(cluster_list), label="Cluster")
    return feature_matrix


def cluster_print(cluster, pollution_set3):
    target = str(cluster.target_set3() if pollution_set3 else cluster.target_set4())
    return "(" + str(cluster.size) + ", " + target + ")"


def label_inject(feature_matrix, cluster_list, pollution_set3=True):
    header = feature_matrix[0]
    for i in range(3, len(header)):
        header[i] += " " + cluster_print(cluster_list[i - 3], pollution_set3)


def main():
    sets = [6]
    dest = "C:/Users/bzpru/Desktop/spambayes-1.1a6/unpollute_stats/Yang_Data_Sets (cluster features)/"

    for i in sets:
        ham = hams[i]
        spam = spams[i]
        data_set = set_dirs[i]

        if i > 5:
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

        time_1 = time.time()
        p_au = ActiveUnlearnDriver.ActiveUnlearner([msgs.HamStream(ham_train, [ham_train]),
                                                   msgs.HamStream(ham_p, [ham_p])],        # Training Ham
                                                   [msgs.SpamStream(spam_train, [spam_train]),
                                                   msgs.SpamStream(spam_p, [spam_p])],     # Training Spam
                                                   msgs.HamStream(ham_test, [ham_test]),          # Testing Ham
                                                   msgs.SpamStream(spam_test, [spam_test]),       # Testing Spam
                                                   distance_opt="inv-match", all_opt=True,
                                                   update_opt="hybrid", greedy_opt=False)

        v_au = ActiveUnlearnDriver.ActiveUnlearner([msgs.HamStream(ham_train, [ham_train]),
                                                    []],
                                                   [msgs.SpamStream(spam_train, [spam_train]),
                                                    []],
                                                   msgs.HamStream(ham_test, [ham_test]),
                                                   msgs.SpamStream(spam_test, [spam_test]))

        time_2 = time.time()
        train_time = seconds_to_english(time_2 - time_1)
        print "Train time:", train_time, "\n"

        with open(dest + data_set + " (unlearn_stats).txt", 'w') as outfile:
            cluster_list = stats(p_au, outfile, data_set, [train_ham, train_spam], [test_ham, test_spam],
                                 [ham_polluted, spam_polluted], total_polluted, total_unpolluted, train_time)

        snipped_cluster_list = [cluster[1] for cluster in cluster_list if cluster[0] < 0]
        feature_matrix = cluster_feature_matrix(v_au, p_au, snipped_cluster_list)
        label_inject(feature_matrix, snipped_cluster_list)

        feature_col_width = max(len(row[1]) for row in feature_matrix) + 2
        feature_num_col_width = max(len(row[0]) for row in feature_matrix) + 2

        with open(dest + data_set + " (Separate Features).txt", 'w') as outfile:
            outfile.write("---------------------------\n")
            outfile.write("Data Set: " + data_set + "\n")
            outfile.write("Vanilla Training: " + str(train_ham) + " ham and " + str(train_spam) + " spam.\n")
            outfile.write("Testing: " + str(test_ham) + " ham and " + str(test_spam) + " spam.\n")
            outfile.write("Pollution Training: " + str(ham_polluted) + " ham and " + str(spam_polluted) +
                          " spam.\n")
            outfile.write("---------------------------\n")
            outfile.write("\n\n")
            outfile.write("Unpolluted and Polluted Most Significant Features:\n")
            outfile.write("---------------------------\n")
            for row in feature_matrix:
                justify = [row[0].ljust(feature_num_col_width)]
                for j in range(1, len(row)):
                    justify.append(row[j].strip().ljust(feature_col_width))
                outfile.write("".join(justify) + "\n")


if __name__ == "__main__":
    main()
