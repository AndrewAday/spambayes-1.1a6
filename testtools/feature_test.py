__author__ = 'Alex'

import os
import sys
import numpy as np

sys.path.insert(-1, os.getcwd())
sys.path.insert(-1, os.path.dirname(os.getcwd()))

from spambayes import ActiveUnlearnDriver
from spambayes.Options import options
from spambayes import msgs
from testtools import data_sets as ds

options["TestDriver", "show_histograms"] = False

hams, spams = ds.hams, ds.spams
set_dirs, dir_enumerate = ds.set_dirs, ds.dir_enumerate


def au_sig_words(au, words):
    c = au.driver.tester.classifier
    features = []
    r_features = []
    assert(len(words) > 0)
    for word in words:
        record = c.wordinfo.get(word)
        if record is not None:
            prob = c.probability(record)
            features.append((prob, word))
            r_features.append((word, prob))
    assert(len(features) > 0), c.wordinfo
    features.sort()
    return features, dict(r_features)


def feature_combine(n, p_features, p_dict, v_features, v_dict):
    assert(len(p_features) > 0)
    p_most_sig = p_features[:n] + p_features[-n:]
    assert(len(p_most_sig) == 2 * n), len(p_most_sig)
    v_most_sig = v_features[:n] + v_features[-n:]
    p_sig_words = set(feature[1] for feature in p_most_sig)
    v_sig_words = set(feature[1] for feature in v_most_sig)
    all_sig_words = p_sig_words.union(v_sig_words)
    features = []

    for word in all_sig_words:
        v_prob = 0
        try:
            v_prob = v_dict[word]

        except KeyError:
            pass

        p_prob = 0
        try:
            p_prob = p_dict[word]

        except KeyError:
            pass

        features.append([word, v_prob, p_prob])

    data = np.array(features)
    features = data[np.argsort(data[:,1])]
    return features, p_most_sig, v_most_sig


def main():
    sets = [0]

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

        p_au = ActiveUnlearnDriver.ActiveUnlearner([msgs.HamStream(ham_train, [ham_train]),
                                                   msgs.HamStream(ham_p, [ham_p])],        # Training Ham
                                                   [msgs.SpamStream(spam_train, [spam_train]),
                                                   msgs.SpamStream(spam_p, [spam_p])],     # Training Spam
                                                   msgs.HamStream(ham_test, [ham_test]),          # Testing Ham
                                                   msgs.SpamStream(spam_test, [spam_test]),       # Testing Spam
                                                   )

        v_au = ActiveUnlearnDriver.ActiveUnlearner([msgs.HamStream(ham_train, [ham_train]),
                                                    []],
                                                   [msgs.SpamStream(spam_train, [spam_train]),
                                                    []],
                                                   msgs.HamStream(ham_test, [ham_test]),
                                                   msgs.SpamStream(spam_test, [spam_test]))

        p_c = p_au.driver.tester.classifier
        v_c = p_au.driver.tester.classifier
        words = set(p_c.wordinfo.keys()).union(v_c.wordinfo.keys())

        p_features, p_dict = au_sig_words(p_au, words)
        v_features, v_dict = au_sig_words(v_au, words)
        features, p_most_sig, v_most_sig = feature_combine(75, p_features, p_dict, v_features, v_dict)
        assert(len(p_most_sig) == 150)
        feature_matrix = [["", "Unpolluted", "Polluted"]] + [[str(i + 1), str(v_most_sig[i][0]) + ": " +
                                                              str(v_most_sig[i][1]), str(p_most_sig[i][0]) + ": " +
                                                              str(p_most_sig[i][1])] for i in range(150)]

        combined_matrix = [["", "Unpolluted", "Polluted"]] + [[feature[0], str(feature[1]), str(feature[2])]
                                                              for feature in features]

        feature_col_width = max(len(item) for row in feature_matrix for item in row) + 2
        combined_col_width = max(len(item) for row in combined_matrix for item in row) + 2

        with open("C:\\Users\\Alex\\Desktop\\unpollute_stats\\Yang_Data_Sets (feature compare)\\" + data_set +
                  ".txt", 'w') as outfile:
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
                outfile.write("".join([row[0].ljust(5), row[1].ljust(feature_col_width), row[2].ljust(feature_col_width)
                                       ]) + "\n")

            outfile.write("\n\n")
            outfile.write("Feature Comparison:\n")
            outfile.write("---------------------------\n")

            for row in combined_matrix:
                outfile.write("".join(word.ljust(combined_col_width) for word in row) + "\n")

if __name__ == "__main__":
    main()
