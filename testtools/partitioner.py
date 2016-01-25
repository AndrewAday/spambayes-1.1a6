import os
import sys
from random import shuffle


"""
Splits testing data into T1 and T2 to cross-validate the accuracy of active unlearning
"""

def partition(ham_count, ham_dir, spam_count, spam_dir, option, features, au=None):
    if option == 'random':
        return random(ham_count, spam_count)
        # return range(ham_count), range(spam_count), [], []
    elif option == 'features':
        return feature_parse(ham_dir, spam_dir, features)
    elif option == 'mislabeled':
        return mislabeled_parse(ham_count, ham_dir, spam_count, spam_dir, au)
    else:
        raise ValueError

def random(ham_count, spam_count):
    ham_indices = range(ham_count)
    spam_indices = range(spam_count)
    shuffle(ham_indices)
    shuffle(spam_indices)
    t1_ham = ham_indices[:len(ham_indices)/2]
    t2_ham = ham_indices[len(ham_indices)/2:]
    t1_spam = spam_indices[:len(spam_indices)/2]
    t2_spam = spam_indices[len(spam_indices)/2:]
    return t1_ham, t1_spam, t2_ham, t2_spam

def feature_parse(ham_dir, spam_dir, features):  # he him his she her them we
    t1_ham = []
    t1_spam = []
    t2_ham = []
    t2_spam = []

    ham_emails = os.listdir(ham_dir)
    spam_emails = os.listdir(spam_dir)
    
    for i,ham in enumerate(ham_emails):
        print 'processing ' + str(i) + '/' + str(len(ham_emails)) + ' ham emails'
        sys.stdout.write("\033[F")
        with open(ham_dir+'/'+ham) as f:
            added = False
            for line in f:
                if any(feature in line.split(' ') for feature in features):
                    t2_ham.append(i)
                    # print line
                    added = True
                    break
            if not added:
                t1_ham.append(i) 

    for i,spam in enumerate(spam_emails):
        print 'processing ' + str(i) + '/' + str(len(spam_emails)) + ' spam emails'
        sys.stdout.write("\033[F")
        with open(spam_dir+'/'+spam) as f:
            added = False
            for line in f:
                if any(feature in line.split(' ') for feature in features):
                    t2_spam.append(i)
                    # print line
                    added = True
                    break
            if not added:
                t1_spam.append(i) 

    return t1_ham, t1_spam, t2_ham, t2_spam

def mislabeled_parse(ham_count, ham_dir, spam_count, spam_dir, au):
    tester = au.driver.tester
    wrong_ham = tester.ham_wrong_examples  # ham called spam
    wrong_spam = tester.spam_wrong_examples  # spam called ham

    wrong_ham = [email.index for email in wrong_ham]
    wrong_spam = [email.index for email in wrong_spam]

    t1_ham = wrong_ham
    t1_spam = wrong_spam
    t2_ham = list(set(range(ham_count)) - set(wrong_ham))
    t2_spam = list(set(range(spam_count)) - set(wrong_spam))

    return t1_ham, t1_spam, t2_ham, t2_spam


    




