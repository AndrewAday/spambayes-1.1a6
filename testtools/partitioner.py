from random import shuffle
"""
Splits testing data into T1 and T2 to cross-validate the accuracy of active unlearning
"""


def partition(ham_count, ham_stream, spam_count, spam_stream, option):
    if option == 'random':
        return random(ham_count, spam_count)
        # return range(ham_count), range(spam_count), [], []

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