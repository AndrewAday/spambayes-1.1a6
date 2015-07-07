from random import choice
from spambayes import TestDriver
from Distance import distance
import heapq
import sys
import os


def d(negs, x, opt=None):
    s = 0
    for neg in negs:
        s += distance(neg, x, opt)
    return s


class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def __len__(self):
        return len(self._queue)

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index -= 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def pushpop(self, item, priority):
        item = heapq.heappushpop(self._queue, (-priority, self._index, item))[-1]
        self._index -= 1
        return item

    def taskify(self):
        l = []
        for item in self._queue:
            l.append(item[-1])
        return l


class ActiveUnlearnDriver:

    # Takes training ham/spam directories, and set for ground truth
    def __init__(self, ham, spam, real_ham, real_spam, opt=None):
        self.set_driver()
        self.hamspams = zip(ham, spam)
        self.set_data()
        self.real_spam = real_spam
        self.real_ham = real_ham
        self.negs = set()
        self.all_negs = False
        self.opt = opt
        self.set_training_nums()
        self.set_dict_nums()
        self.init_ground()
        
    def init_ground(self):
        self.driver.test(self.real_ham, self.real_spam)

    def set_training_nums(self):
        hamstream, spamstream = self.hamspams[0]
        self.driver.train_test(hamstream, spamstream)

    def set_dict_nums(self):
        hamstream, spamstream = self.hamspams[1]
        self.driver.dict_test(hamstream, spamstream)

    def set_driver(self):
        self.driver = TestDriver.Driver()
        self.driver.new_classifier()

    def set_data(self):
        for hamstream, spamstream in self.hamspams:
            self.driver.train(hamstream, spamstream)

    def cluster(self, msg, k):
        heap = PriorityQueue()
        for i in range(4):
            for train in self.driver.tester.train_examples[i]:
                if train != msg:
                    if len(heap) < k:
                        heap.push(train, distance(msg, train, self.opt))

                    else:
                        heap.pushpop(train, distance(msg, train, self.opt))

        l = heap.taskify()
        assert (len(l) == k)
        return l

    def unlearn(self, cluster):
        for msg in cluster:
            self.driver.tester.classifier.unlearn(msg, True)
            self.driver.tester.train_examples[msg.train].remove(msg)

    def learn(self, cluster):
        for msg in cluster:
            self.driver.tester.classifier.learn(msg, True)
            self.driver.tester.train_examples[msg.train].append(msg)

    # TO BE FINISHED/FIXED
    def get_next(self):
        if self.all_negs:
            max_s = 0
            max_v = None
            for e in (self.driver.tester.train_examples[0] - self.negs):
                if d(self.negs, e, self.opt) > max_s:
                    max_s = d(self.negs, e, self.opt)
                    max_v = e
            return max_v

        else:
            pass

    # Returns the theroetical difference in detection rate if a given cluster is to be unlearned;
    # relearns the cluster afterwards
    def detect_rate_diff(self, cluster):
        old_detection_rate = self.driver.tester.correct_classification_rate()
        self.unlearn(cluster)
        self.driver.test(self.real_ham, self.real_spam)
        new_detection_rate = self.driver.tester.correct_classification_rate()
        self.learn(cluster)
        return new_detection_rate - old_detection_rate

    # TEST METHOD
    def detect_rate(self, cluster):
        self.unlearn(cluster)
        self.driver.test(self.real_ham, self.real_spam)
        detection_rate = self.driver.tester.correct_classification_rate()
        self.learn(cluster)
        return detection_rate

    # TEST METHOD
    def cluster_spam(self, cluster):
        counter = 0
        for msg in cluster:
            if msg.tag.endswith(".spam.txt"):
                counter += 1
        return counter

    def cluster_set3(self, cluster):
        counter = 0
        for msg in cluster:
            if "Set3" in msg.tag:
                counter += 1
        return counter

    # TO BE FIXED
    def active_unlearn(self, k):

        # Select initial message to unlearn (based on mislabeled emails, or randomly)
        # Unlearn email
        # Compare detection rates before and after unlearning
            # If detection rate improves, remove email
            # If detection rate worsenes, keep email
        # Select next email, based on previous email
        # Recursively (?) active-unlearn the next email

        message = self.selectinitial()
        driver = self.driver

        cluster = self.cluster(message, k)

        if self.detect_rate_diff(cluster) < 0: # Message is a pollutant, detection rate improved
            self.unlearn(cluster)
            next_message = self.nextemail(message, True)

        else:
            next_message = self.nextemail(message, False)

        # Compares stats between pre-unlearn and post-unlearn, returns if stream should be removed
        # The set of msgs is supposed to be all classified under one label -- we would potentially be
        # unlearning clusters of emails at a time.
        def unlearn_compare(cluster):
            # return true if pollutant
            # return false if benign

            # SHOULD BE REAL_HAM AND REAL_SPAM
            for hamstream, spamstream in self.hamspams:
                driver.test(hamstream, spamstream)

            diff = self.detect_rate_diff(cluster)

            if diff != 0:
                # Change in detection rate - most definitive
                return diff > 0

            # This is temporary, just to check if this case is actually ever encountered; we'll provide
            # a fix to this later.
            else:

                """
                # no definitive change in detection rate
                # look at impact on spam/ham scores instead
                i_h = 0
                i_s = 0
                threshold = 0
                for ham in self.au.ham:
                    i_h += ham.probdiff
                i_h /= len(self.au.ham)

                for spam in self.au.spam:
                    i_s += spam.probdiff
                i_s /= len(self.au.spam)
                return i_h < -threshold and i_s > threshold
                """

                """
                # still no definitive change
                # look at neighbors in cluster
                counter = 0
                for neighbor in cluster:
                    if unlearn_compare(neighbor):
                """

                raise AssertionError

        if unlearn_compare(message):
            self.au.add_pollutant(message)
            # untrain msg here?
        else:
            self.au.add_benign(message)

        next_msg = self.get_next()
        c = self.cluster(next_msg, k)
        t = unlearn_compare(c)

    # Returns the set of mislabeled emails (from the ground truth) based off of the
    # current classifier state. By default assumes the current state's numbers and
    # tester false positives/negatives have already been generated; if not, it'll run the
    # predict method from the tester.
    def mislabeled(self, update=False):
        tester = self.driver.tester
        if update:
            tester.predict(self.real_ham, False)
            tester.predict(self.real_spam, True)

        mislabeled = set()
        mislabeled += tester.spam_wrong_examples, tester.ham_wrong_examples
        return mislabeled

    # Returns an email to be used as the initial unlearning email based on
    # the mislabeled data (our tests show that the mislabeled and pollutant
    # emails are strongly, ~80%, correlated) if option is true (which is default).
    def selectinitial(self, mislabel=True):
        if mislabel:
            mislabeled = self.mislabeled()
            training = self.hamspams

            # We want to minimize the distances (rowsum) between the email we select
            # and the mislabeled emails. This ensures that the initial email we select
            # is correlated with the mislabeled emails.

            minrowsum = sys.maxint
            init_email = None
            for email in training:
                rowsum = 0
                for email2 in mislabeled:
                    dist = distance(email, email2, "extreme")
                    rowsum += dist ** 2
                if rowsum < minrowsum:
                    minrowsum = rowsum
                    init_email = email

            return init_email

        else:
            pass

    def nextemail(self, msg, is_pollutant):
        training = self.hamspams
        next_email = None
        if is_pollutant:
            min_distance = sys.maxint
            for email in training:
                dist = distance(email, msg, "extreme")
                if dist < min_distance:
                    min_distance = dist
                    next_email = email
        else:
            max_distance = 0

            for email in training:
                dist = distance(email, msg, "extreme")
                if dist > max_distance:
                    max_distance = dist
                    next_email = email

        return next_email
