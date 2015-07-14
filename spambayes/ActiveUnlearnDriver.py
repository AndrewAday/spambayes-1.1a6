from random import choice
from spambayes import TestDriver
from Distance import distance
import heapq
import sys
import copy


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
        l = set()
        for item in self._queue:
            l.add(item[-1])
        return l


class Cluster:

    def __init__(self, msg, size, active_unlearner, opt="extreme"):
        self.clustroid = msg
        self.size = size
        self.active_unlearner = active_unlearner
        self.ham = set()
        self.spam = set()
        self.opt = opt
        self.cluster_set, self.cluster_heap = self.make_cluster()
        self.divide()

    def make_cluster(self):
        heap = PriorityQueue()
        for i in range(len(self.active_unlearner.driver.tester.train_examples)):
            for train in self.active_unlearner.driver.tester.train_examples[i]:
                if train != self.clustroid:
                    if len(heap) < self.size:
                        heap.push(train, distance(self.clustroid, train, self.opt))
                    else:
                        heap.pushpop(train, distance(self.clustroid, train, self.opt))

        l = heap.taskify()
        assert (len(l) == self.size)
        return l, heap

    def divide(self):
        """Divides messages in the cluster between spam and ham"""
        for msg in self.cluster_set:
            if msg.tag.endswith(".ham.txt"):
                self.ham.add(msg)
            elif msg.tag.endswith(".spam.txt"):
                self.spam.add(msg)

    def target_spam(self):
        """Returns a count of the number of spam emails in the cluster"""
        counter = 0
        for msg in self.cluster_set:
            if msg.tag.endswith(".spam.txt"):
                counter += 1
        return counter

    def target_set3(self):
        """Returns a count of the number of Set3 emails in the cluster"""
        counter = 0
        for msg in self.cluster_set:
            if "Set3" in msg.tag:
                counter += 1
        return counter

    def cluster_more(self, n):
        self.size += n
        k = self.size
        for i in range(len(self.active_unlearner.driver.tester.train_examples)):
            for train in self.active_unlearner.driver.tester.train_examples[i]:
                if train != self.clustroid and train not in self.cluster_set:
                    if len(self.cluster_heap) < k:
                        self.cluster_heap.push(train, distance(self.clustroid, train, self.opt))

                    else:
                        self.cluster_heap.pushpop(train, distance(self.clustroid, train, self.opt))
        assert(len(self.cluster_heap) == k), len(self.cluster_heap)
        self.cluster_set = self.cluster_heap.taskify()
        assert(len(self.cluster_heap) == k), len(self.cluster_heap)
        assert(len(self.cluster_set) == k), len(self.cluster_set)


class ProxyCluster:
    def __init__(self, cluster):
        hams = []
        spams = []

        if len(cluster.ham) + len(cluster.spam) != cluster.size:
            print "Updating cluster ham and spam sets for proxy. . ."
            cluster.divide()

        for ham in cluster.ham:
            hams.append(ham)

        for spam in cluster.spam:
            spams.append(spam)

        self.ham = hams
        self.spam = spams
        self.size = cluster.size


class ActiveUnlearner:

    def __init__(self, training_ham, training_spam, testing_ham, testing_spam):
        self.driver = TestDriver.Driver()
        self.set_driver()
        self.hamspams = zip(training_ham, training_spam)
        self.set_data()
        self.testing_spam = testing_spam
        self.testing_ham = testing_ham
        self.negs = set()
        self.all_negs = False
        self.set_training_nums()
        self.set_dict_nums()
        self.init_ground()

    def set_driver(self):
        self.driver.new_classifier()

    def set_data(self):
        for hamstream, spamstream in self.hamspams:
            self.driver.train(hamstream, spamstream)

    def init_ground(self):
        self.driver.test(self.testing_ham, self.testing_spam)

    def set_training_nums(self):
        hamstream, spamstream = self.hamspams[0]
        self.driver.train_test(hamstream, spamstream)

    def set_dict_nums(self):
        hamstream, spamstream = self.hamspams[1]
        self.driver.dict_test(hamstream, spamstream)

    def unlearn(self, cluster):
        self.driver.untrain(cluster.ham, cluster.spam)

        for ham in cluster.ham:
            self.driver.tester.train_examples[ham.train].remove(ham)
        for spam in cluster.spam:
            self.driver.tester.train_examples[spam.train].remove(spam)

    def learn(self, cluster):
        if len(cluster.ham) + len(cluster.spam) != cluster.size:
            print "Updating cluster ham and spam sets. . ."
            cluster.divide()

        self.driver.train(cluster.ham, cluster.spam)

        for ham in cluster.ham:
            self.driver.tester.train_examples[ham.train].append(ham)
        for spam in cluster.spam:
            self.driver.tester.train_examples[spam.train].append(spam)

    def detect_rate_diff(self, cluster):
        """Returns the theoretical difference in detection rate if a given cluster
           is unlearned. Relearns the cluster afterwards"""
        old_detection_rate = self.driver.tester.correct_classification_rate()
        self.unlearn(cluster)
        self.driver.test(self.testing_ham, self.testing_spam)
        new_detection_rate = self.driver.tester.correct_classification_rate()
        self.learn(cluster)
        return new_detection_rate - old_detection_rate

    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

    # TEST METHODS
    def detect_rate(self, cluster):
        """Returns the detection rate if a given cluster is unlearned.
           Relearns the cluster afterwards"""
        self.unlearn(cluster)
        self.driver.test(self.testing_ham, self.testing_spam)
        detection_rate = self.driver.tester.correct_classification_rate()
        self.learn(cluster)
        return detection_rate

        # TEST METHOD
    def start_detect_rate(self, cluster):
        self.unlearn(cluster)
        self.driver.test(self.testing_ham, self.testing_spam)
        detection_rate = self.driver.tester.correct_classification_rate()
        return detection_rate

    def continue_detect_rate(self, cluster, n):
        old_cluster = copy.deepcopy(cluster.cluster_set)
        cluster.cluster_more(n)
        new_cluster = cluster.cluster_set

        new_unlearns = new_cluster - old_cluster
        assert(len(new_unlearns) == len(new_cluster) - len(old_cluster))
        assert(len(new_unlearns) == n), len(new_unlearns)

        unlearn_hams = []
        unlearn_spams = []

        for unlearn in new_unlearns:
            if unlearn.tag.endswith(".ham.txt"):
                unlearn_hams.append(unlearn)

            elif unlearn.tag.endswith(".spam.txt"):
                unlearn_spams.append(unlearn)

            self.driver.tester.train_examples[unlearn.train].remove(unlearn)

        self.driver.untrain(unlearn_hams, unlearn_spams)
        self.driver.test(self.testing_ham, self.testing_spam)
        detection_rate = self.driver.tester.correct_classification_rate()
        return detection_rate

    # Should only be used right after init_ground
    # May be used for actual method
    def select_initial(self):
        init_point = choice(self.driver.tester.ham_wrong_examples + self.driver.tester.spam_wrong_examples +
                            self.driver.tester.unsure_examples)

        min_distance = sys.maxint
        min_point = None

        for i in len(self.driver.tester.train_examples):
            for email in self.driver.tester.train_examples[i]:
                current_distance = distance(email, init_point, "extreme")
                if current_distance < min_distance:
                    min_point = email
                    min_distance = current_distance

        return min_point, init_point

    # May be used for actual method
    def determine_cluster(self, center, increment):
        print "\nDetermining appropriate cluster around", center.tag, "...\n"
        old_detection_rate = self.driver.tester.correct_classification_rate()
        counter = 0
        cluster = Cluster(center, increment, self)
        self.unlearn(cluster)
        self.driver.test(self.testing_ham, self.testing_spam)
        new_detection_rate = self.driver.tester.correct_classification_rate()

        if new_detection_rate < old_detection_rate:
            print "Center is inviable."
            self.learn(cluster)
            return False

        else:
            proxy_cluster = None
            unlearn_hams = []
            unlearn_spams = []
            new_unlearns = set()

            while new_detection_rate >= old_detection_rate:
                counter += 1
                print "\nExploring cluster of size", (counter + 1) * increment, "...\n"
                old_detection_rate = new_detection_rate
                proxy_cluster = ProxyCluster(cluster)
                old_cluster_set = copy.deepcopy(cluster.cluster_set)
                cluster.cluster_more(increment)
                new_cluster_set = cluster.cluster_set

                new_unlearns = new_cluster_set - old_cluster_set
                assert(len(new_unlearns) == len(new_cluster_set) - len(old_cluster_set))
                assert(len(new_unlearns) == increment), len(new_unlearns)

                unlearn_hams = []
                unlearn_spams = []

                for unlearn in new_unlearns:
                    if unlearn.tag.endswith(".ham.txt"):
                        unlearn_hams.append(unlearn)

                    elif unlearn.tag.endswith(".spam.txt"):
                        unlearn_spams.append(unlearn)

                    self.driver.tester.train_examples[unlearn.train].remove(unlearn)

                self.driver.untrain(unlearn_hams, unlearn_spams)
                self.driver.test(self.testing_ham, self.testing_spam)
                new_detection_rate = self.driver.tester.correct_classification_rate()

            for unlearn in new_unlearns:
                self.driver.tester.train_examples[unlearn.train].append(unlearn)
            self.driver.train(unlearn_hams, unlearn_spams)
            print "Appropriate cluster found."
            return proxy_cluster, counter

    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

"""
    # -----------------------------------------------------------------------------------
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
                # still no definitive change
                # look at neighbors in cluster
                counter = 0
                for neighbor in cluster:
                    if unlearn_compare(neighbor):


                raise AssertionError

        if unlearn_compare(message):
            self.au.add_pollutant(message)
            # untrain msg here?
        else:
            self.au.add_benign(message)

        next_msg = self.get_next()
        c = self.cluster(next_msg, k)
        t = unlearn_compare(c)

    # -----------------------------------------------------------------------------------

    # Returns the set of mislabeled emails (from the ground truth) based off of the
    # current classifier state. By default assumes the current state's numbers and
    # tester false positives/negatives have already been generated; if not, it'll run the
    # predict method from the tester.
    def mislabeled(self, update=False):
        tester = self.driver.tester
        if update:
            tester.predict(self.testing_ham, False)
            tester.predict(self.testing_spam, True)

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
"""