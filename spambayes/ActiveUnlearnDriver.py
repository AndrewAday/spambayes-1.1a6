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

        if self.clustroid.tag.endswith(".ham.txt"):
            self.ham.add(self.clustroid)

        elif self.clustroid.tag.endswith(".spam.txt"):
            self.spam.add(self.clustroid)

    def target_spam(self):
        """Returns a count of the number of spam emails in the cluster"""
        counter = 0
        for msg in self.cluster_set:
            if msg.tag.endswith(".spam.txt"):
                counter += 1

        if self.clustroid.tag.endswith(".spam.txt"):
            counter += 1

        return counter

    def target_set3(self):
        """Returns a count of the number of Set3 emails in the cluster"""
        counter = 0
        for msg in self.cluster_set:
            if "Set3" in msg.tag:
                counter += 1

        if "Set3" in self.clustroid.tag:
            counter += 1

        return counter

    def cluster_more(self, n):
        old_cluster_set = copy.deepcopy(self.cluster_set)
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

        new_elements = self.cluster_set - old_cluster_set
        assert(len(new_elements) == n), len(new_elements)

        for msg in new_elements:
            if msg.tag.endswith(".ham.txt"):
                self.ham.add(msg)
            elif msg.tag.endswith(".spam.txt"):
                self.spam.add(msg)

        return new_elements


class ProxyCluster:
    def __init__(self, cluster):
        hams = []
        spams = []
        cluster_set = set()

        if len(cluster.ham) + len(cluster.spam) != cluster.size + 1:
            print "\nUpdating cluster ham and spam sets for proxy...\n"
            cluster.divide()

        else:
            print "\nProxy cluster ham/spam sets do not need updating; continuing.\n"

        for ham in cluster.ham:
            hams.append(ham)
            cluster_set.add(ham)

        for spam in cluster.spam:
            spams.append(spam)
            cluster_set.add(spam)

        self.ham = hams
        self.spam = spams
        self.size = cluster.size
        self.cluster_set = cluster_set

    def target_set3(self):
        counter = 0

        for msg in self.cluster_set:
            if "Set3" in msg.tag:
                counter += 1
        return counter


class ActiveUnlearner:

    def __init__(self, training_ham, training_spam, testing_ham, testing_spam, threshold=90, increment=100,):
        self.increment = increment
        self.threshold = threshold
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
        if len(cluster.ham) + len(cluster.spam) != cluster.size + 1:
            print "\nUpdating cluster ham and spam sets...\n"
            cluster.divide()

        self.driver.train(cluster.ham, cluster.spam)

        for ham in cluster.ham:
            self.driver.tester.train_examples[ham.train].append(ham)
        for spam in cluster.spam:
            self.driver.tester.train_examples[spam.train].append(spam)

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

    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

    def determine_cluster(self, center):
        """ Given a chosen starting center and a given increment of cluster size, it continues to grow and cluster more
            until the detection rate hits a maximum peak (i.e. optimal cluster); if first try is a decrease, reject this
            center and return False."""

        print "\nDetermining appropriate cluster around", center.tag, "...\n"
        old_detection_rate = self.driver.tester.correct_classification_rate()
        counter = 0
        cluster = Cluster(center, self.increment, self)
        self.unlearn(cluster)
        self.driver.test(self.testing_ham, self.testing_spam)
        new_detection_rate = self.driver.tester.correct_classification_rate()

        if new_detection_rate <= old_detection_rate:
            print "\nCenter is inviable.\n"
            self.learn(cluster)
            return False

        else:
            proxy_cluster = None
            unlearn_hams = []
            unlearn_spams = []
            new_unlearns = set()

            while new_detection_rate > old_detection_rate:
                counter += 1
                print "\nExploring cluster of size", (counter + 1) * self.increment, "...\n"

                old_detection_rate = new_detection_rate
                proxy_cluster = ProxyCluster(cluster)
                """
                old_cluster_set = copy.deepcopy(cluster.cluster_set)
                """
                new_unlearns = cluster.cluster_more(self.increment)
                """
                new_cluster_set = cluster.cluster_set

                new_unlearns = new_cluster_set - old_cluster_set

                assert(len(new_unlearns) == len(new_cluster_set) - len(old_cluster_set))
                """
                assert(len(new_unlearns) == self.increment), len(new_unlearns)

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

            # This part is done because we've clustered just past the peak point, so we need to go back
            # one increment and relearn the extra stuff.
            for unlearn in new_unlearns:
                self.driver.tester.train_examples[unlearn.train].append(unlearn)
            self.driver.train(unlearn_hams, unlearn_spams)

            assert(proxy_cluster.size == self.increment * counter), counter

            print "\nAppropriate cluster found, with size " + str(proxy_cluster.size) + ".\n"
            return proxy_cluster

    # -----------------------------------------------------------------------------------
    def active_unlearn(self, outfile, test=False):

        # Select initial message to unlearn (based on mislabeled emails)
        # Unlearn email
        # Compare detection rates before and after unlearning
            # If detection rate improves, remove email
            # If detection rate worsenes, keep email
        # Select next email, based on previous email
        # Recursively (?) active-unlearn the next email

        cluster_list = []
        chosen = set()
        cluster_count = 0
        detection_rate = self.driver.tester.correct_classification_rate()

        if detection_rate < self.threshold:
            current = self.select_initial()
            cluster = self.determine_cluster(current)
            chosen.add(current)
            self.driver.test(self.testing_ham, self.testing_spam)

            # Keep trying new clusters based off of points from mislabeled until we get a viable cluster
            while not cluster:
                current = self.select_initial(chosen)
                cluster = self.determine_cluster(current)
                chosen.add(current)
                self.driver.test(self.testing_ham, self.testing_spam)

            cluster_list.append(cluster)
            cluster_count += 1
            print "\nUnlearned", cluster_count, "cluster(s) so far.\n"

            detection_rate = self.driver.tester.correct_classification_rate()
            print "\nCurrent detection rate achieved is " + str(detection_rate) + ".\n"
            if outfile is not None:
                outfile.write(str(cluster_count) + ": " + str(detection_rate) + ", " + str(cluster.size + 1) + ", " +
                              str(cluster.target_set3()) + "\n")

            while detection_rate < self.threshold:
                current = self.select_initial(chosen)
                cluster = self.determine_cluster(current)
                chosen.add(current)
                self.driver.test(self.testing_ham, self.testing_spam)

                while not cluster:
                    current = self.select_initial(chosen)
                    cluster = self.determine_cluster(current)
                    chosen.add(current)
                    self.driver.test(self.testing_ham, self.testing_spam)

                cluster_list.append(cluster)
                cluster_count += 1
                print "\nUnlearned", cluster_count, "cluster(s) so far.\n"

                detection_rate = self.driver.tester.correct_classification_rate()
                print "\nCurrent detection rate achieved is " + str(detection_rate) + ".\n"
                if outfile is not None:
                    outfile.write(str(cluster_count) + ": " + str(detection_rate) + ", " + str(cluster.size + 1) + ", "
                                  + str(cluster.target_set3()) + "\n")
        if test:
            return cluster_list

        print "\nThreshold achieved after", cluster_count, "clusters unlearned.\n"
    # -----------------------------------------------------------------------------------

    def mislabeled(self, update=False):
        """ Returns the set of mislabeled emails (from the ground truth) based off of the
            current classifier state. By default assumes the current state's numbers and
            tester false positives/negatives have already been generated; if not, it'll run the
            predict method from the tester."""
        tester = self.driver.tester
        if update:
            tester.predict(self.testing_ham, False)
            tester.predict(self.testing_spam, True)

        mislabeled = set()
        for wrong_ham in tester.ham_wrong_examples:
            mislabeled.add(wrong_ham)

        for wrong_spam in tester.spam_wrong_examples:
            mislabeled.add(wrong_spam)

        for unsure in tester.unsure_examples:
            mislabeled.add(unsure)

        return mislabeled

    def select_initial(self, chosen=set(), row_sum=False):
        """ Returns an email to be used as the initial unlearning email based on
            the mislabeled data (our tests show that the mislabeled and pollutant
            emails are strongly, ~80%, correlated) if option is true (which is default)."""
        mislabeled = self.mislabeled()
        print "Chosen:", chosen
        print "Total Chosen: ", len(chosen)
        if row_sum:

            # We want to minimize the distances (rowsum) between the email we select
            # and the mislabeled emails. This ensures that the initial email we select
            # is correlated with the mislabeled emails.

            minrowsum = sys.maxint
            init_email = None
            for i in range(len(self.driver.tester.train_examples)):
                for email in self.driver.tester.train_examples[i]:
                    rowsum = 0
                    for email2 in mislabeled:
                        dist = distance(email, email2, "extreme")
                        rowsum += dist ** 2
                    if rowsum < minrowsum:
                        minrowsum = rowsum
                        init_email = email

            return init_email

        else:

            # This chooses an arbitrary point from the mislabeled emails and simply finds the email
            # in training that is closest to this point.
            try:
                mislabeled_point = choice(list(mislabeled - chosen))

            except:
                raise AssertionError(str(mislabeled))

            min_distance = sys.maxint
            init_email = None

            for i in range(len(self.driver.tester.train_examples)):
                for email in self.driver.tester.train_examples[i]:
                    current_distance = distance(email, mislabeled_point, "extreme")
                    if current_distance < min_distance:
                        init_email = email
                        min_distance = current_distance

            return init_email

    """
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