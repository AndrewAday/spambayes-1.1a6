from random import choice, shuffle
from spambayes import TestDriver, quickselect
from Distance import distance
from itertools import chain
import heapq
import sys
import copy
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
        l = set()
        for item in self._queue:
            l.add(item[-1])
        return l


class Cluster:

    def __init__(self, msg, size, active_unlearner, opt="extreme", working_set=None, sort_first=True):
        self.clustroid = msg
        self.size = size
        self.active_unlearner = active_unlearner
        self.sort_first = sort_first
        self.working_set = working_set
        self.ham = set()
        self.spam = set()
        self.opt = opt
        self.dist_list = self.distance_array()
        """
        self.cluster_set, self.cluster_heap = self.make_cluster()
        """
        self.cluster_set = self.make_cluster()
        self.divide()

    def distance_array(self):
        dist_list = []
        train_examples = self.active_unlearner.driver.tester.train_examples
        if self.working_set is not None:
            """
            for i in range(len(self.active_unlearner.driver.tester.train_examples)):
                for train in self.active_unlearner.driver.tester.train_examples[i]:
                    if train != self.clustroid:
                        dist_list.append((distance(self.clustroid, train, self.opt), train))
            """
            dist_list = [(distance(self.clustroid, train, self.opt), train) for train in chain(train_examples[0], train_examples[1], train_examples[2],
                train_examples[3]) if train != self.clustroid]

        else:
            dist_list = [(distance(self.clustroid, train, self.opt), train) for train in self.working_set if train != self.clustroid]

        if sort_first:
            return dict_list.sort()

        else:
            return dict_list

    def make_cluster(self):
        """
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
        """
        if self.sort_first:
            return set(item[1] for item in self.dist_list[:self.size])

        else:
            k_smallest = quickselect.k_smallest
            return set(item[1] for item in k_smallest(self.dist_list, self.size))

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

    def target_set4(self):
        """Returns a count of the number of Set4 emails in the cluster"""
        counter = 0
        for msg in self.cluster_set:
            if "Set4" in msg.tag:
                counter += 1

        if "Set4" in self.clustroid.tag:
            counter += 1

        return counter

    def cluster_more(self, n):
        old_cluster_set = self.cluster_set
        self.size += n
        """
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
        """
        if sort_first:
            new_cluster_set = set(item[1] for item in self.dict_list[:self.size])
        else:
            k_smallest = quickselect.k_smallest
            new_cluster_set = set(item[1] for item in k_smallest(self.dist_list, self.size))

        new_elements = list(item for item in new_cluster_set if item not in old_cluster_set)
        self.cluster_set = new_cluster_set

        assert(len(self.cluster_set) == self.size), len(self.cluster_set)
        assert(len(new_elements) == n), len(new_elements)

        for msg in new_elements:
            if msg.tag.endswith(".ham.txt"):
                self.ham.add(msg)
            elif msg.tag.endswith(".spam.txt"):
                self.spam.add(msg)

        return new_elements


class ProxyCluster:
    def __init__(self, cluster):

        if len(cluster.ham) + len(cluster.spam) != cluster.size + 1:
            print "\nUpdating cluster ham and spam sets for proxy...\n"
            cluster.divide()

        else:
            print "\nProxy cluster ham/spam sets do not need updating; continuing.\n"

        hams = [ham for ham in cluster.ham]
        spams = [spam for spam in cluster.spam]
        cluster_set = set(msg for msg in chain(cluster.ham, cluster.spam))

        assert(len(cluster_set) == len(cluster.ham) + len(cluster.spam))

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

    def target_set4(self):
        counter = 0

        for msg in self.cluster_set:
            if "Set4" in msg.tag:
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
        self.init_ground(True)
        self.mislabeled_chosen = set()
        self.current_detection_rate = self.driver.tester.correct_classification_rate()
        print "Initial detection rate:", self.current_detection_rate

    def set_driver(self):
        self.driver.new_classifier()

    def set_data(self):
        for hamstream, spamstream in self.hamspams:
            self.driver.train(hamstream, spamstream)

    def init_ground(self, first_test=False):
        if first_test:
            self.driver.test(self.testing_ham, self.testing_spam, first_test)

        else:
            self.driver.test(self.driver.tester.truth_examples[1], self.driver.tester.truth_examples[0], first_test)

    def set_training_nums(self):
        hamstream, spamstream = self.hamspams[0]
        self.driver.train_test(hamstream, spamstream)

    def set_dict_nums(self):
        hamstream, spamstream = self.hamspams[1]
        self.driver.dict_test(hamstream, spamstream)

    def unlearn(self, cluster):
        if len(cluster.ham) + len(cluster.spam) != cluster.size + 1:
            print "\nUpdating cluster ham and spam sets...\n"
            cluster.divide()

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

    def detect_rate(self, cluster):
        """Returns the detection rate if a given cluster is unlearned.
           Relearns the cluster afterwards"""
        self.unlearn(cluster)
        self.init_ground()
        detection_rate = self.driver.tester.correct_classification_rate()
        self.learn(cluster)
        return detection_rate

    def start_detect_rate(self, cluster):
        self.unlearn(cluster)
        self.init_ground()
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
        self.init_ground()
        detection_rate = self.driver.tester.correct_classification_rate()
        return detection_rate

    # --------------------------------------------------------------------------------------------------------------

    def determine_cluster(self, center, working_set=None, binary=False):
        """ Given a chosen starting center and a given increment of cluster size, it continues to grow and cluster more
            until the detection rate hits a maximum peak (i.e. optimal cluster); if first try is a decrease, reject this
            center and return False."""

        print "\nDetermining appropriate cluster around", center.tag, "...\n"
        old_detection_rate = self.current_detection_rate
        counter = 0
        cluster = Cluster(center, self.increment, self, working_set=working_set)

        # Test detection rate after unlearning cluster
        self.unlearn(cluster)
        self.init_ground()
        new_detection_rate = self.driver.tester.correct_classification_rate()

        if new_detection_rate <= old_detection_rate:    # Detection rate worsens - Reject
            print "\nCenter is inviable.\n"
            proxy_cluster = ProxyCluster(cluster)
            self.learn(cluster)
            """
            print "\nResetting numbers...\n"
            self.init_ground()
            """

            return False, proxy_cluster

        else:                                           # Detection rate improves - Grow cluster
            proxy_cluster = None
            unlearn_hams = []
            unlearn_spams = []
            new_unlearns = set()

            while new_detection_rate > old_detection_rate:
                counter += 1
                print "\nExploring cluster of size", (counter + 1) * self.increment, "...\n"

                old_detection_rate = new_detection_rate
                proxy_cluster = ProxyCluster(cluster)
                new_unlearns = cluster.cluster_more(self.increment)

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
                self.init_ground()
                new_detection_rate = self.driver.tester.correct_classification_rate()

            # This part is done because we've clustered just past the peak point, so we need to go back
            # one increment and relearn the extra stuff.
            for unlearn in new_unlearns:
                self.driver.tester.train_examples[unlearn.train].append(unlearn)
            self.driver.train(unlearn_hams, unlearn_spams)

            assert(proxy_cluster.size == self.increment * counter), counter

            print "\nAppropriate cluster found, with size " + str(proxy_cluster.size) + ".\n"
            """
            print "\nResetting numbers...\n"
            self.init_ground()
            """
            self.current_detection_rate = old_detection_rate
            return True, proxy_cluster

    # -----------------------------------------------------------------------------------
    def active_unlearn(self, outfile, test=False, pollution_set3=True):

        cluster_list = []
        chosen = self.mislabeled_chosen
        cluster_count = 0
        attempt_count = 0
        detection_rate = self.current_detection_rate

        while detection_rate < self.threshold:
            current = self.select_initial(chosen)
            attempt_count += 1
            cluster = self.determine_cluster(current)
            print "\nAttempted", attempt_count, "attempts so far.\n"

            while not cluster[0]:
                current = self.select_initial(chosen)
                attempt_count += 1
                cluster = self.determine_cluster(current)
                print "\nAttempted", attempt_count, "attempts so far.\n"

            cluster_list.append(cluster[1])
            cluster_count += 1
            print "\nUnlearned", cluster_count, "cluster(s) so far.\n"

            detection_rate = self.current_detection_rate
            print "\nCurrent detection rate achieved is " + str(detection_rate) + ".\n"
            if outfile is not None:
                if pollution_set3:
                    outfile.write(str(cluster_count) + ", " + str(attempt_count) + ": " + str(detection_rate) + ", " +
                                  str(cluster[1].size + 1) + ", " + str(cluster[1].target_set3()) + "\n")

                else:
                    outfile.write(str(cluster_count) + ", " + str(attempt_count) + ": " + str(detection_rate) + ", " +
                                  str(cluster[1].size + 1) + ", " + str(cluster[1].target_set4()) + "\n")
                outfile.flush()
                os.fsync(outfile)

        if test:
            return cluster_list

        print "\nThreshold achieved after", cluster_count, "clusters unlearned and", attempt_count, "attempts.\n"
    # -----------------------------------------------------------------------------------

    def brute_force_active_unlearn(self, outfile, test=False, center_iteration=True, pollution_set3=True):
        cluster_list = []
        cluster_count = 0
        rejection_count = 0
        rejections = set()
        training = self.shuffle_training()
        original_training_size = len(training)
        detection_rate = self.current_detection_rate
        print "\nCurrent detection rate achieved is " + str(detection_rate) + ".\n"

        while len(training) > 0:
            print "\nStarting new round of untraining;", len(training), "out of", original_training_size, "training left" \
                                                                                                          ".\n"

            current = training[len(training) - 1]
            cluster = self.determine_cluster(current, working_set=training)

            if not cluster[0]:
                print "\nMoving on from inviable cluster center...\n"
                if center_iteration:
                    training.remove(current)
                    rejections.add(current)
                    rejection_count += 1

                else:
                    for msg in cluster[1].cluster_set:
                        if msg not in rejections:
                            training.remove(msg)
                            rejections.add(msg)

                    if current not in rejections:
                        training.remove(current)
                        rejections.add(current)
                    rejection_count += 1

                print "\nRejected", rejection_count, "attempt(s) so far.\n"

            else:
                cluster_list.append(cluster[1])
                print "\nRemoving cluster from shuffled training set...\n"

                for msg in cluster[1].cluster_set:
                    if msg not in rejections:
                        training.remove(msg)
                        rejections.add(msg)

                if current not in rejections:
                    rejections.add(current)
                    training.remove(current)

                cluster_count += 1
                print "\nUnlearned", cluster_count, "cluster(s) so far.\n"

                detection_rate = self.current_detection_rate
                print "\nCurrent detection rate achieved is " + str(detection_rate) + ".\n"
                if outfile is not None:
                    if pollution_set3:
                        outfile.write(str(cluster_count) + ", " + str(rejection_count + cluster_count) + ": " +
                                      str(detection_rate) + ", " + str(cluster[1].size + 1) + ", " +
                                      str(cluster[1].target_set3()) + "\n")

                    else:
                        outfile.write(str(cluster_count) + ", " + str(rejection_count + cluster_count) + ": " +
                                      str(detection_rate) + ", " + str(cluster[1].size + 1) + ", " +
                                      str(cluster[1].target_set4()) + "\n")

                    outfile.flush()
                    os.fsync(outfile)

        if test:
            return cluster_list

        print "\nIteration through training space complete after", cluster_count, "clusters unlearned and", \
            rejection_count, "rejections made.\n"

        print "\nFinal detection rate: " + str(detection_rate) + ".\n"

    def shuffle_training(self):
        training = []
        train_examples = self.driver.tester.train_examples
        training = [train for train in chain(train_examples[0], train_examples[1], train_examples[2], train_examples[3])]
        return shuffle(training)

    def get_mislabeled(self, update=False):
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

    def select_initial(self, use_rowsum=False):
        """ Returns an email to be used as the initial unlearning email based on
            the mislabeled data (our tests show that the mislabeled and pollutant
            emails are strongly, ~80%, correlated) if option is true (which is default)."""
        mislabeled = self.get_mislabeled()
        print "Chosen: ", self.mislabeled_chosen
        print "Total Chosen: ", len(self.mislabeled_chosen)
        if use_rowsum:
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
                mislabeled_point = choice(list(mislabeled - self.mislabeled_chosen))
                self.mislabeled_chosen.add(mislabeled_point)
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
