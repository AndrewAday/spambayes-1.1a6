from random import choice, shuffle
from spambayes import TestDriver, quickselect
from Distance import distance
from itertools import chain
import sys
import gc
import copy
import os
from math import sqrt

phi = (1 + sqrt(5)) / 2
tol = 50
dec = 20


def chosen_sum(chosen, x, opt=None):
    """Given a given msg and a set of chosen emails, returns the sum of distances from the given msg."""
    s = 0
    for msg in chosen:
        s += distance(msg, x, opt)
    return s


def cluster_au(au, gold=False, test=False, pos_cluster_opt=0):
    """Clusters the training space of an ActiveUnlearner and returns the list of clusters."""
    print "\n-----------------------------------------------------\n"
    cluster_list = []
    training = au.shuffle_training()
    print "\nResetting mislabeled...\n"
    mislabeled = au.get_mislabeled(update=True)
    au.mislabeled_chosen = set()
    print "\nClustering...\n"
    original_training_size = len(training)
    while len(training) > 0:
        print "\n-----------------------------------------------------\n"
        print "\n" + str(len(training)) + " emails out of " + str(original_training_size) + \
              " still unclustered.\n"

        current = cluster_methods(au, "mislabeled", training, mislabeled)
        pre_cluster_rate = au.current_detection_rate

        cluster_result = determine_cluster(current, au, working_set=training, gold=gold, impact=True,
                                           pos_cluster_opt=pos_cluster_opt)
        while cluster_result is None:
            current = cluster_methods(au, "mislabeled", training, mislabeled)
            cluster_result = determine_cluster(current, au, working_set=training, gold=gold, impact=True,
                                               pos_cluster_opt=pos_cluster_opt)
        net_rate_change, cluster = cluster_result

        post_cluster_rate = au.current_detection_rate

        assert(post_cluster_rate == pre_cluster_rate), str(pre_cluster_rate) + " " + str(post_cluster_rate)

        cluster_list.append([net_rate_change, cluster])
        print "\nRemoving cluster from shuffled training set...\n"
        for email in cluster.cluster_set:
            training.remove(email)

    cluster_list.sort()
    print "\nClustering process done and sorted.\n"
    return cluster_list


def cluster_methods(au, method, working_set, mislabeled):
    """Given a desired clustering method, returns the next msg to cluster around."""
    if method == "random":
        return working_set[len(working_set) - 1]

    if method == "mislabeled":
        return au.select_initial(au.distance_opt, mislabeled, option="mislabeled", working_set=working_set)

    else:
        raise AssertionError("Please specify clustering method.")


def determine_cluster(center, au, pos_cluster_opt, working_set=None, gold=False, impact=False, test_waters=False):
    """Given a chosen starting center and a given increment of cluster size, it continues to grow and cluster more
    until the detection rate hits a maximum peak (i.e. optimal cluster); if first try is a decrease, reject this
    center and return False.
    """

    print "\nDetermining appropriate cluster around", center.tag, "...\n"
    old_detection_rate = au.current_detection_rate
    first_state_rate = au.current_detection_rate
    counter = 0
    cluster = Cluster(center, au.increment, au, working_set=working_set, distance_opt=au.distance_opt)

    # Test detection rate after unlearning cluster
    au.unlearn(cluster)
    au.init_ground()
    new_detection_rate = au.driver.tester.correct_classification_rate()

    if new_detection_rate <= old_detection_rate:    # Detection rate worsens - Reject
        print "\nCenter is inviable.\n"
        if pos_cluster_opt != 2:
            au.learn(cluster)
        second_state_rate = new_detection_rate
        net_rate_change = second_state_rate - first_state_rate
        au.current_detection_rate = first_state_rate
        if pos_cluster_opt == 1:
            return None

        elif pos_cluster_opt == 2:
            print "\nDecrementing until cluster is positive...\n"
            return neg_cluster_decrementer(au, first_state_rate, cluster)

        return net_rate_change, cluster

    elif cluster.size < au.increment:
        if impact:
            au.learn(cluster)
            second_state_rate = new_detection_rate
            net_rate_change = second_state_rate - first_state_rate
            au.current_detection_rate = first_state_rate
            return net_rate_change, cluster

        else:
            return cluster

    else:                                           # Detection rate improves - Grow cluster
        if gold:
            cluster = au.cluster_by_gold(cluster, old_detection_rate, new_detection_rate, counter, test_waters)

        else:
            cluster = au.cluster_by_increment(cluster, old_detection_rate, new_detection_rate, counter)

        if impact:
            au.learn(cluster)
            second_state_rate = au.current_detection_rate
            net_rate_change = second_state_rate - first_state_rate
            au.current_detection_rate = first_state_rate
            return net_rate_change, cluster

        else:
            return cluster


def cluster_print_stats(outfile, pollution_set3, detection_rate, cluster, cluster_count, attempt_count):
    """Prints stats for a given unlearned cluster and the present state of the machine after unlearning."""
    if outfile is not None:
        if pollution_set3:
            outfile.write(str(cluster_count) + ", " + str(attempt_count) + ": " + str(detection_rate) + ", " +
                          str(cluster[1].size) + ", " + str(cluster[1].target_set3()) + "\n")

        else:
            outfile.write(str(cluster_count) + ", " + str(attempt_count) + ": " + str(detection_rate) + ", " +
                          str(cluster[1].size) + ", " + str(cluster[1].target_set4()) + "\n")
        outfile.flush()
        os.fsync(outfile)

    else:
        pass


def neg_cluster_decrementer(au, first_state_rate, cluster):
    new_relearns = cluster.cluster_less(dec)
    au.divide_new_elements(new_relearns, False)
    au.init_ground()
    net_change_rate = au.driver.tester.correct_classification_rate() - first_state_rate
    while net_change_rate <= 0 < cluster.size:
        new_relearns = cluster.cluster_less(dec)
        au.divide_new_elements(new_relearns, False)
        au.init_ground()
        net_change_rate = au.driver.tester.correct_classification_rate() - first_state_rate

    if cluster.size > 0:
        return net_change_rate, cluster

    else:
        raise AssertionError


class Cluster:
    def __init__(self, msg, size, active_unlearner, distance_opt, working_set=None, sort_first=True, separate=True):
        self.clustroid = msg
        if msg.train == 1 or msg.train == 3:
            self.train = [1, 3]

        elif msg.train == 0 or msg.train == 2:
            self.train = [0, 2]
        self.size = size
        self.active_unlearner = active_unlearner
        self.sort_first = sort_first
        self.working_set = working_set
        self.ham = set()
        self.spam = set()
        self.opt = distance_opt
        self.dist_list = self.distance_array(separate)
        self.cluster_set = self.make_cluster()
        self.divide()

    def __repr__(self):
        return "(" + self.clustroid.tag + ", " + str(self.size) + ")"

    def distance_array(self, separate):
        """Returns a list containing the distances from each email to the center."""
        train_examples = self.active_unlearner.driver.tester.train_examples

        if separate:
            if self.working_set is None:
                dist_list = [(distance(self.clustroid, train, self.opt), train) for train in chain(train_examples[0],
                                                                                                   train_examples[1],
                                                                                                   train_examples[2],
                                                                                                   train_examples[3])
                             if train.train in self.train]

            else:
                dist_list = [(distance(self.clustroid, train, self.opt), train) for train in self.working_set if
                             train.train in self.train]
                assert(len(dist_list) > 0)
        else:
            if self.working_set is None:
                dist_list = [(distance(self.clustroid, train, self.opt), train) for train in chain(train_examples[0],
                                                                                                   train_examples[1],
                                                                                                   train_examples[2],
                                                                                                   train_examples[3])]

            else:
                dist_list = [(distance(self.clustroid, train, self.opt), train) for train in self.working_set]

        if self.sort_first:
            dist_list.sort()

        return dist_list

    def make_cluster(self):
        """Constructs the initial cluster of emails."""
        if self.size > len(self.dist_list):
            print "\nTruncating cluster size...\n"
            self.size = len(self.dist_list)

        if self.sort_first:
            return set(item[1] for item in self.dist_list[:self.size])

        else:
            k_smallest = quickselect.k_smallest
            return set(item[1] for item in k_smallest(self.dist_list, self.size))

    def divide(self):
        """Divides messages in the cluster between spam and ham."""
        for msg in self.cluster_set:
            if msg.train == 1 or msg.train == 3:
                self.ham.add(msg)
            elif msg.train == 0 or msg.train == 2:
                self.spam.add(msg)
            else:
                raise AssertionError

    def target_spam(self):
        """Returns a count of the number of spam emails in the cluster."""
        counter = 0
        for msg in self.cluster_set:
            if msg.tag.endswith(".spam.txt"):
                counter += 1

        return counter

    def target_set3(self):
        """Returns a count of the number of Set3 emails in the cluster."""
        counter = 0
        for msg in self.cluster_set:
            if "Set3" in msg.tag:
                counter += 1

        return counter

    def target_set4(self):
        """Returns a count of the number of Set4 emails in the cluster."""
        counter = 0
        for msg in self.cluster_set:
            if "Set4" in msg.tag:
                counter += 1

        return counter

    def cluster_more(self, n):
        """Expands the cluster to include n more emails and returns these additional emails.
           If n more is not available, cluster size is simply truncated to include all remaining
           emails."""
        old_cluster_set = self.cluster_set
        if self.size + n <= len(self.dist_list):
            self.size += n

        else:
            print "\nTruncating cluster size...\n"
            if len(self.dist_list) > 0:
                self.size = len(self.dist_list)

        if self.sort_first:
            new_cluster_set = set(item[1] for item in self.dist_list[:self.size])
        else:
            k_smallest = quickselect.k_smallest
            new_cluster_set = set(item[1] for item in k_smallest(self.dist_list, self.size))

        new_elements = list(item for item in new_cluster_set if item not in old_cluster_set)
        self.cluster_set = new_cluster_set

        assert(len(self.cluster_set) == self.size), len(self.cluster_set)

        for msg in new_elements:
            if msg.train == 1 or msg.train == 3:
                self.ham.add(msg)
            elif msg.train == 0 or msg.train == 2:
                self.spam.add(msg)

        return new_elements

    def cluster_less(self, n):
        """Contracts the cluster to include n less emails and returns the now newly excluded emails."""
        old_cluster_set = self.cluster_set
        self.size -= n
        assert(self.size >= 0), "Cluster size would become negative!"
        if self.sort_first:
            new_cluster_set = set(item[1] for item in self.dist_list[:self.size])
        else:
            k_smallest = quickselect.k_smallest
            new_cluster_set = set(item[1] for item in k_smallest(self.dist_list, self.size))

        new_elements = list(item for item in old_cluster_set if item not in new_cluster_set)
        self.cluster_set = new_cluster_set

        assert(len(self.cluster_set) == self.size), str(len(self.cluster_set)) + " " + str(self.size)
        assert(len(new_elements) == n), len(new_elements)        

        for msg in new_elements:
            if msg.train == 1 or msg.train == 3:
                self.ham.remove(msg)

            elif msg.train == 0 or msg.train == 2:
                self.spam.remove(msg)

        return new_elements


class ActiveUnlearner:
    def __init__(self, training_ham, training_spam, testing_ham, testing_spam, threshold=95, increment=100,
                 distance_opt="extreme", all_opt=False, update_opt="hybrid", greedy_opt=False):
        self.distance_opt = distance_opt
        self.all = all_opt
        self.greedy = greedy_opt
        self.update = update_opt
        self.increment = increment
        self.threshold = threshold
        self.driver = TestDriver.Driver()
        self.set_driver()
        self.hamspams = zip(training_ham, training_spam)
        self.set_data()
        self.testing_spam = testing_spam
        self.testing_ham = testing_ham
        self.set_training_nums()
        self.set_pol_nums()
        self.init_ground(True)
        self.mislabeled_chosen = set()
        self.training_chosen = set()
        self.current_detection_rate = self.driver.tester.correct_classification_rate()
        print "Initial detection rate:", self.current_detection_rate

    def set_driver(self):
        """Instantiates a new classifier for the driver."""
        self.driver.new_classifier()

    def set_data(self):
        """Trains ActiveUnlearner on provided training data."""
        for hamstream, spamstream in self.hamspams:
            self.driver.train(hamstream, spamstream)

    def init_ground(self, first_test=False, update=False):
        """Runs on the testing data to check the detection rate. If it's not the first test, it tests based on cached
        test msgs."""
        if first_test:
            self.driver.test(self.testing_ham, self.testing_spam, first_test, all_opt=self.all)

        else:
            if self.update == "pure":
                update = True
            self.driver.test(self.driver.tester.truth_examples[1], self.driver.tester.truth_examples[0], first_test,
                             update=update, all_opt=self.all)

    def set_training_nums(self):
        """Tests on initial vanilla training msgs to determine prob scores."""
        hamstream, spamstream = self.hamspams[0]
        self.driver.train_test(hamstream, spamstream, all_opt=self.all)

    def set_pol_nums(self):
        """Tests on initial polluted training msgs to determine prob scores."""
        hamstream, spamstream = self.hamspams[1]
        self.driver.pol_test(hamstream, spamstream, all_opt=self.all)

    def unlearn(self, cluster):
        """Unlearns a cluster from the ActiveUnlearner."""
        if len(cluster.ham) + len(cluster.spam) != cluster.size:
            print "\nUpdating cluster ham and spam sets...\n"
            cluster.divide()

        self.driver.untrain(cluster.ham, cluster.spam)

        for ham in cluster.ham:
            self.driver.tester.train_examples[ham.train].remove(ham)
        for spam in cluster.spam:
            self.driver.tester.train_examples[spam.train].remove(spam)

    def learn(self, cluster):
        """Learns a cluster from the ActiveUnlearner."""
        if len(cluster.ham) + len(cluster.spam) != cluster.size:
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
        Relearns the cluster afterwards.
        """
        self.unlearn(cluster)
        self.init_ground()
        detection_rate = self.driver.tester.correct_classification_rate()
        self.learn(cluster)
        return detection_rate

    def start_detect_rate(self, cluster):
        """Determines the detection rate after unlearning an initial cluster."""
        self.unlearn(cluster)
        self.init_ground()
        detection_rate = self.driver.tester.correct_classification_rate()
        return detection_rate

    def continue_detect_rate(self, cluster, n):
        """Determines the detection rate after growing a cluster to be unlearned."""
        old_cluster = copy.deepcopy(cluster.cluster_set)
        cluster.cluster_more(n)
        new_cluster = cluster.cluster_set

        new_unlearns = new_cluster - old_cluster
        assert(len(new_unlearns) == len(new_cluster) - len(old_cluster))
        assert(len(new_unlearns) == n), len(new_unlearns)

        unlearn_hams = []
        unlearn_spams = []

        for unlearn in new_unlearns:
            if unlearn.train == 1 or unlearn.train == 3:
                unlearn_hams.append(unlearn)

            elif unlearn.train == 0 or unlearn.train == 2:
                unlearn_spams.append(unlearn)

            self.driver.tester.train_examples[unlearn.train].remove(unlearn)

        self.driver.untrain(unlearn_hams, unlearn_spams)
        self.init_ground()
        detection_rate = self.driver.tester.correct_classification_rate()
        return detection_rate

    # --------------------------------------------------------------------------------------------------------------

    def divide_new_elements(self, messages, unlearn):
        """Divides a given set of emails to be unlearned into ham and spam lists and unlearns both."""
        hams = []
        spams = []
        for message in messages:
            if message.train == 1 or message.train == 3:
                hams.append(message)

            elif message.train == 0 or message.train == 2:
                spams.append(message)

            else:
                raise AssertionError("Message lacks train attribute.")

            if unlearn:
                self.driver.tester.train_examples[message.train].remove(message)

            else:
                self.driver.tester.train_examples[message.train].append(message)

        if unlearn:
            self.driver.untrain(hams, spams)

        else:
            self.driver.train(hams, spams)

    def cluster_by_increment(self, cluster, old_detection_rate, new_detection_rate, counter):
        """Finds an appropriate cluster around a msg by incrementing linearly until it reaches a peak detection rate."""
        while new_detection_rate > old_detection_rate:
            counter += 1
            print "\nExploring cluster of size", (counter + 1) * self.increment, "...\n"

            old_detection_rate = new_detection_rate
            new_unlearns = cluster.cluster_more(self.increment)

            assert(len(new_unlearns) == self.increment), len(new_unlearns)
            self.divide_new_elements(new_unlearns, True)
            self.init_ground()
            new_detection_rate = self.driver.tester.correct_classification_rate()

        # This part is done because we've clustered just past the peak point, so we need to go back
        # one increment and relearn the extra stuff.

        new_learns = cluster.cluster_less(self.increment)
        assert(cluster.size == self.increment * counter), counter
        self.divide_new_elements(new_learns, False)

        print "\nAppropriate cluster found, with size " + str(cluster.size) + ".\n"
        self.current_detection_rate = old_detection_rate
        return cluster

    def cluster_by_gold(self, cluster, old_detection_rate, new_detection_rate, counter, test_waters):
        """Finds an appropriate cluster around a msg by using the golden section search method."""
        sizes = [0]
        detection_rates = [old_detection_rate]

        new_unlearns = ['a', 'b', 'c']

        if test_waters:
            """First tries several incremental increases before trying golden section search."""
            while (new_detection_rate > old_detection_rate and cluster.size < self.increment * 3) and len(new_unlearns) \
                    > 0:
                counter += 1
                old_detection_rate = new_detection_rate
                print "\nExploring cluster of size", cluster.size + self.increment, "...\n"

                new_unlearns = cluster.cluster_more(self.increment)

                self.divide_new_elements(new_unlearns, True)
                self.init_ground()
                new_detection_rate = self.driver.tester.correct_classification_rate()

        if len(new_unlearns) > 0:
            if new_detection_rate > old_detection_rate:
                return self.try_gold(cluster, sizes, detection_rates, old_detection_rate, new_detection_rate, counter)

            else:
                new_learns = cluster.cluster_less(self.increment)
                self.divide_new_elements(new_learns, False)
                return cluster

        else:
            return cluster

    def try_gold(self, cluster, sizes, detection_rates, old_detection_rate, new_detection_rate, counter):
        extra_cluster = int(phi * cluster.size)
        """
        Performs golden section search on the size of a cluster; grows exponentially at a rate of phi to ensure that
        window ratios will be same at all levels (except edge cases), and uses this to determine the initial window.
        """
        while new_detection_rate > old_detection_rate:
            counter += 1

            sizes.append(cluster.size)
            detection_rates.append(new_detection_rate)
            old_detection_rate = new_detection_rate
            print "\nExploring cluster of size", cluster.size + int(round(extra_cluster)), "...\n"

            new_unlearns = cluster.cluster_more(int(round(extra_cluster)))
            extra_cluster *= phi

            self.divide_new_elements(new_unlearns, True)
            self.init_ground()
            new_detection_rate = self.driver.tester.correct_classification_rate()

        sizes.append(cluster.size)
        detection_rates.append(new_detection_rate)

        cluster, detection_rate, iterations = self.golden_section_search(cluster, sizes, detection_rates)
        print "\nAppropriate cluster found, with size " + str(cluster.size) + " after " + \
              str(counter + iterations) + " tries.\n"

        self.current_detection_rate = detection_rate
        return cluster

    def golden_section_search(self, cluster, sizes, detection_rates):
        """Performs golden section search on a cluster given a provided initial window."""
        print "\nPerforming golden section search...\n"

        left, middle_1, right = sizes[len(sizes) - 3], sizes[len(sizes) - 2], sizes[len(sizes) - 1]
        pointer = middle_1
        iterations = 0
        new_relearns = cluster.cluster_less(right - middle_1)
        self.divide_new_elements(new_relearns, False)

        assert(len(sizes) == len(detection_rates)), len(sizes) - len(detection_rates)
        f = dict(zip(sizes, detection_rates))

        middle_2 = right - (middle_1 - left)

        while abs(right - left) > tol:
            print "\nWindow is between " + str(left) + " and " + str(right) + ".\n"
            try:
                assert(middle_1 < middle_2)

            except AssertionError:
                self.switch_middles(middle_1, middle_2, cluster)

            print "Middles are " + str(middle_1) + " and " + str(middle_2) + ".\n"

            try:
                rate_1 = f[middle_1]

            except KeyError:
                rate_1 = self.evaluate_left_middle(pointer, middle_1, cluster, f)
                iterations += 1

            try:
                rate_2 = f[middle_2]

            except KeyError:
                rate_2 = self.evaluate_right_middle(pointer, middle_1, cluster, f)
                iterations += 1

            if rate_1 > rate_2:
                right = middle_2
                middle_2 = middle_1
                middle_1 = right - int((right - left) / phi)

            else:
                left = middle_1
                middle_1 = middle_2
                middle_2 = left + int((right - left) / phi)

        size = int(float(left + right) / 2)
        assert (left <= size <= right), str(left) + ", " + str(right)
        if pointer < size:
            new_unlearns = cluster.cluster_more(size - pointer)
            assert(cluster.size == size), str(size) + " " + str(cluster.size)
            self.divide_new_elements(new_unlearns, True)

        elif pointer > size:
            new_relearns = cluster.cluster_less(pointer - size)
            assert(cluster.size == size), str(size) + " " + str(cluster.size)
            self.divide_new_elements(new_relearns, False)

        else:
            raise AssertionError("Pointer is at the midpoint of the window.")

        self.init_ground()
        detection_rate = self.driver.tester.correct_classification_rate()
        iterations += 1

        return cluster, detection_rate, iterations

    def switch_middles(self, middle_1, middle_2, cluster):
        """
        Switches the middles during golden section search. This is necessary when the exponential probing reaches the
        end of the training space and causes problems of truncation.
        """
        print "\nSwitching out of order middles...\n"
        middle_1, middle_2 = middle_2, middle_1
        pointer = middle_1
        if cluster.size > pointer:
            new_relearns = cluster.cluster_less(cluster.size - pointer)
            self.divide_new_elements(new_relearns, False)

        elif cluster.size < pointer:
            new_unlearns = cluster.cluster_more(pointer - cluster.size)
            self.divide_new_elements(new_unlearns, True)

    def evaluate_left_middle(self, pointer, middle_1, cluster, f):
        """Evaluates the detection rate at the left middle during golden section search."""
        if pointer > middle_1:
            new_relearns = cluster.cluster_less(pointer - middle_1)
            pointer = middle_1
            print "Pointer is at " + str(pointer) + ".\n"
            assert(cluster.size == pointer), cluster.size
            self.divide_new_elements(new_relearns, False)
            self.init_ground()
            rate_1 = self.driver.tester.correct_classification_rate()
            f[middle_1] = rate_1

        elif pointer < middle_1:
            raise AssertionError("Pointer is on the left of middle_1.")

        else:
            assert(cluster.size == pointer), cluster.size
            self.init_ground()
            rate_1 = self.driver.tester.correct_classification_rate()
            if middle_1 in f:
                raise AssertionError("Key should not have been in f.")

            else:
                f[middle_1] = rate_1

        return rate_1

    def evaluate_right_middle(self, pointer, middle_2, cluster, f):
        """Evaluates the detection rate at the right middle during the golden section search."""
        if pointer < middle_2:
            new_unlearns = cluster.cluster_more(middle_2 - pointer)
            pointer = middle_2
            print "Pointer is at " + str(pointer) + ".\n"
            assert(cluster.size == pointer), cluster.size
            self.divide_new_elements(new_unlearns, True)
            self.init_ground()
            rate_2 = self.driver.tester.correct_classification_rate()
            f[middle_2] = rate_2

        elif pointer > middle_2:
            raise AssertionError("Pointer is on the right of middle_2.")

        else:
            raise AssertionError("Pointer is at the same location as middle_2.")

        return rate_2

    # -----------------------------------------------------------------------------------

    def brute_force_active_unlearn(self, outfile, test=False, center_iteration=True, pollution_set3=True, gold=False,
                                   pos_cluster_opt=0):
        """Attempts to improve the the machine by iterating through the training space and unlearning any clusters that
        improve the state of the machine.
        """
        cluster_list = []
        try:
            cluster_count = 0
            rejection_count = 0
            rejections = set()
            training = self.shuffle_training()
            original_training_size = len(training)
            detection_rate = self.current_detection_rate
            print "\nCurrent detection rate achieved is " + str(detection_rate) + ".\n"

            while len(training) > 0:
                print "\n-----------------------------------------------------\n"
                print "\nStarting new round of untraining;", len(training), "out of", original_training_size, "training left" \
                                                                                                              ".\n"

                current = training[len(training) - 1]
                cluster = determine_cluster(current, self, working_set=training, gold=gold,
                                            pos_cluster_opt=pos_cluster_opt)

                if cluster[0] <= 0:
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

                        rejection_count += 1

                    print "\nRejected", rejection_count, "attempt(s) so far.\n"

                else:
                    cluster_list.append(cluster[1])
                    print "\nRemoving cluster from shuffled training set...\n"

                    for msg in cluster[1].cluster_set:
                        if msg not in rejections:
                            training.remove(msg)
                            rejections.add(msg)

                    cluster_count += 1
                    print "\nUnlearned", cluster_count, "cluster(s) so far.\n"

                    detection_rate = self.current_detection_rate
                    print "\nCurrent detection rate achieved is " + str(detection_rate) + ".\n"
                    cluster_print_stats(outfile, pollution_set3, detection_rate, cluster, cluster_count,
                                        rejection_count + cluster_count)

            print "\nIteration through training space complete after", cluster_count, "clusters unlearned and", \
                rejection_count, "rejections made.\n"

            print "\nFinal detection rate: " + str(detection_rate) + ".\n"

            if test:
                return cluster_list

        except KeyboardInterrupt:
            return cluster_list

    def impact_active_unlearn(self, outfile, test=False, pollution_set3=True, gold=False, pos_cluster_opt=0):
        """
        Attempts to improve the machine by first clustering the training space and then unlearning clusters based off
        of perceived impact to the machine.

        pos_cluster_opt values: 0 = treat negative clusters like any other cluster, 1 = only form positive clusters,
        2 = shrink negative clusters until positive, 3 = ignore negative clusters after clustering (only applicable in
        greedy checking)
        """
        unlearned_cluster_list = []
        try:
            cluster_count = 0
            attempt_count = 0
            detection_rate = self.current_detection_rate

            cluster_count, attempt_count = self.lazy_unlearn(detection_rate, unlearned_cluster_list,
                                                             cluster_count, attempt_count,
                                                             outfile, pollution_set3, gold, pos_cluster_opt)

            print "\nThreshold achieved or all clusters consumed after", cluster_count, "clusters unlearned and", \
                attempt_count, "clustering attempts.\n"

            print "\nFinal detection rate: " + str(self.current_detection_rate) + ".\n"
            if test:
                return unlearned_cluster_list

        except KeyboardInterrupt:
            return unlearned_cluster_list

    # -----------------------------------------------------------------------------------

    def lazy_unlearn(self, detection_rate, unlearned_cluster_list, cluster_count, attempt_count, outfile,
                     pollution_set3, gold, pos_cluster_opt):
        """
        After clustering, unlearns all clusters with positive impact in the cluster list, in reverse order. This is
        due to the fact that going in the regular order usually first unlearns a large cluster that is actually not
        polluted.

        This is because in the polluted state of the machine, this first big cluster is perceived as a high
        impact cluster, but after unlearning several (large) polluted clusters first (with slightly smaller impact but
        still significant), this preserves the large (and unpolluted) cluster.
        """
        cluster_list = cluster_au(self, gold=gold, pos_cluster_opt=pos_cluster_opt)
        attempt_count += 1

        while detection_rate <= self.threshold and cluster_list[len(cluster_list) - 1][0] > 0:
            list_length = len(cluster_list)
            j = 0
            if not self.greedy:
                while cluster_list[j][0] <= 0:
                    j += 1

                indices = range(j, len(cluster_list))

            else:
                indices = list(reversed(range(j, len(cluster_list))))

            for i in indices:
                cluster = cluster_list[i]
                print "\n-----------------------------------------------------\n"
                print "\nChecking cluster " + str(j + 1) + " of " + str(list_length) + "...\n"
                j += 1
                old_detection_rate = detection_rate
                if pos_cluster_opt == 3 and self.greedy:
                    if cluster[0] <= 0:
                        continue
                self.unlearn(cluster[1])
                self.init_ground(update=True)
                detection_rate = self.driver.tester.correct_classification_rate()
                if detection_rate > old_detection_rate:
                    cluster_count += 1
                    unlearned_cluster_list.append(cluster)
                    self.current_detection_rate = detection_rate
                    cluster_print_stats(outfile, pollution_set3, detection_rate, cluster, cluster_count, attempt_count)
                    print "\nCurrent detection rate achieved is " + str(detection_rate) + ".\n"

                else:
                    self.learn(cluster[1])
                    detection_rate = old_detection_rate

            if detection_rate > self.threshold:
                break

            else:
                # Below actually deletes the contents of the list, and removes all of the clusters from the namespace
                # del cluster_list simply removes the label from the namespace
                del cluster_list[:]
                cluster_list = cluster_au(self, gold, pos_cluster_opt=pos_cluster_opt)
                attempt_count += 1
                gc.collect()

        return cluster_count, attempt_count

    # -----------------------------------------------------------------------------------

    def shuffle_training(self):
        """Copies the training space and returns a shuffled working set. This provides the simulation of randomly
        iterating through the training space, without the complication of actually modifying the training space itself
        while doing so.
        """
        train_examples = self.driver.tester.train_examples
        training = [train for train in chain(train_examples[0], train_examples[1], train_examples[2],
                                             train_examples[3])]
        shuffle(training)
        return training

    def get_mislabeled(self, update=False):
        """
        Returns the set of mislabeled emails (from the ground truth) based off of the
        current classifier state. By default assumes the current state's numbers and
        tester false positives/negatives have already been generated; if not, it'll run the
        predict method from the tester.
        """
        if update:
            self.init_ground()

        mislabeled = set()
        tester = self.driver.tester
        for wrong_ham in tester.ham_wrong_examples:
            mislabeled.add(wrong_ham)

        for wrong_spam in tester.spam_wrong_examples:
            mislabeled.add(wrong_spam)

        for unsure in tester.unsure_examples:
            mislabeled.add(unsure)

        return mislabeled

    def select_initial(self, mislabeled=None, option="mislabeled", working_set=None):
        """Returns an email to be used as the next seed for a cluster."""

        if option == "row_sum":
            return self.row_sum_initial(working_set, mislabeled)

        if option == "mislabeled":
            return self.mislabeled_initial(working_set, mislabeled)

        if option == "max_sum":
            return self.max_sum_initial(working_set)

    def row_sum_initial(self, working_set, mislabeled):
        """Returns the email with the smallest row sum from the set of mislabeled emails."""
        if mislabeled is None:
            mislabeled = self.get_mislabeled()
        t_e = self.driver.tester.train_examples
        minrowsum = sys.maxint
        init_email = None

        training = chain(t_e[0], t_e[1], t_e[2], t_e[3]) if working_set is None else working_set

        for email in training:
            rowsum = 0
            for email2 in mislabeled:
                dist = distance(email, email2, self.distance_opt)
                rowsum += dist ** 2
            if rowsum < minrowsum:
                minrowsum = rowsum
                init_email = email

        return init_email

    def mislabeled_initial(self, working_set, mislabeled):
        """Chooses an arbitrary point from the mislabeled emails and returns the training email closest to it."""
        if mislabeled is None:
            mislabeled = self.get_mislabeled()
        t_e = self.driver.tester.train_examples

        print "Total Chosen: ", len(self.mislabeled_chosen)

        try:
            mislabeled_point = choice(list(mislabeled - self.mislabeled_chosen))
            self.mislabeled_chosen.add(mislabeled_point)
        except:
            raise AssertionError(str(mislabeled))

        min_distance = sys.maxint
        init_email = None

        training = chain(t_e[0], t_e[1], t_e[2], t_e[3]) if working_set is None else working_set

        for email in training:
            current_distance = distance(email, mislabeled_point, self.distance_opt)
            if current_distance < min_distance:
                init_email = email
                min_distance = current_distance

        return init_email

    def max_sum_initial(self, working_set):
        """
        Returns the email that is "furthest" from the set of chosen seeds, by finding the email with the highest
        sum of distances.
        """
        print "Total Chosen: ", len(self.training_chosen)
        t_e = self.driver.tester.train_examples

        try:
            max_sum = 0
            init_email = None

            training = chain(t_e[0], t_e[1], t_e[2], t_e[3]) if working_set is None else working_set

            for email in training:
                current_sum = chosen_sum(self.training_chosen, email, self.distance_opt)
                if current_sum > max_sum and email not in self.training_chosen:
                    init_email = email
                    max_sum = current_sum

            assert(init_email is not None)
            self.training_chosen.add(init_email)
            return init_email

        except AssertionError:
            print "Returning initial seed based off of mislabeled...\n"
            init_email = self.select_initial(option="mislabeled")
            self.training_chosen.add(init_email)
            return init_email
