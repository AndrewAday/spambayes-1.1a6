from random import choice, shuffle
from spambayes import TestDriver, quickselect
from Distance import distance
from itertools import chain
import sys
import copy
import os
from math import sqrt

phi = (1 + sqrt(5)) / 2


def chosen_sum(chosen, x, opt=None):
    s = 0
    for msg in chosen:
        s += distance(msg, x, opt)
    return s


def cluster_au(au, gold=False, test=False):
    cluster_list = []
    training = au.shuffle_training()
    print "\nResetting mislabeled...\n"
    mislabeled = au.get_mislabeled(update=True)
    au.mislabeled_chosen = set()
    original_training_size = len(training)
    while len(training) > 0:
        print "\n-----------------------------------------------------\n"
        print "\n" + str(len(training)) + " emails out of " + str(original_training_size) + \
              " still unclustered.\n"

        current = cluster_methods(au, "mislabeled", training, mislabeled)
        pre_cluster_rate = au.current_detection_rate
        net_rate_change, cluster = determine_cluster(current, au, working_set=training, gold=gold, impact=True)
        post_cluster_rate = au.current_detection_rate

        assert(post_cluster_rate == pre_cluster_rate), str(pre_cluster_rate) + " " + str(post_cluster_rate)

        cluster_list.append([net_rate_change, cluster])
        print "\nRemoving cluster from shuffled training set...\n"
        for email in cluster[1].cluster_set:
            training.remove(email)

    cluster_list.sort()
    print "\nClustering process done and sorted.\n"
    if not test:
        cluster_list = [pair[1] for pair in cluster_list]
        return cluster_list

    else:
        return cluster_list


def cluster_methods(au, method, working_set, mislabeled):
    if method == "random":
        return working_set[len(working_set) - 1]

    if method == "mislabeled":
        return au.select_initial(au.distance_opt, mislabeled, option="mislabeled", working_set=working_set)

    else:
        raise AssertionError("Please specify clustering method.")


def determine_cluster(center, au, working_set=None, gold=False, tolerance=50, impact=False, test_waters=False):
    """ Given a chosen starting center and a given increment of cluster size, it continues to grow and cluster more
        until the detection rate hits a maximum peak (i.e. optimal cluster); if first try is a decrease, reject this
        center and return False."""

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
        au.learn(cluster)
        second_state_rate = new_detection_rate
        net_rate_change = second_state_rate - first_state_rate
        au.current_detection_rate = first_state_rate
        if impact:
            return net_rate_change, (False, cluster, None)

        else:
            return False, cluster, None

    elif cluster.size < au.increment:
        if impact:
            au.learn(cluster)
            second_state_rate = new_detection_rate
            net_rate_change = second_state_rate - first_state_rate
            au.current_detection_rate = first_state_rate
            return net_rate_change, (True, cluster, None)

        else:
            return True, cluster, None

    else:                                           # Detection rate improves - Grow cluster
        if gold:
            cluster = au.cluster_by_gold(cluster, old_detection_rate, new_detection_rate, counter, tolerance,
                                         test_waters)

        else:
            cluster = au.cluster_by_increment(cluster, old_detection_rate, new_detection_rate, counter)

        if impact:
            au.learn(cluster[1])
            second_state_rate = au.current_detection_rate
            net_rate_change = second_state_rate - first_state_rate
            au.current_detection_rate = first_state_rate
            return net_rate_change, cluster

        else:
            return cluster


def cluster_print_stats(outfile, pollution_set3, detection_rate, cluster, cluster_count, attempt_count):
    if outfile is not None:
        if pollution_set3:
            outfile.write(str(cluster_count) + ", " + str(attempt_count) + ": " + str(detection_rate) + ", " +
                          str(cluster[1].size) + ", " + str(cluster[1].target_set3()) + ", " + str(cluster[2]) +
                          "\n")

        else:
            outfile.write(str(cluster_count) + ", " + str(attempt_count) + ": " + str(detection_rate) + ", " +
                          str(cluster[1].size) + ", " + str(cluster[1].target_set4()) + ", " + str(cluster[2]) +
                          "\n")
        outfile.flush()
        os.fsync(outfile)

    else:
        pass


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
                 distance_opt="extreme"):
        self.distance_opt = distance_opt
        self.increment = increment
        self.threshold = threshold
        self.driver = TestDriver.Driver()
        self.set_driver()
        self.hamspams = zip(training_ham, training_spam)
        self.set_data()
        self.testing_spam = testing_spam
        self.testing_ham = testing_ham
        self.set_training_nums()
        self.set_dict_nums()
        self.init_ground(True)
        self.mislabeled_chosen = set()
        self.training_chosen = set()
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
        if len(cluster.ham) + len(cluster.spam) != cluster.size:
            print "\nUpdating cluster ham and spam sets...\n"
            cluster.divide()

        self.driver.untrain(cluster.ham, cluster.spam)

        for ham in cluster.ham:
            self.driver.tester.train_examples[ham.train].remove(ham)
        for spam in cluster.spam:
            self.driver.tester.train_examples[spam.train].remove(spam)

    def learn(self, cluster):
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
        return True, cluster, None

    def cluster_by_gold(self, cluster, old_detection_rate, new_detection_rate, counter, tolerance, test_waters):
        sizes = [0]
        detection_rates = [old_detection_rate]

        new_unlearns = ['a', 'b', 'c']

        if test_waters:
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
                return self.try_gold(cluster, sizes, detection_rates, old_detection_rate, new_detection_rate, counter,
                                     tolerance)

            else:
                new_learns = cluster.cluster_less(self.increment)
                self.divide_new_elements(new_learns, False)
                return True, cluster, None

        else:
            return True, cluster, None

    def try_gold(self, cluster, sizes, detection_rates, old_detection_rate, new_detection_rate, counter, tolerance):
        extra_cluster = int(phi * cluster.size)
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

        cluster, detection_rate, iterations = self.golden_section_search(cluster, len(sizes) - 3,
                                                                         len(sizes) - 2, len(sizes) - 1,
                                                                         tolerance, sizes, detection_rates)
        print "\nAppropriate cluster found, with size " + str(cluster.size) + " after " + \
              str(counter + iterations) + " tries.\n"

        if (counter + iterations) <= float(cluster.size) / float(self.increment):
            print "Gold is at least as efficient as straight up incrementing.\n"
            efficient = True

        else:
            print "Gold is less efficient than striaght up incrementing.\n"
            efficient = False

        self.current_detection_rate = detection_rate
        return True, cluster, efficient

    def golden_section_search(self, cluster, left_index, middle_index, right_index, tolerance, sizes, detection_rates):
        print "\nPerforming golden section search...\n"

        left = sizes[left_index]
        middle_1 = sizes[middle_index]
        right = sizes[right_index]
        pointer = middle_1
        iterations = 0
        new_relearns = cluster.cluster_less(right - middle_1)
        self.divide_new_elements(new_relearns, False)

        assert(len(sizes) == len(detection_rates)), len(sizes) - len(detection_rates)
        f = dict(zip(sizes, detection_rates))

        middle_2 = right - (middle_1 - left)

        while abs(right - left) > tolerance:
            print "\nWindow is between " + str(left) + " and " + str(right) + ".\n"
            try:
                assert(middle_1 < middle_2)

            except AssertionError:
                print "\nSwitching out of order middles...\n"
                middle_1, middle_2 = middle_2, middle_1
                pointer = middle_1
                if cluster.size > pointer:
                    new_relearns = cluster.cluster_less(cluster.size - pointer)
                    self.divide_new_elements(new_relearns, False)

                elif cluster.size < pointer:
                    new_unlearns = cluster.cluster_more(pointer - cluster.size)
                    self.divide_new_elements(new_unlearns, True)

            print "Middles are " + str(middle_1) + " and " + str(middle_2) + ".\n"
            try:
                rate_1 = f[middle_1]

            except KeyError:
                if pointer > middle_1:
                    new_relearns = cluster.cluster_less(pointer - middle_1)
                    pointer = middle_1
                    print "Pointer is at " + str(pointer) + ".\n"
                    assert(cluster.size == pointer), cluster.size
                    self.divide_new_elements(new_relearns, False)
                    self.init_ground()
                    rate_1 = self.driver.tester.correct_classification_rate()
                    iterations += 1
                    f[middle_1] = rate_1

                elif pointer < middle_1:
                    raise AssertionError("Pointer is on the left of middle_1.")

                else:
                    assert(cluster.size == pointer), cluster.size
                    self.init_ground()
                    rate_1 = self.driver.tester.correct_classification_rate()
                    iterations += 1
                    if middle_1 in f:
                        raise AssertionError("Key should not have been in f.")

                    else:
                        f[middle_1] = rate_1

            try:
                rate_2 = f[middle_2]

            except KeyError:
                if pointer < middle_2:
                    new_unlearns = cluster.cluster_more(middle_2 - pointer)
                    pointer = middle_2
                    print "Pointer is at " + str(pointer) + ".\n"
                    assert(cluster.size == pointer), cluster.size
                    self.divide_new_elements(new_unlearns, True)
                    self.init_ground()
                    rate_2 = self.driver.tester.correct_classification_rate()
                    iterations += 1
                    f[middle_2] = rate_2

                elif pointer > middle_2:
                    raise AssertionError("Pointer is on the right of middle_2.")

                else:
                    raise AssertionError("Pointer is at the same location as middle_2.")

            if rate_1 > rate_2:
                right = middle_2
                middle_2 = middle_1
                middle_1 = right - int((right - left) / phi)

            else:
                left = middle_1
                middle_1 = middle_2
                middle_2 = left + int((right - left) / phi)

        size = int(float(left + right) / 2)
        assert (left <= size), left
        assert (size <= right), right
        if pointer < size:
            new_unlearns = cluster.cluster_more(size - pointer)
            assert(cluster.size == size), str(size) + " " + str(cluster.size)
            self.divide_new_elements(new_unlearns, True)
            self.init_ground()
            detection_rate = self.driver.tester.correct_classification_rate()
            iterations += 1

        elif pointer > size:
            new_relearns = cluster.cluster_less(pointer - size)
            assert(cluster.size == size), str(size) + " " + str(cluster.size)
            self.divide_new_elements(new_relearns, False)
            self.init_ground()
            detection_rate = self.driver.tester.correct_classification_rate()
            iterations += 1

        else:
            raise AssertionError("Pointer is at the midpoint of the window.")

        return cluster, detection_rate, iterations

    # -----------------------------------------------------------------------------------
    def active_unlearn(self, outfile, test=False, pollution_set3=True, gold=False, select_initial="mislabeled"):
        cluster_list = []
        try:
            cluster_count = 0
            attempt_count = 0
            detection_rate = self.current_detection_rate

            while detection_rate < self.threshold:
                print "\n-----------------------------------------------------\n"
                current = self.select_initial(self.distance_opt, select_initial)
                attempt_count += 1
                cluster = determine_cluster(current, self, gold=gold)
                print "\nAttempted", attempt_count, "attempt(s) so far.\n"

                while not cluster[0]:
                    print "\n-----------------------------------------------------\n"
                    current = self.select_initial(select_initial, self.distance_opt)
                    attempt_count += 1
                    cluster = determine_cluster(current, self, gold=gold)
                    print "\nAttempted", attempt_count, "attempt(s) so far.\n"

                cluster_list.append(cluster[1])
                cluster_count += 1
                print "\nUnlearned", cluster_count, "cluster(s) so far.\n"

                detection_rate = self.current_detection_rate
                print "\nCurrent detection rate achieved is " + str(detection_rate) + ".\n"
                cluster_print_stats(outfile, pollution_set3, detection_rate, cluster, cluster_count,
                                    attempt_count)

            if test:
                return cluster_list

            print "\nThreshold achieved after", cluster_count, "clusters unlearned and", attempt_count, "attempts.\n"

            print "\nFinal detection rate: " + str(detection_rate) + ".\n"

        except KeyboardInterrupt:
            return cluster_list

    # -----------------------------------------------------------------------------------

    def brute_force_active_unlearn(self, outfile, test=False, center_iteration=True, pollution_set3=True, gold=False):
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
                cluster = determine_cluster(current, self, working_set=training, gold=gold)

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
    # -----------------------------------------------------------------------------------

    def greatest_impact_active_unlearn(self, outfile, test=False, pollution_set3=True, gold=False, working_model=False,
                                       unlearn_method="vigilant"):
        unlearned_cluster_list = []
        try:
            cluster_count = 0
            attempt_count = 0
            detection_rate = self.current_detection_rate
            old_detection_rate = self.current_detection_rate
            if working_model:
                training_ham = self.driver.tester.train_examples[1] + self.driver.tester.train_examples[3]
                training_spam = self.driver.tester.train_examples[0] + self.driver.tester.train_examples[2]
                testing_ham = self.testing_ham
                testing_spam = self.testing_spam

                print "\nTraining model active unlearner...\n"
                working_au = ActiveUnlearner(training_ham, training_spam, testing_ham, testing_spam)

                print "\n-----------------------------------------------------\n"
                print "\nClustering...\n"
                cluster_list = cluster_au(au=working_au, gold=gold)

            else:
                print "\n-----------------------------------------------------\n"
                print "\nClustering...\n"
                cluster_list = cluster_au(self, gold=gold)

            cluster = None
            if unlearn_method == "frugal":
                cluster_count, attempt_count = self.frugal_unlearn(old_detection_rate, detection_rate, cluster,
                                                                   cluster_list, unlearned_cluster_list,
                                                                   cluster_count, attempt_count, outfile,
                                                                   pollution_set3)

            elif unlearn_method == "vigilant":
                cluster_count, attempt_count = self.vigilant_unlearn(detection_rate, cluster_list,
                                                                     unlearned_cluster_list, cluster_count,
                                                                     attempt_count,
                                                                     outfile, pollution_set3, gold)

            elif unlearn_method == "lazy":
                cluster_count, attempt_count = self.lazy_unlearn(detection_rate, cluster_list, unlearned_cluster_list,
                                                                 cluster_count, attempt_count,
                                                                 outfile, pollution_set3, gold)

            print "\nThreshold achieved or all clusters consumed after", cluster_count, "clusters unlearned and", \
                attempt_count, "clustering attempts.\n"

            print "\nFinal detection rate: " + str(self.current_detection_rate) + ".\n"
            if test:
                return unlearned_cluster_list

        except KeyboardInterrupt:
            return unlearned_cluster_list

    # -----------------------------------------------------------------------------------
    def vigilant_unlearn(self, detection_rate, cluster_list, unlearned_cluster_list,
                         cluster_count, attempt_count, outfile, pollution_set3, gold):
        while detection_rate <= self.threshold and cluster_list[len(cluster_list) - 1][0] > 0:
            print "\n-----------------------------------------------------\n"
            cluster_count += 1
            attempt_count += 1
            cluster = cluster_list[len(cluster_list) - 1]
            self.unlearn(cluster[1])
            unlearned_cluster_list.append(cluster)
            self.init_ground()
            detection_rate = self.driver.tester.correct_classification_rate()
            self.current_detection_rate = detection_rate
            cluster_print_stats(outfile, pollution_set3, detection_rate, cluster, cluster_count, attempt_count)
            print "\nCurrent detection rate achieved is " + str(detection_rate) + ".\n"
            print "\nClustering...\n"
            cluster_list = []
            cluster_list = cluster_au(self, gold)

        return cluster_count, attempt_count

    def lazy_unlearn(self, detection_rate, cluster_list, unlearned_cluster_list, cluster_count, attempt_count, outfile,
                     pollution_set3, gold):
        attempt_count += 1
        while detection_rate <= self.threshold and cluster_list[len(cluster_list) - 1][0] > 0:
            list_length = len(cluster_list)
            counter = 1
            for i in range(len(cluster_list)):
                cluster = cluster_list[i]
                print "\n-----------------------------------------------------\n"
                print "\nChecking cluster " + str(counter) + " of " + str(list_length) + "...\n"
                counter += 1
                old_detection_rate = detection_rate
                self.unlearn(cluster[1])
                self.init_ground()
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
                cluster_list = []
                cluster_list = cluster_au(self, gold)
                attempt_count += 1

        return cluster_count, attempt_count

    def frugal_unlearn(self, old_detection_rate, detection_rate, cluster, cluster_list, unlearned_cluster_list,
                       cluster_count, attempt_count, outfile, pollution_set3):
        while old_detection_rate <= detection_rate <= self.threshold and len(cluster_list) > 0:
            print "\n-----------------------------------------------------\n"
            cluster_count += 1
            attempt_count += 1
            cluster = cluster_list[len(cluster_list) - 1]
            old_detection_rate = detection_rate
            self.unlearn(cluster[1])
            cluster_list.remove(cluster)
            unlearned_cluster_list.append(cluster)
            self.init_ground()
            detection_rate = self.driver.tester.correct_classification_rate()
            cluster_print_stats(outfile, pollution_set3, detection_rate, cluster, cluster_count, attempt_count)
            print "\nCurrent detection rate achieved is " + str(detection_rate) + ".\n"

        if detection_rate < old_detection_rate:
            print "\nReversing back one cluster...\n"
            self.learn(cluster[1])
            cluster_list.append(cluster)
            cluster_count -= 1
            unlearned_cluster_list.remove(cluster)
            self.init_ground()
            detection_rate = self.driver.tester.correct_classification_rate()
            print "\nCurrent detection rate achieved is " + str(detection_rate) + ".\n"

        return cluster_count, attempt_count

    # -----------------------------------------------------------------------------------

    def shuffle_training(self):
        train_examples = self.driver.tester.train_examples
        training = [train for train in chain(train_examples[0], train_examples[1], train_examples[2],
                                             train_examples[3])]
        shuffle(training)
        return training

    def get_mislabeled(self, update=False):
        """ Returns the set of mislabeled emails (from the ground truth) based off of the
            current classifier state. By default assumes the current state's numbers and
            tester false positives/negatives have already been generated; if not, it'll run the
            predict method from the tester."""
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

    def select_initial(self, distance_opt, mislabeled=None, option="mislabeled", working_set=None):
        """ Returns an email to be used as the initial unlearning email based on
            the mislabeled data (our tests show that the mislabeled and pollutant
            emails are strongly, ~80%, correlated) if option is true (which is default)."""
        if mislabeled is None:
            mislabeled = self.get_mislabeled()

        t_e = self.driver.tester.train_examples
        if option == "rowsum":
            # We want to minimize the distances (rowsum) between the email we select
            # and the mislabeled emails. This ensures that the initial email we select
            # is correlated with the mislabeled emails.

            minrowsum = sys.maxint
            init_email = None

            training = chain(t_e[0], t_e[1], t_e[2], t_e[3]) if working_set is None else working_set

            for email in training:
                rowsum = 0
                for email2 in mislabeled:
                    dist = distance(email, email2, distance_opt)
                    rowsum += dist ** 2
                if rowsum < minrowsum:
                    minrowsum = rowsum
                    init_email = email

            return init_email

        if option == "mislabeled":
            print "Total Chosen: ", len(self.mislabeled_chosen)
            # This chooses an arbitrary point from the mislabeled emails and simply finds the email
            # in training that is closest to this point.
            try:
                mislabeled_point = choice(list(mislabeled - self.mislabeled_chosen))
                self.mislabeled_chosen.add(mislabeled_point)
            except:
                raise AssertionError(str(mislabeled))

            min_distance = sys.maxint
            init_email = None

            training = chain(t_e[0], t_e[1], t_e[2], t_e[3]) if working_set is None else working_set

            for email in training:
                current_distance = distance(email, mislabeled_point, distance_opt)
                if current_distance < min_distance:
                    init_email = email
                    min_distance = current_distance

            return init_email

        if option == "max_sum":
            print "Total Chosen: ", len(self.training_chosen)
            try:
                max_sum = 0
                init_email = None

                training = chain(t_e[0], t_e[1], t_e[2], t_e[3]) if working_set is None else working_set

                for email in training:
                    current_sum = chosen_sum(self.training_chosen, email, distance_opt)
                    if current_sum > max_sum and email not in self.training_chosen:
                        init_email = email
                        max_sum = current_sum

                assert(init_email is not None)
                self.training_chosen.add(init_email)
                return init_email

            except AssertionError:
                print "Returning initial seed based off of mislabeled...\n"
                init_email = self.select_initial(self.distance_opt, option="mislabeled")
                self.training_chosen.add(init_email)
                return init_email
