from spambayes.Options import get_pathname_option
from benignfilemover import BenignFileMover
from os import listdir
from foo import random_words

class InjectionPolluter:

    # Inject a common feature into some mislabeled benign data samples

    def __init__(self, number, number_inject = 0, inject_type = 1):
        self.number = number

        self.h_injected = get_pathname_option("TestDriver", "ham_directories") % 3 + "/"
        self.s_injected = get_pathname_option("TestDriver", "spam_directories") % 3 + "/"

        if inject_type is 1:
            self.spam_feature = "$"

            self.feature = self.spam_feature
        elif inject_type is 2:
            self.ham_feature = ""

            self.feature = self.ham_feature
        elif inject_type is 3:
            self.number_inject = number_inject

            self.random_feature = ""
            for word in random_words(self.number_inject):
                self.random_feature = self.random_feature + word + " "

            self.feature = self.random_feature

    # Remove all injected features
    def reset(self):
        print "Resetting ..."

        for email in listdir(self.h_injected):
            print "Clearing pollution from " + email
            current = open(self.h_injected + email, 'r')
            lines = current.readlines()
            current.close()
            new_lines = []
            for line in lines:
                if self.feature not in line:
                    new_lines.append(line)
                else:
                    continue
            print new_lines

            current = open(self.h_injected + email, 'w')
            for line in new_lines:
                current.write(line)
            current.close()

        for email in listdir(self.s_injected):
            print "Clearing pollution from " + email
            current = open(self.s_injected + email, 'r')
            lines = current.readlines()
            current.close()
            new_lines = []
            for line in lines:
                if self.feature not in line:
                    new_lines.append(line)
                else:
                    continue

            current = open(self.s_injected + email, 'w')
            for line in new_lines:
                current.write(line)
            current.close()

    def injectfeatures(self):
        for email in listdir(self.h_injected):
            current = open(self.h_injected + email, 'a')
            current.write("\n" + self.feature)
            current.close()
        for email in listdir(self.s_injected):
            current = open(self.s_injected + email, 'a')
            current.write("\n" + self.feature)
            current.close()

def main():
    IP = InjectionPolluter(6000, 5, 3)

    IP.reset()


if __name__ == "__main__":
    main()
