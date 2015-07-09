from spambayes.Options import get_pathname_option
import mislabeledfilemover, benignfilemover
from os import listdir

class InjectionPolluter:

    # Inject a common feature into some mislabeled benign data samples

    def __init__(self, number):
        self.number = number
        self.mislabeler = benignfilemover.BenignFileMover(self.number)

        self.h_mislabeled = get_pathname_option("TestDriver", "ham_directories") % 3 + "/"
        self.s_mislabeled = get_pathname_option("TestDriver", "spam_directories") % 3 + "/"

        self.feature = "!!! " \
                       "$$$ " \
                       "#1 " \
                       "100% " \
                       "Free " \
                       "Act Now! " \
                       "All natural " \
                       "As seen on " \
                       "Attention " \
                       "Bad credit " \
                       "Bargain " \
                       "Best price " \
                       "Billion " \
                       "Certified " \
                       "Cost " \
                       "Dear Friend " \
                       "Decision " \
                       "Discount " \
                       "Double your income " \
                       "Eliminate debt " \
                       "Extra income " \
                       "Fast cash " \
                       "Fees " \
                       "Financial freedom " \
                       "FREE " \
                       "Guarantee " \
                       "Hot " \
                       "Horny " \
                       "Increase " \
                       "Join millions " \
                       "Lose weight " \
                       "Lowest price " \
                       "Make money fast " \
                       "Marketing " \
                       "Million dollars " \
                       "Money " \
                       "Money making " \
                       "No medical exams " \
                       "No purchase necessary " \
                       "Offer " \
                       "Online pharmacy " \
                       "Opportunity " \
                       "Partners " \
                       "Performance " \
                       "Please read " \
                       "Rates " \
                       "Satisfaction guaranteed " \
                       "Selling " \
                       "Sex " \
                       "Success " \
                       "Trial " \
                       "Visit our website "

    # Remove all injected features
    def reset(self):
        print "Resetting ..."

        for email in listdir(self.h_mislabeled):
            print "Clearing pollution from " + email
            current = open(self.h_mislabeled + email, 'r')
            lines = current.readlines()
            print lines
            current.close()

            new_lines = []
            for line in lines:
                if self.feature not in line:
                    new_lines.append(line)
                else:
                    continue

            print new_lines

            current = open(self.h_mislabeled + email, 'w')
            for line in new_lines:
                current.write(line)
            current.close()

        for email in listdir(self.s_mislabeled):
            print "Clearing pollution from " + email
            current = open(self.s_mislabeled + email, 'r')
            lines = current.readlines()
            print lines
            current.close()

            new_lines = []
            for line in lines:
                if self.feature not in line:
                    new_lines.append(line)
                else:
                    continue

            print new_lines

            current = open(self.s_mislabeled + email, 'w')
            for line in new_lines:
                current.write(line)
            current.close()

    def injectfeatures(self):
        self.mislabeler.random_move_file()

        for email in listdir(self.h_mislabeled):
            current = open(self.h_mislabeled + email, 'a')
            current.write("\n" + self.feature)
            current.close()
        for email in listdir(self.s_mislabeled):
            current = open(self.s_mislabeled + email, 'a')
            current.write("\n" + self.feature)
            current.close()

def main():
    IP = InjectionPolluter(6000)

    IP.reset()


if __name__ == "__main__":
    main()
