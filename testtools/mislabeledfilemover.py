import os
import sys

sys.path.insert(-1, os.getcwd())
sys.path.insert(-1, os.path.dirname(os.getcwd()))

from os import listdir
from random import choice
from shutil import move
from spambayes.Options import get_pathname_option


class MislabeledFileMover:
    # A class that moves a given number of randomly selected files
    # from Set1 to Set3 of both the spam and ham directories. The
    # class randomly divides the given number between ham and spam.

    def __init__(self, number, train_dir=1, test_dir=2, dest_dir=3):
        self.NUMBER = number
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.dest_dir = dest_dir

        self.ham_num = self.NUMBER
        self.ham_train = get_pathname_option("TestDriver", "ham_directories") % train_dir + "/"
        self.ham_test = get_pathname_option("TestDriver", "ham_directories") % test_dir + "/"
        self.ham_destination = get_pathname_option("TestDriver", "spam_directories") % dest_dir + "/"
        self.ham_source_files = listdir(self.ham_train)
        self.ham_destination_files = listdir(self.ham_destination)

        self.spam_num = self.NUMBER
        self.spam_train = get_pathname_option("TestDriver", "spam_directories") % train_dir + "/"
        self.spam_test = get_pathname_option("TestDriver", "spam_directories") % test_dir + "/"
        self.spam_destination = get_pathname_option("TestDriver", "ham_directories") % dest_dir + "/"
        self.spam_source_files = listdir(self.spam_train)
        self.spam_destination_files = listdir(self.spam_destination)

    def reset(self):
        """Returns all files in Set3 of both spam and ham to their respective Set1"""
        print "Replacing Files..."

        for ham in self.ham_destination_files:
            print " - \tReturning " + ham + " from Spam Set" + str(self.dest_dir) + " to Ham Set" + str(self.train_dir)
            move(self.ham_destination + ham, self.ham_train + ham)

        for spam in self.spam_destination_files:
            print " - \tReturning " + spam + " from Ham Set" + str(self.dest_dir) + " to Spam Set" + str(self.train_dir)
            move(self.spam_destination + spam, self.spam_train + spam)

    def print_filelist(self):
        """Prints the number of files in all Sets, for both Spam and Ham"""
        dir_list = [(self.train_dir, "train"), (self.test_dir, "test"), (self.dest_dir, "destination")]
        dir_list.sort()

        print "File List:"
        print " - \tFiles in Ham Set1: " + str(listdir(self.attr_name(0, dir_list[0][1])).__len__())
        print " - \tFiles in Ham Set2: " + str(listdir(self.attr_name(0, dir_list[1][1])).__len__())
        print " - \tFiles in Ham Set3: " + str(listdir(self.attr_name(0, dir_list[2][1])).__len__())
        print " - \tFiles in Spam Set1: " + str(listdir(self.attr_name(1, dir_list[0][1])).__len__())
        print " - \tFiles in Spam Set2: " + str(listdir(self.attr_name(1, dir_list[1][1])).__len__())
        print " - \tFiles in Spam Set3: " + str(listdir(self.attr_name(1, dir_list[2][1])).__len__())

    def attr_name(self, ham_or_spam, dir_type):
        # Ham
        if ham_or_spam == 0:
            attribute_name = "ham_" + dir_type
            return getattr(self, attribute_name)

        # Spam
        elif ham_or_spam == 1:
            attribute_name = "spam_" + dir_type
            return getattr(self, attribute_name)

        else:
            raise AssertionError

    def random_move_file(self):
        """Moves a number of random files from Set1 of both spam and ham to Set3"""

        # move 'number' files to the  ddestination
        print "Number of Ham Files to Move: " + str(self.ham_num)
        for i in range(self.ham_num):
            ham = choice(self.ham_source_files)

            self.ham_source_files.remove(ham)

            print "(" + str(i + 1) + ")" + "\tMoving file " + ham

            if ham in self.ham_destination_files:
                i -= 1
                continue
            else:
                move(self.ham_train + ham, self.ham_destination + ham)

        print "Number of Spam Files to Move: " + str(self.spam_num)
        for i in range(self.spam_num):
            spam = choice(self.spam_source_files)

            self.spam_source_files.remove(spam)

            print "(" + str(i + 1) + ")" + "\tMoving file " + spam

            if spam in self.spam_destination_files:
                i -= 1
                continue
            else:
                move(self.spam_train + spam, self.spam_destination + spam)

        self.print_filelist()


def main():
    f = MislabeledFileMover(3)
    f.print_filelist()

if __name__ == "__main__":
    main()
