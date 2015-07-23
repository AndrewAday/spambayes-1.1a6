from os import listdir, remove
import os
import sys

sys.path.insert(-1, os.getcwd())
sys.path.insert(-1, os.path.dirname(os.getcwd()))
from spambayes.Options import get_pathname_option


class DictionaryWriter:
    # A simple class to write dictionaries to Set 3 of the spam directories.
    # Used for unlearning experiments.

    def __init__(self, num_files, dir_num=3, dictionary=True, wordlist=True, words=True, wordsEn=True):
        self.NUMFILES = num_files
        self.destination = get_pathname_option("TestDriver", "spam_directories") % dir_num + "/"
        self.destination_files = listdir(self.destination)
        self.dictionary = dictionary
        self.wordlist = wordlist
        self.words = words
        self.wordsEn = wordsEn

    def reset(self):
        """Deletes all dictionary files"""
        print "Deleting Dictionary Files..."

        for dictionary in listdir(self.destination):
            remove(self.destination + dictionary)

    def write(self):

        self.reset()

        print "Initial # of Files: " + str(len(self.destination_files))

        for i in range(0, self.NUMFILES):
            print "Preparing wordlist.txt #" + str(i + 1)
            file_in = open("wordlist.txt", 'r')
            file_out = open(self.destination + "00000wordlist" + str(i + 1) + ".spam" + ".txt", 'w')

            file_out.write(file_in.read())

            file_in.close()
            file_out.close()

        self.destination_files = listdir(self.destination)
        print "Final # of Files: " + str(len(self.destination_files))


def main():
    dw = DictionaryWriter(100)

    dw.write()

if __name__ == "__main__":
    main()
