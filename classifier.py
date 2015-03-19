from __future__ import division
import re
from collections import defaultdict

class Model(object):
    """A model trained on spam and ham, used for classifying other text."""

    delimiters = [' ', '\t', '\n', '"', '.', ',', ';', ':', '/', '?', '!', '&',
                  '[', ']', '{', '}', '(', ')', '<', '>']
    min_spamicity = 0.01
    max_spamicity = 0.99

    def __init__(self):
        self.__spam = defaultdict(int)
        self.__ham = defaultdict(int)
        self.__n_spam = 0
        self.__n_ham = 0

    def get_words(self, text):
        pattern = '|'.join(map(re.escape, self.delimiters))
        return re.split(pattern, text)

    def train_spam(self, text):
        self.__n_spam += 1
        for word in self.get_words(text):
            self.__spam[word.lower()] += 1

    def train_ham(self, text):
        self.__n_ham += 1
        for word in self.get_words(text):
            self.__ham[word.lower()] += 1

    def classify_word(self, word):
        spam_count = self.__spam[word.lower()]
        ham_count = 2 * self.__ham[word.lower()]

        if spam_count + ham_count < 5:
            return 0.5

        ham_frequency = min(1, ham_count / self.__n_ham)
        spam_frequency = min(1, spam_count / self.__n_spam)
        spamicity = spam_frequency / (ham_frequency + spam_frequency)
        return max(self.min_spamicity, min(self.max_spamicity, spamicity))
