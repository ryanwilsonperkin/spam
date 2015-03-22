from __future__ import division
import re
from collections import defaultdict
from math import log, exp

class NotTrained(Exception): pass

class Model(object):
    """A model trained on spam and ham, used for classifying other text."""

    delimiters = [' ', '\t', '\n', '"', '.', ',', ';', ':', '/', '?', '!', '&',
                  '[', ']', '{', '}', '(', ')', '<', '>']

    def __init__(self, epsilon=1e-16, spam_bonus=1, ham_bonus=2):
        self.__spam = defaultdict(int)
        self.__ham = defaultdict(int)
        self.__n_spam = 0
        self.__n_ham = 0
        self.spam_bonus = spam_bonus
        self.ham_bonus = ham_bonus
        self.min_spamicity = epsilon
        self.max_spamicity = 1 - epsilon

    def get_words(self, text):
        """Split message into words using class delimiters."""
        pattern = '|'.join(map(re.escape, self.delimiters))
        return re.split(pattern, text)

    def train_spam(self, text):
        """Train the model using a spam message."""
        self.__n_spam += 1
        for word in self.get_words(text):
            self.__spam[word.lower()] += 1

    def train_ham(self, text):
        """Train the model using a non-spam message."""
        self.__n_ham += 1
        for word in self.get_words(text):
            self.__ham[word.lower()] += 1

    def classify_word(self, word):
        """Compute the probability that 'word' indicates a message is spam."""
        if self.__n_ham == 0 or self.__n_spam == 0:
            raise NotTrained()

        spam_count = self.spam_bonus * self.__spam[word.lower()]
        ham_count = self.ham_bonus * self.__ham[word.lower()]

        # Classify as neutral if word is very uncommon
        if spam_count + ham_count < 5:
            return 0.5

        ham_frequency = min(self.max_spamicity, ham_count / self.__n_ham)
        spam_frequency = min(self.max_spamicity, spam_count / self.__n_spam)
        spamicity = spam_frequency / (ham_frequency + spam_frequency)
        return max(self.min_spamicity, min(self.max_spamicity, spamicity))

    def classify_text(self, text, n_samples=15):
        """Compute the probability that a message is spam."""
        if self.__n_ham == 0 or self.__n_spam == 0:
            raise NotTrained()

        # Retrieve the n_samples most interesting words
        # 'Most interesting' measured as distance from neutral 0.5
        spamicities = sorted(
            map(self.classify_word, self.get_words(text)),
            key=lambda x: abs(x-0.5),
            reverse=True
        )[:n_samples]

        # Compute probability that interesting words are not spam
        hamicities = map(lambda x: 1-x, spamicities)

        # Naive bayesian analysis of combined interesting words
        spam_frequency = exp(sum(log(s) for s in spamicities)) or self.min_spamicity
        ham_frequency = exp(sum(log(s) for s in hamicities)) or self.min_spamicity
        spamicity = spam_frequency / (ham_frequency + spam_frequency)
        return max(self.min_spamicity, min(self.max_spamicity, spamicity))
