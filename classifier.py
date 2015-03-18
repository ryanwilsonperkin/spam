from collections import defaultdict

class Model(object):
    """A model trained on spam and ham, used for classifying other text."""

    def __init__(self):
        self.__spam = defaultdict(int)
        self.__ham = defaultdict(int)

    def get_words(self, text):
        return text.split()

    def train_spam(self, text):
        for word in self.get_words(text):
            self.__spam[word] += 1

    def train_ham(self, text):
        for word in self.get_words(text):
            self.__ham[word] += 1
