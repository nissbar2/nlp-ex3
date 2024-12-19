# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import nltk

nltk.download("brown")

import math
import operator
from collections import Counter, defaultdict
from itertools import chain

from nltk.corpus import brown


def pairwise(data):
    iterator = iter(data)
    a = next(iterator, None)
    for b in iterator:
        yield a, b
        a = b


def preprocess(sentences):
    processed = []

    for s in sentences:
        processed_sentence = []
        for word, tag in s:
            if "-" in tag:
                tag = tag.split("-")[0]
            elif "+" in tag:
                tag = tag.split("+")[0]
            processed_sentence.append((word, tag))
        processed.append(processed_sentence)

    return processed


def download_dataset():
    # Use a breakpoint in the code line below to debug your script.
    preprocessed = brown.tagged_sents(categories="news")
    sentences = preprocess(preprocessed)
    train_size = int(0.9 * len(sentences))
    test_size = int(0.1 * len(sentences))

    train = sentences[:train_size]
    test = sentences[train_size:]

    return train, test


class Baseline:
    def __init__(self):
        self.fitted = False

    def fit(self, data):
        flat = list(chain.from_iterable(data))
        self.fitted = True
        self.conditional_counter = Counter(flat)
        self.unique_words = set(map(operator.itemgetter(0), flat))
        self.all_tags = set(map(operator.itemgetter(1), flat))
        self.pred_dict = {}

        for word in self.unique_words:
            self.pred_dict[word] = self._predict(word)

    def _predict(self, word):
        best_tag = None
        best = 0
        for tag in self.all_tags:
            count = self.conditional_counter[(word, tag)]
            if count > best:
                best = count
                best_tag = tag
        return best_tag

    def predict(self, word):
        return self.pred_dict.get(word, "NN")


def error_rate(predictions):
    agrements = 0
    for y_pred, y_true in predictions:
        if y_pred == y_true:
            agrements += 1
    return 1 - (agrements / len(predictions))


def compute_errors(baseline, predictions):
    total = error_rate(predictions)
    known_words_error = error_rate(
        list(filter(lambda x: x[0] in baseline.unique_words, predictions))
    )
    unknown_words_error = error_rate(
        list(filter(lambda x: x[0] not in baseline.unique_words, predictions))
    )
    return total, known_words_error, unknown_words_error


class Emission:
    def __init__(self):
        self.fitted = False

    def fit(self, data):
        flat = list(chain.from_iterable(data))
        self.unique_word = set(map(operator.itemgetter(0), flat))
        self.conditional_counter = Counter(flat)
        self.tags_counter = Counter(map(operator.itemgetter(1), flat))

    def prob(self, word, tag):
        if word not in self.unique_word:
            return 1 / self.tags_counter["NN"]
        return self.conditional_counter[(word, tag)] / self.tags_counter[tag]


class Transition:
    def __init__(self):
        self.fitted = False
        self.pairs = None
        self.START = "START"
        self.START_WORD = "__START__"
        self.STOP = "STOP"
        self.STOP_WORD = "__STOP__"

    def fit(self, dataset):
        self.pairs = defaultdict(list)
        tags = [
            [self.START] + list(map(operator.itemgetter(1), sentence)) + [self.STOP]
            for sentence in dataset
        ]

        flat = list(chain.from_iterable(dataset))
        self.all_tags = set(map(operator.itemgetter(1), flat))
        self.all_tags.add(self.START)
        self.all_tags.add(self.STOP)

        for tags_seq in tags:
            for t1, t2 in pairwise(tags_seq):
                self.pairs[t1].append(t2)

        self._probs = {}
        for t1 in self.all_tags:
            for t2 in self.all_tags:
                self._probs[(t1, t2)] = self._prob(t1, t2)

        self.fitted = True

    def prob(self, tag, previous):
        return self._probs[(tag, previous)]

    def _prob(self, tag, previous):
        c = Counter(self.pairs[previous])
        return c[tag] / len(self.pairs[previous]) if self.pairs[previous] else 0


class HMM:
    def __init__(self):
        self.transmission = Transition()
        self.emission = Emission()

    def fit(self, data):
        self.transmission.fit(data)
        self.emission.fit(data)

    def predict(self, sentence):
        """
        Run Viterbi Algorithm.

        :param sentence:
        :return:
        """
        table = list()  # type of list[dict[tag, tuple[probability, back-pointer]]]
        table.append(defaultdict(lambda: (1, None)))  # pi(0, *) = 1, bp(0, *) = None
        S_0 = [self.transmission.START]
        S = self.emission.tags_counter.keys()

        for k in range(1, 1 + len(sentence)):
            S_k = S
            S_k_1 = S if k != 1 else S_0
            table.append({})
            for current in S_k:
                max_prob = -math.inf
                bp = None
                for prev in S_k_1:
                    prob = (
                        table[k - 1][prev][0]
                        * self.transmission.prob(tag=current, previous=prev)
                        * self.emission.prob(sentence[k - 1], current)
                    )
                    if prob > max_prob:
                        max_prob = prob
                        bp = prev

                table[k][current] = (max_prob, bp)

        # extract solution
        max_prob = -math.inf
        max_bp = None
        seq = []
        for tag in S:
            prob, bp = table[-1][tag]
            prob = prob * self.transmission.prob(self.transmission.STOP, previous=tag)
            if prob > max_prob:
                max_prob = prob
                max_bp = tag

        bp = max_bp
        for k in reversed(range(1, 1 + len(sentence))):
            seq.append(bp)
            if bp is None:
                print()
            _, bp = table[k][bp]

        result = list(reversed(seq))[1:]
        return result


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    train, test = download_dataset()
    baseline = Baseline()
    baseline.fit(train)

    predictions = []
    for word, true_tag in chain.from_iterable(test):
        predictions.append((baseline.predict(word), true_tag))

    total_error, known_words_error, unknown_words_error = compute_errors(
        baseline, predictions
    )
    print("Task B:")
    print(f"\ttotal error rate is: {total_error}")
    print(f"\tknown words error rate is: {known_words_error}")
    print(f"\tunknown words error rate is: {unknown_words_error}")

    hmm = HMM()
    hmm.fit(train)
    hmm_predictions = []

    # hmm.predict(test[16])

    for i, sentence in enumerate(test):
        print(f"sentence {i}")
        sen = list(map(operator.itemgetter(0), sentence))
        sen_tags = list(map(operator.itemgetter(1), sentence))
        hmm_predictions.extend(list(zip(hmm.predict(sen), sen_tags)))

    total_error, known_words_error, unknown_words_error = compute_errors(
        baseline, hmm_predictions
    )
    print("Task C:")
    print(f"\ttotal error rate is: {total_error}")
    print(f"\tknown words error rate is: {known_words_error}")
    print(f"\tunknown words error rate is: {unknown_words_error}")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
