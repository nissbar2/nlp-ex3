# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import nltk

nltk.download("brown")
import math
import operator
import pprint
import re
from collections import Counter, OrderedDict, defaultdict
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import brown


def create_confusion_matrix(true_tags, predicted_tags):
    """
    Create a confusion matrix from true and predicted tags.
    Args:
        true_tags: List of actual/true tags
        predicted_tags: List of predicted tags
    Returns:
        numpy.ndarray: Confusion matrix where entry (i,j) represents
        the count of tokens with true tag i and predicted tag j
    """
    # Get unique tags to determine matrix size
    unique_tags = sorted(list(set(true_tags + predicted_tags)))
    n_tags = len(unique_tags)

    # Create tag to index mapping
    tag_to_idx = {tag: idx for idx, tag in enumerate(unique_tags)}

    # Initialize confusion matrix
    confusion_matrix = np.zeros((n_tags, n_tags), dtype=np.int32)

    # Fill confusion matrix
    for true, pred in zip(true_tags, predicted_tags):
        i = tag_to_idx[true]
        j = tag_to_idx[pred]
        confusion_matrix[i, j] += 1

    return confusion_matrix


def plot_confusion_matrix(confusion_matrix):
    """
    Visualize confusion matrix using pyplot heatmap.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix)
    # Customize plot
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Tags")
    plt.ylabel("True Tags")

    # Rotate tick labels for better readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot if path is provided

    plt.show()


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


REGEXES = OrderedDict(
    [  # type of dict[str, regex]
        ("__twoDigitNum", r"^[0-9]{2}$"),
        ("__fourDigitNum", r"^\d{4}$"),
        ("__numbers", r"^\d+$"),
        ("__digitAndAlpha", r"^(?=.*[A-Za-z])(?=.*\d).+$"),
        ("__allCaps", r"^[A-Z]+$"),
        ("__capitalWord", r"^[A-Z][a-z]*$"),
    ]
)


def apply_regex(word):
    pseudo_word = None
    for regex_word, pattern in REGEXES.items():
        if re.search(pattern, word):
            return regex_word
    return pseudo_word


def preprocess_pseudo_words(sentences, low_freq_words):
    processed = []
    for s in sentences:
        processed_sentence = []
        for word, tag in s:
            if word in low_freq_words:
                pseudo_word = apply_regex(word)
                processed_sentence.append((pseudo_word, tag))
            else:
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
    known_words = list(filter(lambda x: x[0] in baseline.unique_words, predictions))
    unknown_words = list(
        filter(lambda x: x[0] not in baseline.unique_words, predictions)
    )
    known_words_error = error_rate(known_words)
    unknown_words_error = error_rate(unknown_words)
    return total, known_words_error, unknown_words_error


class Emission:
    def __init__(self, smoothing=False):
        self.fitted = False
        self.smoothing = smoothing

    def fit(self, data):
        flat = list(chain.from_iterable(data))
        self.unique_word = set(map(operator.itemgetter(0), flat))
        self.conditional_counter = Counter(flat)
        self.tags_counter = Counter(map(operator.itemgetter(1), flat))
        self._probs = {}
        for word in self.unique_word:
            for tag in self.tags_counter.keys():
                self._probs[(word, tag)] = self._prob(word, tag)

    def _prob(self, word, tag):
        if not self.smoothing:
            return self.conditional_counter[(word, tag)] / self.tags_counter[tag]
        else:
            return (self.conditional_counter[(word, tag)] + 1) / (
                self.tags_counter[tag] + len(self.unique_word)
            )

    def prob(self, word, tag):
        if word not in self.unique_word:
            return 1 / self.tags_counter["NN"]
        return self._probs[(word, tag)]


class Transition:
    def __init__(self):
        self.fitted = False
        self.pairs = None
        self.START = "__START__"
        self.STOP = "__STOP__"

    def fit(self, dataset):
        self.pairs = defaultdict(list)
        # for t1, t2 in self._token_gen(dataset):
        #    self.pairs[t1].append(t2)
        tags = [
            [self.START] + list(map(operator.itemgetter(1), sentence)) + [self.STOP]
            for sentence in dataset
        ]
        # tags = list(map(operator.itemgetter(1), augmented_data))

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
                if t2 == self.STOP:
                    continue
                self._probs[(t1, t2)] = self._prob(t1, t2)

        self.fitted = True

    def prob(self, tag, previous):
        return self._probs[(tag, previous)]

    def _prob(self, tag, previous):
        # TODO: verify the calculated probs!!!!
        c = Counter(self.pairs[previous])
        return c[tag] / len(self.pairs[previous])

    def _token_gen(self, data):
        augmented_data = [[("IGNORED", self.START)] + sentence for sentence in data]
        yield from pairwise(map(operator.itemgetter(1), augmented_data))


class HMM:
    def __init__(self, smoothing=False):
        self.transmission = Transition()
        self.emission = Emission(smoothing)

    def fit(self, data):
        self.transmission.fit(data)
        self.emission.fit(data)

    def predict(self, sentence):
        """
        Run Viterbi Algorithm.
        """
        table = list()  # type of list[dict[tag, tuple[float, backpointer]]]
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
            prob, bp = table[len(sentence)][tag]
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

        result = list(reversed(seq))
        return result


def gen_low_frequency_words(train):
    flat = list(chain.from_iterable(train))
    train_words = Counter(map(operator.itemgetter(0), flat))
    low_freq_words = set()
    for word in train_words.keys():
        if train_words[word] < 5:
            low_freq_words.add(word)
    return low_freq_words


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    train, test = download_dataset()
    baseline = Baseline()
    baseline.fit(train)

    # predictions = []
    # for (word, true_tag) in chain.from_iterable(test):
    #     predictions.append((baseline.predict(word), true_tag))

    # total_error, known_words_error, unknown_words_error = compute_errors(baseline, predictions)
    # print("Task B:")
    # print(f"\ttotal error rate is: {total_error}")
    # print(f"\tknown words error rate is: {known_words_error}")
    # print(f"\tunknown words error rate is: {unknown_words_error}")
    #
    # hmm = HMM()
    # hmm.fit(train)
    # hmm_predictions = []
    #
    # # hmm.predict(test[16])
    #
    # for i, sentence in enumerate(test):
    #     print(f"sentence {i}")
    #     sen = list(map(operator.itemgetter(0), sentence))
    #     sen_tags = list(map(operator.itemgetter(1), sentence))
    #     hmm_predictions.extend(list(zip(hmm.predict(sen), sen_tags)))
    #
    # total_error, known_words_error, unknown_words_error = compute_errors(baseline, hmm_predictions)
    # print("Task C:")
    # print(f"\ttotal error rate is: {total_error}")
    # print(f"\tknown words error rate is: {known_words_error}")
    # print(f"\tunknown words error rate is: {unknown_words_error}")
    #
    # hmm_smoothing = HMM(smoothing=True)
    # hmm_smoothing.fit(train)
    # hmm_smoothing_predictions = []
    #
    # # hmm.predict(test[16])
    #
    # for i, sentence in enumerate(test):
    #     print(f"sentence {i}")
    #     sen = list(map(operator.itemgetter(0), sentence))
    #     sen_tags = list(map(operator.itemgetter(1), sentence))
    #     hmm_smoothing_predictions.extend(list(zip(hmm_smoothing.predict(sen), sen_tags)))
    #
    # total_error, known_words_error, unknown_words_error = compute_errors(baseline, hmm_smoothing_predictions)
    # print("Task D:")
    # print(f"\ttotal error rate is: {total_error}")
    # print(f"\tknown words error rate is: {known_words_error}")
    # print(f"\tunknown words error rate is: {unknown_words_error}")
    low_freq_words = gen_low_frequency_words(train)
    train_pseudo_processed = preprocess_pseudo_words(
        train, low_freq_words=low_freq_words
    )
    test_pseudo_processed = preprocess_pseudo_words(test, low_freq_words=low_freq_words)
    # hmm = HMM()
    # hmm.fit(train_pseudo_processed)
    # hmm_predictions = []
    #
    # # hmm.predict(test[16])
    #
    # for i, sentence in enumerate(test_pseudo_processed):
    #     print(f"sentence {i}")
    #     sen = list(map(operator.itemgetter(0), sentence))
    #     sen_tags = list(map(operator.itemgetter(1), sentence))
    #     hmm_predictions.extend(list(zip(hmm.predict(sen), sen_tags)))
    #
    # total_error, known_words_error, unknown_words_error = compute_errors(baseline, hmm_predictions)
    # print("Task E.2:")
    # print(f"\ttotal error rate is: {total_error}")
    # print(f"\tknown words error rate is: {known_words_error}")
    # print(f"\tunknown words error rate is: {unknown_words_error}")

    hmm_smoothing = HMM(smoothing=True)
    hmm_smoothing.fit(train_pseudo_processed)
    hmm_smoothing_predictions = []

    # hmm.predict(test[16])

    for i, sentence in enumerate(test_pseudo_processed):
        print(f"sentence {i}")
        sen = list(map(operator.itemgetter(0), sentence))
        sen_tags = list(map(operator.itemgetter(1), sentence))
        hmm_smoothing_predictions.extend(
            list(zip(hmm_smoothing.predict(sen), sen_tags))
        )

    total_error, known_words_error, unknown_words_error = compute_errors(
        baseline, hmm_smoothing_predictions
    )
    print("Task E.3:")
    print(f"\ttotal error rate is: {total_error}")
    print(f"\tknown words error rate is: {known_words_error}")
    print(f"\tunknown words error rate is: {unknown_words_error}")
    confusion_mat = create_confusion_matrix(
        list(map(operator.itemgetter(1), hmm_smoothing_predictions)),
        list(map(operator.itemgetter(0), hmm_smoothing_predictions)),
    )
    plot_confusion_matrix(confusion_mat)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
