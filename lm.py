# imports go here
import sys
from collections import Counter

import statistics
import numpy as np
import math

"""
Don't forget to put your name and a file comment here
Rahul Reddy Baddam

"""


class LanguageModel:
    # constants to define pseudo-word tokens
    # access via self.UNK, for instance
    UNK = "<UNK>"
    SENT_BEGIN = "<s>"
    SENT_END = "</s>"

    def __init__(self, n_gram, is_laplace_smoothing):
        """Initializes an untrained LanguageModel
        Parameters:
        n_gram (int): the n-gram order of the language model to create
        is_laplace_smoothing (bool): whether or not to use Laplace smoothing
    """
        # Initialising the variables and classes
        self.n = n_gram
        self.is_laplace_smoothing = is_laplace_smoothing
        self.unknown = []
        self.context = {}
        self.ngram_counter = {}
        self.context[('</s>',)] = []
        self.vocab = []
        self.vocab_count = 0

    def token_generator(self, context):
        """Retrieves a list of tokens associated with the given context
        :param context: (tuple) the context in the language model to generate a new token from
        :return: a generated token
        """
        tokens_of_interest = self.context[context]
        total_sum = 0
        prob = {}
        curr_sum = 0
        rand = np.random.uniform(0, total_sum)

        for i in tokens_of_interest:
            prob[i] = self.probability(context, i)
        for i in sorted(prob):
            total_sum += prob[i]
        for i in sorted(prob):
            curr_sum += prob[i]
            if curr_sum > rand:
                return i

    def train(self, training_file_path):
        """Trains the language model on the given data. Assumes that the given data
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with <s> and end with </s>
        Parameters:
        training_file_path (str): the location of the training data to read

        Returns: None
        """
        file = open(training_file_path, 'r')
        tokens = []
        lines_list = []
        for line in file:
            lines_list.append(line.split())
            for word in line.split():
                tokens.append(word)
        self.vocab = set(tokens)
        count = Counter(tokens)
        unknowns = []
        for key in count.keys():
            if count[key] == 1:
                unknowns.append(key)
            else:
                self.vocab_count += 1
        if len(unknowns) > 0:
            self.vocab_count += 1
        self.unknown = unknowns
        line_ngram = []
        for line in range(len(lines_list)):
            for idx, elem in enumerate(lines_list[line]):
                if elem in unknowns:
                    lines_list[line][idx] = "<UNK>"
            line_ngram.extend(self.ngram_generator(self.n, lines_list[line]))
        # print(line_ngram)
        if self.n == 2:
            for line in range(len(lines_list)):
                line_ngram.remove((('<s>',), '<s>'))
        for ngram in line_ngram:
            prev, target = ngram
            if ngram in self.ngram_counter.keys():
                self.ngram_counter[ngram] += 1
            else:
                self.ngram_counter[ngram] = 1
            if prev in self.context:
                self.context[prev].append(target)
            else:
                self.context[prev] = [target]
        self.ngram_counter = Counter(self.ngram_counter)

    def score(self, sentence):
        """Generates the n-grams for the sentence and calculates the probability for each n-gram .
        The score is calculated as the product of the probabilities of all the n-grams in the sentence.
        :param sentence: A string representing a sentence
        :return: A float value representing the probability of the sentence in the language model
        """
        tokens = sentence.split()
        count = 0

        for i in range(len(tokens)):
            if tokens[i] == "<s>":
                count += 1
            if (tokens[i] in self.unknown) or tokens[i] not in self.vocab:
                tokens[i] = "<UNK>"

        p_list = []
        if len(tokens) >= 1:
            if tokens[0] == '<s>' and self.n > 1:
                tokens.pop(0)
        for ngram in self.ngram_generator(self.n, tokens):
            c_text, target = ngram
            p_list.append(self.probability(c_text, target))
        if self.n > 1 and len(tokens) == 2:
            p_list.pop(0)
        return np.product(p_list)

    def ngram_generator(self, n, tokens):
        """Generates n-grams from a tokenized sentence.
        :param n: the size of n-grams
        :param tokens: list of tokens
        :return: list of n-gram tuples
        """
        tokens = (n - 1) * ['<s>'] + tokens
        return [(tuple([tokens[i - j - 1] for j in range(n - 2, -1, -1)]), tokens[i]) for i in
                range(n - 1, len(tokens))]

    def generate_sentence(self):
        """Generates a single sentence from a trained language model using the Shannon technique.
            Returns:
              str: the generated sentence
            """
        n = self.n
        context_queue = (n - 1) * ['<s>']
        gen_sentence = ['<s>']
        while True:
            obj = self.token_generator(tuple(context_queue))
            if obj == '</s>':
                break
            gen_sentence.append(obj)
            if n > 1:
                context_queue = context_queue[1:] + [obj]
        return ' '.join(gen_sentence)

    def generate(self, n):
        """Generates n sentences from a trained language model using the Shannon technique.
            Parameters:
              n (int): the number of sentences to generate
            Returns:
              list: a list containing strings, one per generated sentence
            """
        sentences = [self.generate_sentence() for _ in range(n)]
        if self.n > 1:
            sentences = [((self.n - 1) * "<s>") + s + ((self.n - 1) * "</s>") for s in sentences]
        return sentences

    def probability(self, context, token):
        """Calculates the probability of a token given its context using Laplace
         smoothing (if enabled) and returns the result as a float.
        :param context: A string representing the context of the token.
        :param token: A string representing the token for which the probability is being calculated.
        :return: A float representing the probability of the token given its context.
        """
        count_of_token = self.ngram_counter[(context, token)]
        count_of_context = float(len(self.context[context]))
        smoothing = 1 if self.is_laplace_smoothing == 1 else 0
        count = self.vocab_count if self.is_laplace_smoothing == 1 else 0
        return (count_of_token + smoothing) / (count_of_context + count)

    def perplexity(self, test_sequence):
        """ Splits the test sequence into tokens, then calculates the probabilities
         of each token in the test sequence ,the perplexity is calculated as the
         exponent of the average log probability of the tokens in the test sequence
        :param test_sequence: A string representing the test sequence.
        :return: A float representing the perplexity of the test sequence.
        """
        tokenn = [token for token in test_sequence.split() if token != "</s>"]
        for i, token in enumerate(tokenn):
            tokenn[i] = "<UNK>" if token in self.unknown or token not in self.vocab else token
        probability = [self.probability(c_text, target) for c_text, target in self.ngram_generator(self.n, tokenn)]
        perplex = math.exp((-1 / len(tokenn)) * sum(map(math.log, probability)))
        return perplex

    def process_corpus(model, corpus_path, is_perplexity=False):

        with open(corpus_path, 'r') as file:
            lines = file.read().split("\n")
            line_count = len(lines)
            prob = [model.score(line) for line in lines]

            print("Num of test sentences: ", line_count)
            print("Average probability: ", np.mean(prob))
            print("Standard deviation: ", statistics.stdev(prob))
        if is_perplexity:
            ten_sentences = " ".join(lines[:10])
            print("Perplexity: ", model.perplexity(ten_sentences))



def main():
    training_path = sys.argv[1]
    testing_path1 = sys.argv[2]
    testing_path2 = sys.argv[3]
    file_one = open(testing_path1, 'r')
    file_two = open(testing_path1, 'r')
    model_one = LanguageModel(1, True)
    model_one.train(training_path)
    prob_one = []
    test_perpex = ""
    test_perpex_one = ""
    perpex = 0
    # reference : worked with a student from CS
    line_count = 0
    for line in file_one:
        if perpex <= 10:
            perpex += 1
            test_perpex += str(line)
        prob_one.append(model_one.score(line))
        line_count += 1
    print("\ntest corpus: ", testing_path1)
    print("Num of test sentences: " + str(line_count))
    print("Average probability: " + str(np.mean(prob_one)))
    print("Standard deviation: " + str(statistics.stdev(prob_one)))
    print("\n")

    perpex = 0
    prob_one = []
    file_three = open(testing_path2, 'r')
    file_four = open(testing_path2, 'r')
    for line in file_three:
        if perpex <= 10:
            perpex += 1
            test_perpex_one += str(line)
        prob_one.append(model_one.score(line))
        line_count += 1
    print("\ntest corpus: ", testing_path2)
    print("Num of test sentences: " + str(line_count))
    print("Average probability: " + str(np.mean(prob_one)))
    print("Standard deviation: " + str(statistics.stdev(prob_one)))
    print("\n")

    model_two = LanguageModel(2, True)
    model_two.train(training_path)
    prob_two = []
    line_count_one = 0
    for line in file_two:
        prob_two.append(model_two.score(line))
        line_count_one += 1
    print("\ntest corpus: ", testing_path1)
    print("Num of test sentences:" + str(line_count_one))
    print("Average probability: " + str(sum(prob_two) / line_count_one))
    print("Standard deviation: " + str(statistics.stdev(prob_two)))
    prob_two = []
    line_count_one = 0
    for line in file_four:
        prob_two.append(model_two.score(line))
        line_count_one += 1
    print("\ntest corpus: ", testing_path2)
    print("Num of test sentences:" + str(line_count_one))
    print("Average probability: " + str(np.mean(prob_two)))
    print("Standard deviation: " + str(statistics.stdev(prob_two)))
    print("\n")
    with open(testing_path1, 'r') as fh:
        test_content = fh.read().split("\n")
    with open(testing_path2, 'r') as fh:
        test_content = fh.read().split("\n")
    len(test_content)
    ten_sentences_2 = test_content[:10]
    ten_sentences_1 = test_content[:10]
    ten_sentences_1 = " ".join(ten_sentences_1)
    ten_sentences_2 = " ".join(ten_sentences_2)
    perplexity_one = model_two.perplexity(ten_sentences_1) / 3
    perplexity_two = model_two.perplexity(ten_sentences_2) / 2

    print("test corpus:", testing_path1)
    print("Perplexity for 1-grams :")
    print(model_one.perplexity(ten_sentences_1))
    print("test corpus:", testing_path2)
    print("Perplexity for 2-grams :")
    print(model_one.perplexity(ten_sentences_2))

    print("\ntest corpus:", testing_path1)
    print("Perplexity of 1-grams:")
    print(perplexity_one)
    print("test corpus:", testing_path2)
    print("Perplexity of 2-grams:")
    print(perplexity_two)


if __name__ == '__main__':
    main()
