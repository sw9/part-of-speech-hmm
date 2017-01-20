# Include your imports here, if any are used.

import collections

TAGS = ('NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM', 'CONJ',
        'PRT', '.', 'X')


def load_corpus(path):
    ret = []
    with open(path) as f:
        for line in f:
            sentence = []

            for string in line.split():
                sentence.append(tuple(string.split("=")))

            ret.append(sentence)
    return ret


class Tagger(object):

    def __init__(self, sentences):
        pi_smooth = 1e-10
        a_smooth = 1e-10
        b_smooth = 1e-10
        self.pi = {}
        self.a = {}
        self.b = {}
        for t in TAGS:
            self.pi[t] = 0
            self.a[t] = {}
            for t2 in TAGS:
                self.a[t][t2] = 0
            self.b[t] = collections.defaultdict(int)

        for sentence in sentences:
            if sentence:
                (_, tag) = sentence[0]
                self.pi[tag] += 1

            for i in xrange(1, len(sentence)):
                (_, tag) = sentence[i]
                (_, prev_tag) = sentence[i - 1]
                self.a[prev_tag][tag] += 1

            for (token, tag) in sentence:
                self.b[tag][token] += 1

        total = 0.0
        for t in TAGS:
            total += self.pi[t] + pi_smooth
        for t in TAGS:
            self.pi[t] = float(self.pi[t] + pi_smooth)/total

        for t in TAGS:
            total = 0.0
            for t2 in TAGS:
                total += self.a[t][t2] + a_smooth
            for t2 in TAGS:
                self.a[t][t2] = float(self.a[t][t2] + a_smooth)/total

        for t in TAGS:
            total = b_smooth
            for token in self.b[t]:
                total += self.b[t][token]
            for token in self.b[t]:
                self.b[t][token] = float(self.b[t][token] + b_smooth)/total
            self.b[t]["<UNK>"] = b_smooth/total

    def most_probable_tags(self, tokens):
        ret = []
        for token in tokens:
            max_val = -1
            max_tag = ""
            for t in TAGS:
                if token in self.b[t]:
                    val = self.b[t][token]
                else:
                    val = self.b[t]["<UNK>"]

                if val > max_val:
                    max_val = val
                    max_tag = t

            ret.append(max_tag)
        return ret

    def viterbi_tags(self, tokens):
        delta = [[0.0 for _ in TAGS] for _ in tokens]
        back = [[0 for _ in TAGS] for _ in tokens]

        if not tokens:
            return []

        for i in xrange(len(TAGS)):
            t = TAGS[i]
            b = self.b[t][tokens[0]] if tokens[0] in self.b[t] \
                else self.b[t]["<UNK>"]
            delta[0][i] = self.pi[t]*b

        for t in xrange(1, len(tokens)):
            for j in xrange(len(TAGS)):
                max_i = 0
                max_val = -1
                for i in xrange(len(TAGS)):
                    tag_i = TAGS[i]
                    tag_j = TAGS[j]
                    val = delta[t-1][i]*self.a[tag_i][tag_j]

                    if val > max_val:
                        max_val = val
                        max_i = i
                back[t][j] = max_i
                tag_j = TAGS[j]
                b = self.b[tag_j][tokens[t]] if tokens[t] in self.b[tag_j] \
                    else self.b[tag_j]["<UNK>"]
                delta[t][j] = max_val*b

        max_i = 0
        max_val = -1
        ret = []
        for i in xrange(len(TAGS)):
            if delta[-1][i] > max_val:
                max_val = delta[-1][i]
                max_i = i
        ret.append(TAGS[max_i])
        last = max_i
        for t in xrange(len(tokens) - 2, -1, -1):
            last = back[t+1][last]
            ret.append(TAGS[last])

        return list(reversed(ret))
