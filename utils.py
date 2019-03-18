import codecs
import collections
import os
import nltk
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from operator import itemgetter
import json

import io

# decoder 需要两种格式的目标句子
# decoder inputs : <START> X Y Z
# decoder labels : X Y Z <STOP>

PAD = "<PAD>"
UNK = "<UNK>"
START = "<START>"
STOP = "<STOP>"


def batch_data(source, target, text, batch_size, pad_code):
    for batch_i in range(0, len(source)//batch_size):
        start_i = batch_i * batch_size
        source_batch = source[start_i:start_i + batch_size]
        target_batch = target[start_i:start_i + batch_size]
        text_batch = text[start_i:start_i + batch_size]

        source_lengths = [len(x) for x in source_batch]
        target_lengths = [len(x) for x in target_batch]
        text_lengths = [len(x) for x in text_batch]

        source_batch = np.array(pad_sentence_batch(source_batch, pad_code))
        target_batch = np.array(pad_sentence_batch(target_batch, pad_code))
        text_batch = np.array(pad_sentence_batch(text_batch, pad_code))

        yield (source_batch, target_batch, text_batch, source_lengths, target_lengths, text_lengths)


def pad_sentence_batch(sentence_batch, pad_code):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_code] * (max_sentence - len(sentence))
            for sentence in sentence_batch]


def pad(seq, max_size, pad_char):
    return seq + [pad_char] * (max_size - len(seq))


def parse_input(text):
    tok = [z for z in text.split(' ')]
    return [z for z in tok]
    # return [z.lower() for z in tok.tokenize(text)]


def preprocess(pairs):
    xs = []
    ys = []
    zs = []
    vocab_x = set()
    vocab_y = set()
    vocab_z = set()

    counts = {}

    for (x, y, z) in pairs:
        x_words = parse_input(x)
        y_words = parse_input(y)
        z_words = parse_input(z)

        for w in x_words:
            counts[w] = counts.get(w, 0) + 1

        for w in y_words:
            counts[w] = counts.get(w, 0) + 1

        for w in z_words:
            counts[w] = counts.get(w, 0) + 1

        xs.append(x_words)
        ys.append(y_words)
        zs.append(z_words)

        vocab_x.update(x_words)
        vocab_y.update(y_words)
        vocab_z.update(z_words)

    return xs, ys, zs, vocab_x, vocab_y, vocab_z, counts


def words2ids(text, vocab_to_int, eos=None):
    r = [vocab_to_int.get(x, vocab_to_int[UNK]) for x in text]

    if eos is not None:
        r.append(eos)

    return r


def load_pairs(xpath, ypath, zpath):
    fx = open(xpath)
    fy = open(ypath)
    fz = open(zpath)

    data_x = fx.read().rstrip().split("\n")
    data_y = fy.read().rstrip().split("\n")
    data_z = fz.read().rstrip().split("\n")

    fx.close()
    fy.close()
    fz.close()

    dx = []
    dy = []
    dz = []

    for i in range(len(data_x)):
        if len(data_y[i].split(' ')) < 30 and len(data_x[i].split(' ')) < 200:
            dx.append(data_x[i])
            dy.append(data_y[i])
            dz.append(data_z[i])

    assert(len(dx) == len(dy) and len(dy) == len(dz))

    return dx, dy, dz


def create_lookup_tables(vocab, counts, cutoff_size=None, size_limit=None):
    """
    Create lookup tables for vocabulary
    :param vocab: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """

    vocab_to_int = {}
    int_to_vocab = {}

    _sorted = sorted(vocab, reverse=True, key=lambda x: counts[x])

    for i, word in enumerate([PAD, UNK, START, STOP] + _sorted):
        if size_limit is not None and i > size_limit:
            break

        vocab_to_int[word] = i
        int_to_vocab[i] = word

        if cutoff_size is not None and i > 3 and counts[word] < cutoff_size:
            break

    return vocab_to_int, int_to_vocab


def save_vocab(path, params):
    with open(path, 'w') as f:
        json.dump(params, f)


def load_vocab(path):
    with open(path,'r') as f:
        return json.load(f)


# draw heatmap
def plot_attention_matrix(src, trg, matrix, name="attention_matrix.png"):
    src = [str(item) for item in src.split(' ')]
    trg = [str(item) for item in trg.split(' ')]
    df = pd.DataFrame(matrix, index=trg, columns=src)
    ax = sns.heatmap(df, linewidths=.5)
    ax.set_xlabel("target")
    ax.set_ylabel("source")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.set_title("Attention heatmap")
    plt.savefig(name, bbox_inches='tight')

    canvas = ax.figure.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()

    plt.gcf().clear()

