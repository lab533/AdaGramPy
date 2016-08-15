import numpy as np
from heapq import heappop, heappush


class Dictionary():
    def __init__(self, id2word):
        self.id2word = id2word
        self.word2id = dict([(word, id) for id, word in enumerate(id2word)])

class VectorModel():
    def __init__(self, frequencies, code, path, In, Out, alpha, d, counts):
        self.frequencies = frequencies
        self.code = code
        self.path = path
        self.In = In  # shape  = (#dims, #menanings, #words)
        self.Out = Out
        self.alpha = alpha
        self.d = d
        self.counts = counts


def load_model(path):
    with open(path, "rb") as file:
        V, M, T = map(int, file.readline().split())
        alpha, d = map(np.float64, file.readline().split())
        max_length = int(file.readline())

        frequencies = np.fromfile(file, dtype=np.int64, count=V)

        code = np.fromfile(file, dtype=np.int8, count=max_length * V)
        code = code.reshape((V, max_length)).T

        path = np.fromfile(file, dtype=np.int32, count=max_length * V)
        path = path.reshape((V, max_length)).T

        counts = np.fromfile(file, dtype=np.float32, count=T * V)
        counts = counts.reshape((V, T)).T

        out = np.fromfile(file, dtype=np.float32, count=V * M)
        out = out.reshape(V, M).T

        inp = np.zeros((M, T, V), dtype=np.float32)
        id2word = []

        for v in range(V):
            word = file.readline()
            id2word.append(word.decode("utf8").strip())
            nsenses = int(file.readline())
            for i in range(nsenses):
                k = int(file.readline())
                k -= 1  # becose in julia indexes from 1
                inp[:, k, v] = np.fromfile(file, dtype=np.float32, count=M)
                file.readline()
        vm = VectorModel(frequencies, code, path, inp, out, alpha, d, counts)
    return vm, Dictionary(id2word)


def mean_beta(a, b):
    return a / (a + b)


def expected_pi(vm, word_id, min_prob=1e-3):
    nb_dims, nb_meanings, nb_words = vm.In.shape
    pi = np.zeros(nb_meanings)
    r = 1.
    senses = 0
    ts = vm.counts[:, word_id].sum()
    for k in range(nb_meanings - 1):
        ts = max(ts - vm.counts[k, word_id], 0.)
        a, b = 1. + vm.counts[k, word_id] - vm.d, vm.alpha + k * vm.d + ts
        pi[k] = mean_beta(a, b) * r
        if pi[k] >= min_prob:
            senses += 1
        r = max(r - pi[k], 0.)

    pi[-1] = r
    if r >= min_prob:
        senses += 1
    return senses, pi


def nearest_neighbors(vm, dictionary, word, meaning, top=10, min_count=1.0):
    # meaning -= 1
    word_id = dictionary.word2id[word]
    vec = vm.In[:, meaning, word_id]
    vec /= np.linalg.norm(vec)
    nb_words = vm.counts.shape[1]

    mask = vm.counts > min_count
    mask[meaning, word_id] = False
    # print(mask[1, vocab.word2id["лукашенко"]])
    linear_mask = mask.reshape((-1))
    sims = np.zeros(linear_mask.shape, dtype=np.float32)
    others = vm.In.reshape((vm.In.shape[0], -1))[:, linear_mask]
    others = others / np.linalg.norm(others, axis=0)  # надо переписать
    sims[linear_mask] = np.dot(vec, others)
    sims[linear_mask == False] = - 1
    heap = []
    for ind, sim in enumerate(sims):
        item = (sim, dictionary.id2word[ind % nb_words], ind // nb_words)
        if len(heap) < top:
            heappush(heap, item)
        elif heap[0] < item:
            heappop(heap)
            heappush(heap, item)
    return sorted(heap, reverse=True)


def disambiguate(vm, dictionary, term, context, use_prior=True, min_prob=1e-3):
    z = np.zeros(vm.In.shape[1], dtype=np.float32)
    context = [dictionary.word2id[c] for c in context]
    word = dictionary.word2id[term]
    if use_prior:
        nsens, z = expected_pi(vm, word)
        z[z < min_prob] = 0
        z = np.log(z)
    for c_word in context:
        z += var_update(vm, word, c_word)

    ninf = np.isinf(z) == False
    z[ninf] = np.exp(z[ninf] - np.max(z[ninf]))
    z[ninf] = z[ninf] / np.sum(z[ninf])
    z[ninf == False] = 0
    return z


def var_update(vm, word_id, cword_id):
    path_mask = vm.code[:, cword_id] >= 0
    path = vm.path[path_mask, cword_id] - 1
    code = 1 - 2 * vm.code[path_mask, cword_id]
    z = np.dot(vm.In[:, :, word_id].T, vm.Out[:, path]) * code
    z = np.sum(logsigmoid(z), axis=1)
    return z


def logsigmoid(x):
    return -np.log(np.exp(-x) + 1)




def build_vector_model(fin, min_count=1):
    pass


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def log_sigmoid(x):
    return -np.log(1. + np.exp(-x))


# word2id = {}
# id2word = {}




