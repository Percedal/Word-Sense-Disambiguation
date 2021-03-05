import math
import pickle
import random
import numpy as np
from queue import Queue
from threading import Thread
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer, cosine_distance

__all__ = ["Huang"]


class _ValuedThread(Thread):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value = None

    def run(self):
        try:
            if self._target:
                self.value = self._target(*self._args, **self._kwargs)
        finally:
            del self._target, self._args, self._kwargs


class Huang:
    """Implementation of the huang et al. algorithm for Word Sens Disambiguation (WSD).
    
    Parameters
    ----------
    ctx_window: int
        Contextual window size. A context define the words that appears before and after a polysemous.
        This context is used to disambiguate words with multiple senses.
        When a context is compute, it is center on a word and this parameter define the total length of all contexts.
        Meaning for a context size of 11, 5 words will taken before and 5 will taken after the center word.
        Those 11 words compose the context.
    seed: int
        Random generator seed.

    See Also
    --------
    'Improving Word Representations via Global Context and Multiple Word Prototypes' huang et al., 2012
    """

    def __init__(self,
                 vector_size=100,
                 ctx_window=5,
                 min_count=1,
                 seed=random.randrange(1e6),
                 nb_worker=1,
                 verbose=False):
        """
        Parameters
        ----------
        vector_size: int
            Size of the embeddings.
        ctx_window: int
            Size of the window to compute context vectors.
            Max word distance to compute context vectors :
            ctx_window = 5 means 5 words before and after a polysemous word will be taken to compute a context
        seed: int
            Seed for the random number generator.
        """
        self._vector_size = vector_size if vector_size > 0 else 100
        self.ctx_window = (ctx_window if ctx_window > 0 else 5) * 2 + 1
        self.min_count = min_count if min_count > 1 else 1
        self.seed = int(seed)
        self.nb_worker = nb_worker if nb_worker > 1 else 1
        self.verbose = verbose
        self._wv = {}
        self._wv_polys = {}
        self._w2v = None

    def __call__(self, sentences, senses_dict):
        """Call to the function ``self.huang()``
        
        See Also
        --------
        Huang.huang()
        """
        self.huang(sentences, senses_dict)

    def _cluster_lemma(self, sentences, positions, nb_senses):
        """TODO"""
        mean_vectors = []
        for i_s, i_w in positions:
            mean_vectors.append(
                _mean_vector(
                    self._w2v.wv,
                    self.__extract_window(
                        sentences[i_s],
                        i_w
                    )
                )
            )
        skm = KMeansClusterer(
            nb_senses,
            cosine_distance,
            rng=random.Random(self.seed),
            repeats=5,
            avoid_empty_clusters=True
        )

        return skm.cluster(mean_vectors, True)

    def __compute_context_vectors(self, sentences, positions, nb_sense):
        """Compute the context vectors of a lemma given its positions in the corpus.
        TODO"""
        mean_vectors = []

        for i_s, i_w in positions:
            window = self.__extract_window(
                sentences[i_s],
                i_w
            )
            context_vector = _mean_vector(
                self._w2v.wv,
                window
            )
            mean_vectors.append(context_vector)

        skm = KMeansClusterer(
            nb_sense,
            cosine_distance,
            rng=random.Random(self.seed),
            repeats=5,
            avoid_empty_clusters=True
        )

        return skm.cluster(mean_vectors, True)

    def __compute_context_vectors_thread_function(self, queue, sentences, senses_dict, positions):
        clusters = {}
        while not queue.empty():
            lemma = queue.get()
            clusters[lemma] = self.__compute_context_vectors(
                sentences,
                positions[lemma],
                senses_dict[lemma]
            )
        return clusters

    def __cluster_lemmas(self, sentences, senses_dict, positions):
        """TODO"""
        if self.nb_worker == 1:
            clusters = {
                lemma:
                    self.__compute_context_vectors(
                        sentences,
                        pos,
                        senses_dict[lemma]
                    )
                for lemma, pos in positions.items()
            }

            return clusters
        else:
            queue = Queue()
            for item in senses_dict.keys():
                queue.put(item=item)

            threads = [
                _ValuedThread(
                    name="t" + str(i),
                    target=self.__compute_context_vectors_thread_function,
                    args=(queue, sentences, senses_dict, positions)
                )
                for i in range(self.nb_worker)
            ]

            for thread in threads:
                thread.start()

            clusters = {}
            for thread in threads:
                thread.join()
                clusters.update(thread.value)

            return clusters

    def __extract_window(self, sentence, pos):
        """Extract a sub-sequence (window) from a list.
        Size of the extract window is define by ``self.ctx_window``.
        
        Parameters
        ----------
        sentence: list
            List of elements.
        pos: int
            Index in the list. Center of the extraction window.

        Returns
        -------
        list
            Return a list of words contains in the given sentence from ``start`` to ``end``.
        """
        start = pos - math.ceil((self.ctx_window - 1) / 2)
        end = pos + math.floor((self.ctx_window - 1) / 2)
        return sentence[max(0, start): min(len(sentence), end) + 1]

    def __gather_occurrences(self, sentences, values):
        """TODO"""
        # gather all occurrences of polysemous words in corpus
        if self.nb_worker == 1:
            return _gather_occurrences(sentences, values)
        else:
            positions = {}
            threads = []
            nb_sentence = len(sentences)
            start = 0
            for i in range(self.nb_worker - 1):
                end = nb_sentence // self.nb_worker * (i + 1)
                threads.append(_ValuedThread(
                    name=i,
                    target=_gather_occurrences,
                    args=(sentences[start:end], values, start)
                ))
                start = end
            threads.append(_ValuedThread(
                name=self.nb_worker,
                target=_gather_occurrences,
                args=(sentences[start:], values, start)
            ))

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()
                for k, v in thread.value.items():
                    if k not in positions:
                        positions[k] = v
                    else:
                        positions[k] += v

            return positions

    def __getitem__(self, word):
        """Get the embedding of a given word. This method does not disambiguate.

        Parameters
        ----------
        word: str
            The word to be represented as a vector.

        Returns
        -------
        list
            A list that is a vectorial representation of the word.
            If the word as multiple senses, return a list of vectors
        
        """
        if word in self._wv_polys:
            return self._wv_polys[word]
        return self._wv[word]

    def get_word_vector(self, word, sentence):
        """Get the embedding of a given word.
        For polysemous words, return the embedding that represent the word in the given context ``sentence``.

        Parameters
        ----------
        word: str
            The word to be represented as a vector.
        sentence: list
            A context sentence in which the word appear. Tokenized sentence, list of string.
        
        Returns
        -------
        list of list
            Integer array, embedding representation of the word
        """
        if word not in self._wv_polys:
            return [self._wv[word]]

        res = []
        for idx in _get_list_indices(sentence, word):
            ctx_vec = _mean_vector(self._wv, self.__extract_window(sentence, idx))
            _, ms = _most_similar(ctx_vec, self._wv_polys[word])
            res.append(ms)

        return res

    def huang(self, sentences, senses_dict):
        """Huang & al. algorithm for Word Sens Disambiguation.

        Parameters
        ----------
        sentences: list
            List of tokenized sentences where a tokenized sentence is a list itself.
        senses_dict: dict
            Dictionary which map polysemous words with their number of senses.
            By default, words not mapped will be considered non polysemous.
        
        Examples
        --------
        corpus = [["I", "like", "AC/DC"],
                ["I", "need", "a", "girl", "like", "you", ",", "yeah", "yeah"],
                ["I", "really", "like", "music"]]
        polysems = {"like" : 2}

        Huang().huang(corpus, polysems)
        """

        # Learn the w2v embeddings
        if self.verbose:
            print("Huang: phase 1, learning word vectors")
        self._w2v = Word2Vec(sentences=sentences,
                             min_count=1,
                             size=self._vector_size,
                             seed=self.seed,
                             workers=self.nb_worker)

        if self.verbose:
            print("Huang: Seek for polysemous words in corpus")
        positions = self.__gather_occurrences(sentences, senses_dict.keys())

        if self.verbose:
            print("Huang: phase 2, compute context vectors")
        clusters = self.__cluster_lemmas(sentences, senses_dict, positions)

        if self.verbose:
            print("Huang: phase 3, re-annotate the corpus")
        # re-annotate the corpus for new w2v embeddings learning, those embeddings are our final ones
        # TODO modif description :
        # la réannotation modifie le corpus original donné en param, voir si ca peu poser pb
        for lemma, pos in positions.items():
            for idx, (i_s, i_w) in enumerate(pos):
                # code the different senses of a lemma with a "#n" where refer to a sens of this lemma
                sentences[i_s][i_w] = lemma + "#" + str(clusters[lemma][idx])
        with open("output/corpus.txt", "w") as file:
            file.write("\n".join([" ".join(line) for line in sentences]))

        if self.verbose:
            print("Huang: phase 4, learning final word vectors")
        # 2nd w2v embeddings learning phase
        self._w2v = Word2Vec(sentences=sentences,
                             size=self._vector_size,
                             min_count=self.min_count,
                             seed=self.seed,
                             workers=self.nb_worker)

        if self.verbose:
            print("Huang: copy word vectors")
        # copy w2v embeddings in attributes
        self._wv = {lemma: self._w2v.wv[lemma] for lemma in self._w2v.wv.vocab}

        # wv for polysemous lemmas
        self._wv_polys = {lemma: [] for lemma in senses_dict.keys()}
        for lemma, senses in senses_dict.items():
            for i_s in range(senses):
                tag = lemma + "#" + str(i_s)

                if tag not in self._w2v.wv:
                    print(f"WARNING: not enough occurrences for {lemma} sense {i_s} in corpus."
                          f"This sense will be removed from the word vectors base"
                          f"or consider reducing the min_count parameter")
                else:
                    # multiple wv for a same lemma in stored in this attribute
                    self._wv_polys[lemma].append(self._w2v.wv[tag])
                    # remove the coded entry for lemmas in the wv dict ("lemma#n" coding)
                    del self._wv[tag]
            if len(self._wv_polys[lemma]) == 0:
                del self._wv_polys[lemma]
            elif len(self._wv_polys[lemma]) == 1:
                self._wv[lemma] = self._wv_polys[lemma][0]
                del self._wv_polys[lemma]
            else:
                # add a new entry in the base wv dict for computation purposes.
                # a polysemous lemma does not appear in those wv but we need to attribute a wv
                # to a polysemous word when calling the getWordVector function.
                # the mapping ion this dict for a polysemous lemma is a mean of it's sense vectors.
                self._wv[lemma] = np.mean(self._wv_polys[lemma], axis=0)
        del self._w2v

        if self.verbose:
            print("Huang: Done !")

    @staticmethod
    def load(path):
        """Load

        See Also
        --------
        save : Save the model

        Parameters
        ----------
        path: str
            Path to a saved model to be loaded

        Returns
        -------
        Huang
            Return a loaded model
        """
        with open(path, "rb") as file:
            return pickle.load(file)

    def most_similar(self):
        """TODO"""
        # for enum, sense in enumerate(h["<acr::cogc>"]):
        #
        #     res = []
        #     for tok in h.vocab():
        #         if tok in h.polysemous_vocab():
        #             continue
        #         res.append((
        #             tok,
        #             sklearn.metrics.pairwise.cosine_similarity([sense], [h[tok]])[0,0]
        #         ))
        #
        #     print(f"sens {enum}")
        #     print(sorted(res, key=lambda x: x[1])[-20:])
        pass

    def polysemous_vocab(self):
        """Get the vocabulary of polysemous words/

        Returns
        -------
        dict_keys
            Set of lemmas
        """
        return list(self._wv_polys.keys())

    def vocab(self):
        """Get the vocabulary (polysemous words included)
        
        Returns
        -------
        list
            Set of lemmas
        """
        return list(self._wv.keys())

    def save(self, path):
        """Save the model

        See Also
        --------
        load : Load a saved model.

        Parameters
        ----------
        path: str
            File path where to save the model
        """
        with open(path, "wb") as file:
            pickle.dump(self, file)


def _mean_vector(wv, window):
    """Compute a contextual vector
    
    Parameters
    ----------
    wv: dict
        A word vector dictionary that map word with their vector representations
    window: list
        List of string
    
    Returns
    -------
    list
        Return a word vector that is the average of the words in the given sentence.
    """
    return np.mean([wv[word] for word in window], axis=0)


def _gather_occurrences(sentences, values, offset=0):
    """
    Gather all occurrences of the given values in the corpus sentences argument.
    Return a dict that map every positions in the corpus of the given values to be found.

    Parameters
    ----------
    sentences: list of list of str
        List of sentences where a sentence is tokenized and represented as a list of string.
    values: list
        List of specific values to be found.

    Returns
    -------
    dict
        Return a dict that map a str with a list of tuple (aka positions)
        of every positions in the corpus of the given values.
    """

    positions = {}  # map lemma with pos in corpus
    for i_s, sentence in enumerate(sentences):
        for lemma in values:
            if lemma in sentence:
                pos = [(i_s + offset, p) for p in _get_list_indices(sentence, lemma)]
                if lemma not in positions:
                    positions[lemma] = pos
                else:
                    positions[lemma] += pos
    return positions


def _get_list_indices(lst, item):
    """Get the indices in the list (``lst``) that are equal to the given ``item``
    
    Parameters
    ----------
    lst: list
        List of items
    item: any
        The item to find in the list

    Yields
    ------
    int
        Index of a matching ``item`` in the list
    """
    for i in range(len(lst)):
        if lst[i] == item:
            yield i


def _most_similar(vector, candidates):
    """Return from a list of vectors (``candidates``) the most cosine similar to the given ``vector``
    
    Parameters
    ----------
    vector: list
        A vector
    candidates: list of list
        A list of vectors
    
    Returns
    -------
    idx: int
            Index of the most similar vector in the candidates list
    candidates: list
            A vector that is the most similar to the given one
    """
    idx = 0
    for i, c in enumerate(candidates):
        if cosine_distance(vector, c) > cosine_distance(vector, candidates[idx]):
            idx = i
    return idx, candidates[idx]


if __name__ == "__main__":
    corpus = [["I", "like", "AC/DC"],
              ["I", "need", "a", "girl", "like", "you", ",", "yeah", "yeah"],
              ["I", "really", "like", "music"],
              ["I", "like", "testing", "my", "code", "and", "I", "like", "putting", "3", "times", "like", "in", "my",
               "sentence"]]
    polysems = {"like": 2}

    h = Huang(vector_size=10, seed=10, nb_worker=2)
    h(corpus, polysems)
    print(h.get_word_vector("I", None))
    print(h.get_word_vector("like", ["I", "really", "like", "testing"]))
    print(h.vocab())
    print(h.polysemous_vocab())
