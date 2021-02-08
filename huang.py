import math
import random
import numpy as np
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer, cosine_distance

class Huang:
    """TODO"""

    def __init__(self, vector_size=100, ctx_window=11, seed=random.randrange(1e9)):
        """
        Parameters
        ----------
        vector_size: int
            Size of the embeddings.
        ctx_window: int
            Size of the window to compute context vectors.
        seed: int
            Seed for the random number generator.
        """
        self._vector_size = vector_size
        self.ctx_window = ctx_window
        self.seed = seed
        self._wv = {}
        self._senses = {}

    def __call__(self, sentences, senses_dict):
        """Call to the function ``self.huang()``
        
        See Also
        --------
        Huang.huang()
        """
        self.huang(sentences, senses_dict)
    
    def extractWindow(self, sentence, pos):
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
        return sentence[max(0, start) : min(len(sentence), end)+1]
    
    def __getitem__(self, word):
        """Get the embedding of a given word. Thie method does not disambiguate.

        Parameters
        ----------
        word: str
            The word to be represented as a vector.

        Returns
        -------
        list
            A list that is a vectorial reprensentation of the word.
            If the word as multiple senses, return a list of vectors
        
        """
        if word in self._senses:
            return self._senses[word]
        return self._wv[word]

    def getWordVector(self, word, sentence):
        """Get the embedding of a given word.
        For polysemous words, return the embedding that represent the word in the given context ``sentence``.

        Parameters
        ----------
        word: int
            The word to be represented as a vector.
        sentence: list
            A context sentence in which the word appear. Tokenized sentence, list of string.
        
        Returns
        -------
        list
            Integer array, embedding representation of the word
        """
        #TODO -> lors du calcul du vecteur de ctx, des mots polysemiques peuvent y apparaitre
        # cependant, ceux ne disposent pas d'un embedding unique (utilisé dans la phase d'apprentissage)
        # mais de plusieurs (on est dans la phase apres apprentissage). Il faut definir un embeddging pour les mots polysémiques
        # afin de calculer le vecteur de context.
        # solution -> moyenne des embeddings de sens
        if not word in self._senses:
            return self._wv[word]

        res = []
        for idx in self.__getListIndices(sentence, word):
            ctx_vec = self.__computeContextVector(self._wv, self.extractWindow(sentence, idx))
            _,ms = self.__mostSimilare(ctx_vec, self._senses[word])
            res.append(ms)

        if len(res) == 1:
            return res[0]
        return res

    def huang(self, sentences, senses_dict):
        """Huang & al. algorithm for Word Sens Disambiguation.

        Parameters
        ----------
        sentences: list
            List of tokenized sentences where a tokanized sentence is a list itself.
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
        
        #Learn the w2v embeddings
        w2v = Word2Vec(sentences=sentences,
                       size=self._vector_size,
                       min_count=1,
                       seed=self.seed)

        #compute the context vectors
        for lemma, nsenses in senses_dict.items():
            mean_vectors = []
            for sentence in sentences:
                for idx in self.__getListIndices(sentence, lemma):
                    mean_vectors.append(
                        self.__computeContextVector(
                            w2v.wv,
                            self.extractWindow(sentence, idx)
                        )
                    )
             
            skm = KMeansClusterer(  nsenses,
                                    cosine_distance,
                                    rng=random.Random(self.seed),
                                    repeats=5,
                                    avoid_empty_clusters=True)
            
            clustering = skm.cluster(mean_vectors, True)

        #reannotate the corpus for new w2v embeddings learning, those embeddings are our final ones
        #TODO -> la réannotation modifie le corpus original donné en param, voir si ca peu poser pb (sinon faire une corpie de ce corpus)...
        for lemma, _ in senses_dict.items():
            idx = 0
            for i_s in range(len(sentences)):
                for i_w in range(len(sentences[i_s])):
                    if sentences[i_s][i_w] == lemma:
                        sentences[i_s][i_w] = lemma + "#" + str(clustering[idx])
                        idx += 1
        w2v = Word2Vec(sentences=sentences,
                       size=self._vector_size,
                       min_count=1,
                       seed=self.seed)

        #copy w2v embeddings in attributes
        for lemma in w2v.wv.vocab:
            slemma = lemma.split("#")
            if len(slemma) == 1:
                self._wv[lemma] = w2v.wv[lemma]
            else:
                l = slemma[0]
                if not l in self._senses:
                    self._senses[l] = [w2v.wv[lemma]]
                else:
                    self._senses[l].append(w2v.wv[lemma])
                    
        #compute naif embeddings for polysemous words
        for lemma, nsenses in senses_dict.items():
            self._wv[lemma] = np.mean(self._senses[lemma], axis=0)
    
    def polysemousVocab(self):
        """Get the vocabulary of polysemous words/

        Returns
        -------
        dict_keys
            Set of lemmas
        """
        return self._senses.keys()

    def vocab(self):
        """Get the vocabulary (polysemous words included)
        
        Returns
        -------
        dict_keys
            Set of lemmas
        """
        return self._wv.keys()
    
    @staticmethod
    def __computeContextVector(wv, window):
        """TODO"""
        #TODO : changer la somme en moyenne (pour normaliser les vecteurs)
        return np.sum([wv[word] for word in window], axis=0)
    
    @staticmethod
    def __getListIndices(lst, item):
        """TODO"""
        for i in range(len(lst)):
            if lst[i] == item:
                yield i
    
    @staticmethod
    def __mostSimilare(vector, candidates):
        """TODO"""
        idx = 0
        for i,c in enumerate(candidates):
            if cosine_distance(vector, c) > cosine_distance(vector, candidates[idx]):
                idx = i
        return idx, candidates[idx]


if __name__ == "__main__":
    corpus = [["I", "like", "AC/DC"],
              ["I", "need", "a", "girl", "like", "you", ",", "yeah", "yeah"],
              ["I", "really", "like", "music"],
              ["I", "like", "testing", "my", "code", "and", "I", "like", "putting", "3", "times", "like", "in", "my", "sentence"]]
    polysems = {"like" : 2}

    h = Huang(vector_size=10, seed=10)
    h(corpus, polysems)
    print(h.getWordVector("I", None))
    print(h.getWordVector("like", ["I", "really", "like", "testing"]))
    print(h.vocab())
    print(h.polysemousVocab())