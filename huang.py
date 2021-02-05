import math
import random
import numpy as np
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer, cosine_distance

class Huang:
    """
    TODO
    """

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
        self._w2v = None
        self._wv = {}

    def __call__(self, sentences, senses_dict):
        """Call to the function huang(sentences, senses_dict)
        
        See Also
        --------
        huang
        """
        self.huang(sentences, senses_dict)
    
    def __computeContextVector(self, window):
        """Compute a context vector"""
        return np.sum([self._w2v.wv[word] for word in window], axis=0)
    
    def __getitem__(self, item):
        """Get the embedding of the given item.

        Parameters
        ----------
        item: str
            The word to be represented as a vector.

        Returns
        -------
        list
            A list that is a vectorial reprensentation of the word.
        
        """
        if item in self._wv:
            return self._wv[item]
            #TODO -> Retourner un seul vecteur (suivant le contexte d'apparition)
        return self._w2v.wv[item]

    # def getWordVector(self, word, context):
    #     """
    #     TODO -> docstring
    #     """
    #     #BUG -> pb quand w2v ne connais pas un mot...
    #     ctx_vec = self.__computeContextVector(context)
        
    #     nearest = []
    #     dist = -1
    #     for vec in self._wv[word]:
    #         print(cosine_distance(ctx_vec, vec)) 

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
        self._w2v = Word2Vec(sentences=sentences,
                             size=self.vector_size,
                             min_count=1,
                             seed=self.seed)

        #compute the context vectors
        for lemma, nsenses in senses_dict.items():
            mean_vectors = []
            for sentence in sentences:
                for idx in self.__getListIndices(sentence, lemma):
                    mean_vectors.append(
                        self.__computeContextVector(
                            self.extractWindow( sentence, 
                                                idx - math.floor((self.ctx_window - 1) / 2),
                                                idx + math.ceil((self.ctx_window - 1) / 2))))
             
            skm = KMeansClusterer(  nsenses,
                                    cosine_distance,
                                    rng=random.Random(self.seed),
                                    repeats=5,
                                    avoid_empty_clusters=True)
            
            clustering = skm.cluster(mean_vectors, True)

        #reannote the corpus for new w2v embeddings learning, those embeddings are our final ones
        #TODO -> la réannotation modifie le corpus original donné en param, voir si ca peu poser pb (sinon faire une corpie de ce corpus)...
        for lemma, _ in senses_dict.items():
            idx = 0
            for i_s in range(len(sentences)):
                for i_w in range(len(sentences[i_s])):
                    if sentences[i_s][i_w] == lemma:
                        sentences[i_s][i_w] = lemma + "#" + str(clustering[idx])
                        idx += 1
        self._w2v = Word2Vec(sentences=sentences,
                             size=self.vector_size,
                             min_count=1,
                             seed=self.seed)

        for lemma, nsenses in senses_dict.items():
            self._wv[lemma] = []
            for i in range(nsenses):
                self._wv[lemma].append(self._w2v[lemma+"#"+str(i)])
    
    @staticmethod
    def extractWindow(sentence, start, end):
        """Return the items in ``sentence`` between ``start`` and ``end``.
        Returns
        -------
        list
            Return a list of words contains in the given sentence from ``start`` to ``end``.
        """
        return sentence[max(0, start) : min(len(sentence), end)+1]
    
    @staticmethod
    def __getListIndices(lst, item):
        for i in range(len(lst)):
            if lst[i] == item:
                yield i


if __name__ == "__main__":
    corpus = [["I", "like", "AC/DC"],
              ["I", "need", "a", "girl", "like", "you", ",", "yeah", "yeah"],
              ["I", "really", "like", "music"],
              ["I", "like", "testing", "my", "code", "and", "I", "like", "putting", "3", "times", "like", "in", "my", "sentence"]]
    polysems = {"like" : 2}

    h = Huang(vector_size=10, seed=10)
    h(corpus, polysems)
    h.huang()
    print(h["like"])