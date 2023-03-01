from collections import OrderedDict
from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np


class BoW(TransformerMixin):
    """
    Bag of words tranformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow
        words = {}
        for raw in X:
            for word in raw.split():
                if word in words.keys():
                    words[word] += 1
                else:
                    words[word] = 1
        self.bow = sorted(list(words.items()), key = lambda x: x[1], reverse=True)
        self.bow = [elem[0] for elem in self.bow][:self.k]
        # fit method must always return self
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """

        result = [0] * len(self.bow)
        for word in text.split():
            try:
                ind = self.bow.index(word)
                result[ind] += 1
            except ValueError:
                continue
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow


class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize

        # self.idf[term] = log(total # of documents / # of documents with term in it)
        self.idf = OrderedDict()


    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        tokens = set(' '.join(X).split())
        tokens_cnt = {token: 0 for token in tokens}
        for token in tokens:
            n = 0
            for doc in X:
                if token in doc:
                    tokens_cnt[token] += 1
            tokens_cnt[token] = np.log(X.shape[0] / tokens_cnt[token])
        
        tokens = sorted(tokens_cnt.items(), key=lambda x: x[1], reverse=True)[:self.k]
        self.idf = OrderedDict(tokens)

        # fit method must always return self
        return self


    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """

        result = [0] * len(self.idf)
        for token in text.split():
            try:
                tf = text.count(token)
                idf = self.idf[token]
                idx = list(self.idf.keys()).index(token)
                result[idx] = tf * idf
            except KeyError:
                continue            
        return np.array(result, "float32")


    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])
