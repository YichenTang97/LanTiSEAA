import queue
import logging
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


ps = PorterStemmer()


class BaseDatasetDependentTSTransformer(BaseEstimator, TransformerMixin):
    """ A base time series transformer for time series mapping methods that need to 
    be fitted on train data set before being able to map text samples into time series.
    This kind of transformers will be fitted on each fold's train set and then transform
    sample texts from both train and test sets of each fold.
    """
    pass


class BaseDatasetIndependentTSTransformer(BaseEstimator, TransformerMixin):
    """ A base time series transformer for time series mapping methods that do not need to 
    be fitted on train data set before being able to map text samples into time series.
    This kind of transformers will directly transform text samples from the whole dataset 
    into time series.
    """
    pass


class TokenLenFreqTransformer(BaseDatasetIndependentTSTransformer):
    """ An implementaion of token length frequency time series mapping method

    name : str, default='tokenlenfreq'
        Name (in short) of the time series mapping method, also used as the name of the 
        column storing time series values in the time series dataframe output from the 
        transform method.
    max_length : int, default=14
        The maximum length of the token that will be considered while calculating the distribution.
    split_function : func, default=nltk.tokenize.word_tokenize
        The function used to split a text into a list of tokens.
    """

    def __init__(self, name="tokenlenfreq", max_length=14, split_function=word_tokenize):
        self.name = name
        self.max_length = max_length
        self.split_function = split_function
    
    def fit(self, X, y=None):
        """ Not used
        """

        # Return the classifier
        return self
    
    def token_length_distribution(self, text):
        '''
        Returns a list containing the distribution of token lengths calculated from the tokens split 
        from the text.

        Parameters
        ----------
        text : str
            The text to be mapped into the distribution.

        Returns
        -------
        length_frequencies : list
            The token length distribution of the sample text.
        '''
        
        lengths = list(map(len, self.split_function(text)))
        length_frequencies = []

        if len(lengths) == 0:
            for length in range(1, self.max_length+1):
                length_frequencies.append(0)
        else:
            for length in range(1, self.max_length+1):
                length_count = lengths.count(length)
                percentage = length_count / len(lengths)
                length_frequencies.append(percentage)

        return length_frequencies

    def transform(self, X):
        """ Map text samples into token length frequency time series

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            The input sample texts.

        Returns
        -------
        ts : pandas.DataFrame, shape (n_datapoints, 2)
            The dataframe containing the time series, first column stores the data points, 
            second column stores the ids of the time series
        """
        # Input validation
        check_array(np.reshape(X, (-1, 1)), dtype=str)

        # map to ts
        seq = []
        for counter, text in enumerate(X):
            seq.append(pd.DataFrame({ self.name : np.array(self.token_length_distribution(text)) }))
            seq[-1]['id'] = counter
        return pd.concat(seq)


class TokenLenSeqTransformer(BaseDatasetIndependentTSTransformer):
    """ An implementaion of token length sequence time series mapping method

    Parameters
    ----------
    name : str, default='tokenlenseq'
        Name (in short) of the time series mapping method, also used as the name of the 
        column storing time series values in the time series dataframe output from the 
        transform method.
    split_function : func, default=nltk.tokenize.word_tokenize
        The function used to split a text into a list of tokens.
    """

    def __init__(self, name="tokenlenseq", split_function=word_tokenize):
        self.name = name
        self.split_function = split_function
    
    def fit(self, X, y=None):
        """ Not used
        """

        # Return the classifier
        return self

    def token_length_sequence(self, text):
        """ Split the text sample and compute the length sequence.

        Parameters
        ----------
        text : str
            The text to be splitted and computed into length sequence

        Returns
        -------
        length_sequence : list
            The token length sequence of the sample text.
        """
        length_sequence = []
        tokens = self.split_function(text)
        if len(tokens) == 0: # if no token can be split from the text, the ts has only one value 0
            length_sequence.append(0)
        else:
            for token in tokens:
                length_sequence.append(len(token))
        return length_sequence

    def transform(self, X):
        """ Map text samples into token length sequence time series

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            The input sample texts.

        Returns
        -------
        ts : pandas.DataFrame, shape (n_datapoints, 2)
            The dataframe containing the time series, first column stores the data points, 
            second column stores the ids of the time series
        """
        # Input validation
        check_array(np.reshape(X, (-1, 1)), dtype=str)

        # map to ts
        seq = []
        for counter, text in enumerate(X):
            seq.append(pd.DataFrame({ self.name : np.array(self.token_length_sequence(text)) }))
            seq[-1]['id'] = counter
        return pd.concat(seq)


class WordCntVecTransformer(BaseDatasetDependentTSTransformer):
    """ An implementaion of word count vector time series mapping method

    Parameters
    ----------
    name : str, default='wordcntvec'
        Name (in short) of the time series mapping method, also used as the name of the 
        column storing time series values in the time series dataframe output from the 
        transform method.
    length : int, default=1000
        The number of features to be used by CountVectorizer, also the length of the 
        output time series.
    """

    def __init__(self, name="wordcntvec", length=1000):
        self.name = name
        self.length = 1000
        # constent variables
        self.analyzer = CountVectorizer().build_analyzer()
    
    def stemmed_analyzer(self, text):
        return [ps.stem(word) for word in self.analyzer(text)]
    
    def fit(self, X, y=None):
        """ Fit the CountVectorizer with the texts in X.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            The input sample texts.
        y : Not used

        Returns
        -------
        self : WordCntVecTransformer
            The fitted transformer.
        """
        # fit CountVectorizer
        vectorizer = CountVectorizer(max_features=self.length, analyzer=self.stemmed_analyzer)
        vectorizer.fit(X)
        self.vectorizer = vectorizer

        # Return the classifier
        return self

    def transform(self, X):
        """ Map text samples into word count vector time series

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            The input sample texts.

        Returns
        -------
        ts : pandas.DataFrame, shape (length*n_samples, 2)
            The dataframe containing the time series, first column stores the data points, 
            second column stores the ids of the time series
        """
        # Input validation
        check_array(np.reshape(X, (-1, 1)), dtype=str)

        # fitted validation
        check_is_fitted(self, 'vectorizer')

        # transform to count vectors
        count_vectors = self.vectorizer.transform(X)

        # map to ts
        seq = []
        for counter, vector in enumerate(count_vectors.toarray()):
            seq.append(pd.DataFrame({ self.name : vector }))
            seq[-1]['id'] = counter
        return pd.concat(seq)
    
    def get_classifiers(self):
        """ Get the fitted classifiers/transformers.

        Returns
        -------
         : dict
            A dictionary contains names of all classifiers/transformers along with the 
            classifers/transformers objects.
        """
        return { 'vectorizer' : self.vectorizer }

    def get_fitted_data(self):
        """ Get the fitted data.

        Returns
        -------
         : dict
            A dictionary contains names of all fitted data along with the actual data.
        """
        return {}


def to_stemmed_tokens(text):
    ''' Split a text into stemmed tokens without converting capital letters to lowercase.

    Parameters
    ----------
    text : str 
        The text sample to be splitted.

    Returns
    -------
    tokens : list
        The tokens splitted from the text sample.
    '''
    # without lowering all letters
    tokens = []
    for token in word_tokenize(text):
        # because porter stemmer automatically lowering all capital letters, we need to bring back any capital letter if exist
        stem = ps.stem(token)
        if stem != token:
            stem_char_list = list(stem)
            length = np.minimum(len(stem), len(token))
            for i in range(length):
                # if stemmer lowered a letter, bring it back to capital
                if (stem[i] != token[i]) and (stem[i].upper() == token[i]):
                    stem_char_list[i] = token[i]
            stem = "".join(stem_char_list)
            
        tokens.append(stem)
    return tokens


class TokenFreqTransformer(BaseDatasetDependentTSTransformer):
    """ An implementaion of token frequency time series mapping method

    Parameters
    ----------
    name : str, default='tokenfreq'
        Name (in short) of the time series mapping method, also used as the name of the 
        column storing time series values in the time series dataframe output from the 
        transform method.
    split_function : func, default=authorshipattribution.ts.to_stemmed_tokens
        The function used to split a text into a list of tokens.
    """

    def __init__(self, name="tokenfreq", split_function=to_stemmed_tokens):
        self.name = name
        self.split_function = split_function
    
    def token_frequency_dictionary(self, texts):
        ''' Split an array of texts into words and build a token frequency dictionary from them. 
        The token frequency dictionary will have all unique tokens in the texts as keys and 
        the number of occurrence of them in the texts.
        
        Parameters
        ----------
        texts : array-like, shape (n_samples,)
            The text samples the dictionary would be built from.
        
        Returns
        -------
        tokenfreq_dict : dict
            The token frequency dictionary built from the text samples.
        '''
        tokenfreq_dict = {}
        for text in texts:
            features = self.split_function(text)
            for feature in features:
                if feature not in tokenfreq_dict:
                        tokenfreq_dict[feature] = 1
                else:
                        tokenfreq_dict[feature] += 1
        return tokenfreq_dict
    
    def fit(self, X, y=None):
        """ Compute the token frequency dictionary.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            The input sample texts.
        y : Not used

        Returns
        -------
        self : TokenFreqTransformer
            The fitted transformer.
        """
        # compute token frequency dictionary
        self.tokenfreq_dict = self.token_frequency_dictionary(X)

        # Return the classifier
        return self

    def token_frequencies(self, text):
        ''' Splits text into tokens using the split_function and returns the corresponding 
        frequencies of the tokens in the token frequency dictionary. Tokens that are not found in 
        the token frequency dictionary will be given values of 0.
        
        Parameters
        ----------
        text : str
            The text to be split and transform into frequencies.

        Returns
        -------
        frequencies : list
            The token frequency sequence of the sample text.
        '''
        tokens = self.split_function(text)
        frequencies = []
        for token in tokens:
            try:
                frequencies.append(self.tokenfreq_dict[token])
            except KeyError:
                frequencies.append(0)
        
        return frequencies
    
    def transform(self, X):
        """ Map text samples into token frequency time series

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            The input sample texts.

        Returns
        -------
        ts : pandas.DataFrame, shape (n_datapoints, 2)
            The dataframe containing the time series, first column stores the data points, 
            second column stores the ids of the time series
        """
        # Input validation
        check_array(np.reshape(X, (-1, 1)), dtype=str)

        # fitted validation
        check_is_fitted(self, 'tokenfreq_dict')

        # map to ts
        seq = []
        for counter, text in enumerate(X):
            frequencies = self.token_frequencies(text)
            if len(frequencies) == 0:
                seq.append(pd.DataFrame({ self.name : np.array([0]) }))
            else:
                seq.append(pd.DataFrame({ self.name : np.log1p(np.array(frequencies)) }))
            seq[-1]['id'] = counter
        return pd.concat(seq)

    def get_classifiers(self):
        """ Get the fitted classifiers/transformers.

        Returns
        -------
         : dict
            A dictionary contains names of all classifiers/transformers along with the 
            classifers/transformers objects.
        """
        return {}

    def get_fitted_data(self):
        """ Get the fitted data.

        Returns
        -------
         : dict
            A dictionary contains names of all fitted data along with the actual data.
        """
        return { 'tokenfreq_dict' : self.tokenfreq_dict }


class TokenFreqRankTransformer(BaseDatasetDependentTSTransformer):
    """ An implementaion of token frequency rank time series mapping method

    Parameters
    ----------
    name : str, default='tokenfreqrank'
        Name (in short) of the time series mapping method, also used as the name of the 
        column storing time series values in the time series dataframe output from the 
        transform method.
    split_function : func, default=authorshipattribution.ts.to_stemmed_tokens
        The function used to split a text into a list of tokens.
    """

    def __init__(self, name="tokenfreqrank", split_function=to_stemmed_tokens):
        self.name = name
        self.split_function = split_function

    def token_rank_dictionary(self, tokenfreq_dict):
        ''' Rank the tokens in the token frequency dictionary.
        The tokens in the token frequency dictionary will be ranked based on their numbers of 
        occurrences, and the tokens with the same number of occurrences will be given arbitrary rank 
        in their rank interval.

        Parameters
        ----------
        tokenfreq_dict : dict
            The token frequency dictionary built from text samples.

        Returns
        -------
        tokenrank_dict : dict
            The token rank dictionary built from the token frequency dictionary.
        '''
        
        freq_to_rank_dict = dict.fromkeys(tokenfreq_dict.values())

        # sort the frequency dictionary
        token_freq_sorted = sorted([freq for freq in tokenfreq_dict.values()], reverse=True)
        
        # give ranks for tokens, for multiple orccurrences of the same frequency, keep increment the rank, and store these ranks 
        # in a queue for the corresponding frequence
        for rank, freq in enumerate(token_freq_sorted):
            if freq_to_rank_dict[freq] == None:
                freq_to_rank_dict[freq] = queue.Queue()
            freq_to_rank_dict[freq].put(rank+1)
        
        # Give each token their corresponding rank based on their frequency, for tokens with the same frequency, the ranks stored 
        # in the corresponding queue are given one by one. The assignment is effectively random.
        tokenrank_dict = {}
        for token, freq in tokenfreq_dict.items():
            tokenrank_dict[token] = freq_to_rank_dict[freq].get()
        return tokenrank_dict
    
    def fit(self, X, y=None):
        """ Compute the token frequency dictionary and token rank dictionary.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            The input sample texts.
        y : Not used

        Returns
        -------
        self : TokenFreqRankTransformer
            The fitted transformer.
        """
        # compute token frequency dictionary and token frequency rank dictionary
        tft = TokenFreqTransformer()
        self.tokenfreq_dict = tft.token_frequency_dictionary(X)
        self.tokenrank_dict = self.token_rank_dictionary(self.tokenfreq_dict)

        # Return the classifier
        return self
    
    def transform(self, X):
        """ Map text samples into token frequency time series

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            The input sample texts.

        Returns
        -------
        ts : pandas.DataFrame, shape (n_datapoints, 2)
            The dataframe containing the time series, first column stores the data points, 
            second column stores the ids of the time series
        """
        # Input validation
        check_array(np.reshape(X, (-1, 1)), dtype=str)

        # fitted validation
        check_is_fitted(self, ['tokenfreq_dict', 'tokenrank_dict'])

        # map to ts
        token_max_rank = max(self.tokenrank_dict.values())
    
        seq = []
        for counter, text in enumerate(X):
            
            features = self.split_function(text)
            ranks = []
            for feature in features:
                try:
                    ranks.append(self.tokenrank_dict[feature])
                except KeyError: # if the word is new, give the maximum rank + 1
                    ranks.append(token_max_rank + 1)

            if len(ranks) == 0:
                seq.append(pd.DataFrame({ self.name : np.array([0]) }))
            else:
                seq.append(pd.DataFrame({ self.name : np.log1p(np.array(ranks)) }))
                
            seq[-1]['id'] = counter
            
        return pd.concat(seq)
    
    def get_classifiers(self):
        """ Get the fitted classifiers/transformers.

        Returns
        -------
         : dict
            A dictionary contains names of all classifiers/transformers along with the 
            classifers/transformers objects.
        """
        return {}

    def get_fitted_data(self):
        """ Get the fitted data.

        Returns
        -------
         : dict
            A dictionary contains names of all fitted data along with the actual data.
        """
        return {
            'tokenfreq_dict' : self.tokenfreq_dict,
            'tokenrank_dict' : self.tokenrank_dict
        }
