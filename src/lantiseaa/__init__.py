# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound
from lantiseaa.buffer import Buffer, MemoryBuffer, LocalBuffer

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'LanTiSEAA'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound


class LanTiSEAA():
    """
    The LanTiSEAA class uses time series transformers, feature extractor and baseline method 
    (or baseline predictions) to transform texts; extract, combine, and select features; 
    (compute baseline predictions); combine baseline predictions with time series 
    features; and train the meta-classifier for making predictions.

    LanTiSEAA itself is a classifier and can be used under sklearn framework.

    ...

    Attributes
    ----------
    buffer_ : lantiseaa.buffer.Buffer
        the buffer used to store the data generated during computation
    ts_transformers_ : List
        a list of tuples each containing a time series mapping method name and a ts_transformer
    baseline_classifier_ : Classifier
        the baseline classifier for making baseline predictions (None if not given or if 
        baseline_prediction is used)
    meta_classifier_ : Classifier
        the meta classifier for performing final predictions using time series features
        (and baseline predictions)


    Methods
    -------
    fit(X, y, baseline_prediction=None)
        Fit on the given training data set, extract features and train the classifier

    predict(self, X, baseline_prediction=None)
        Make predictions on the given testing data set

    predict_proba(self, X, baseline_prediction=None)
        Make probability predictions on the given testing data set
    
    """

    def __init__(self, ts_transformers, feature_extractor, baseline_classifier=None, buffer=MemoryBuffer(), random_state=None):
        """Construct new LanTiSEAA object

        Parameters
        ----------
        ts_transformers : List
            a list of tuples each containing a time series mapping method name and a ts_transformer.
            The ts_transformers will be used to map the texts and time series features will be 
            extracted from each type of the time series and then been combined and selected.
        
        feature_extractor : lantiseaa.extractor.TSFeatureExtractor
            the time series feature extractor for extracting and selecting relevant features

        baseline_classifier : Classifier, optional
            the baseline method for making baseline predictions. Ignored in fit, predict and 
            predict_proba if baseline_prediction is given in fit. If is None and baseline_prediction
            is not given, baseline will be ignored and only time series features will be used (default
            is None).

        buffer : lantiseaa.buffer.Buffer, optional
            the buffer used to store the data generated during computation (default is MemoryBuffer).

        random_state : int, optional
            the random seed to be passed in for any process that uses a pseudo random number generator.
            If is None, then all random processes will be using the default setting without specific 
            seeds (default is None).

        """
        pass #TODO

    
    def fit(self, X, y, baseline_prediction=None):
        """Fit on the given training data set, extract features and train the classifier

        Parameters
        ----------
        X : array-like
            the training texts
        
        y : array-like
            the true target values

        baseline_prediction : array_like, optional
            the baseline predictions to be combined with the time series features to train the
            meta-classifier. If is None, the non-None baseline_classifier will be used to make 
            baseline predictions (default is None).

        """
        pass #TODO


    def predict(self, X, baseline_prediction=None):
        """Make predictions on the given testing data set

        Parameters
        ----------
        X : array-like
            the testing texts

        baseline_prediction : array_like, optional
            the baseline predictions to be combined with the time series features for the meta-
            classifier to make predictions. If the baseline_prediction is given in fit, this 
            parameter cannot be None, or an error will be raised, otherwise this parameter is 
            ignored (default is None).

        """
        pass #TODO


    def predict_proba(self, X, baseline_prediction=None):
        """Make probability predictions on the given testing data set

        Parameters
        ----------
        X : array-like
            the testing texts

        baseline_prediction : array_like, optional
            the baseline predictions to be combined with the time series features for the meta-
            classifier to make predictions. If the baseline_prediction is given in fit, this 
            parameter cannot be None, or an error will be raised, otherwise this parameter is 
            ignored (default is None).

        """
        pass #TODO



class IterativeLanTiSEAA():
    """
    The InterativeLanTiSEAA class implements framework for the iterative stacking procedure described 
    in the paper, where individual time series mapping methods were stacked to the baseline method
    one by one until no significant improvement can be made. The framework is implemented in the 
    fit method of this class.

    The InterativeLanTiSEAA class can also work as a classifier, where the selected combination
    will be used to train a final model on the complete data set during fitting, and will be used 
    in predict and predict_proba for making predictions on new data.

    ...

    Attributes
    ----------
    buffer_ : lantiseaa.buffer.Buffer
        the buffer used to store the data generated during computation
    ts_transformers_ : List
        a list of tuples each containing a time series mapping method name and a ts_transformer
    selected_ts_transformers : List
        a list of time series mapping method names indicating the selected methods (ordered in the
        same order as they are selected)
    baseline_classifier_ : Classifier
        the baseline classifier for making baseline predictions
    meta_classifier_ : Classifier
        the meta classifier for performing final predictions using time series features
        (and baseline predictions)


    Methods
    -------
    fit(X, y)
        Perform the iterative stacking procedure and fit on the complete training data set

    predict(self, X)
        Make predictions on the given testing data set

    predict_proba(self, X)
        Make probability predictions on the given testing data set
    
    """

    def __init__(self, ts_transformers, feature_extractor, baseline_classifier=None, buffer=MemoryBuffer(), random_state=None):
        """Construct new LanTiSEAA object

        Parameters
        ----------
        ts_transformers : List
            a list of tuples each containing a time series mapping method name and a ts_transformer.
            The ts_transformers will be used to map the texts and time series features will be 
            extracted from each type of the time series and then been combined and selected.
        
        feature_extractor : lantiseaa.extractor.TSFeatureExtractor
            the time series feature extractor for extracting and selecting relevant features

        baseline_classifier : Classifier, optional
            the baseline method for making baseline predictions. Ignored in fit, predict and 
            predict_proba if baseline_prediction is given in fit. If is None and baseline_prediction
            is not given, baseline will be ignored and only time series features will be used (default
            is None).

        buffer : lantiseaa.buffer.Buffer, optional
            the buffer used to store the data generated during computation (default is MemoryBuffer).

        random_state : int, optional
            the random seed to be passed in for any process that uses a pseudo random number generator.
            If is None, then all random processes will be using the default setting without specific 
            seeds (default is None).

        """
        pass #TODO

    
    def fit(self, X, y):
        """Perform the iterative stacking procedure and fit on the complete training data set

        Parameters
        ----------
        X : array-like
            the training texts
        
        y : array-like
            the true target values

        """
        pass #TODO


    def predict(self, X):
        """Make predictions on the given testing data set

        Parameters
        ----------
        X : array-like
            the testing texts

        """
        pass #TODO


    def predict_proba(self, X):
        """Make probability predictions on the given testing data set

        Parameters
        ----------
        X : array-like
            the testing texts

        """
        pass #TODO