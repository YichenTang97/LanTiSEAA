# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from pkg_resources import get_distribution, DistributionNotFound
from lantiseaa.buffer import Buffer, MemoryBuffer, LocalBuffer
from lantiseaa.extractor import TSFeatureExtractor, TsfreshTSFeatureExtractor
from lantiseaa.nlp import BOWMNB
from lantiseaa.ts import *

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
        a list of ts_transformers
    feature_extractor : lantiseaa.extractor.TSFeatureExtractor
        the time series feature extractor for extracting and selecting relevant features
    baseline_classifier_ : Classifier
        the baseline classifier for making baseline predictions (None if not given or if 
        baseline_prediction is used)
    meta_classifier_ : Classifier
        the meta classifier for performing final predictions using combined time series 
        features (and baseline predictions)
    feature_groups_ : List
        a list of names for the feature groups combined together
    relevant_features_ : pandas.DataFrame
        the relevant features table selected by the TSFeatureExtractor


    Methods
    -------
    fit(X, y, baseline_prediction=None)
        Fit on the given training data set, extract features and train the classifier

    predict(self, X, baseline_prediction=None)
        Make predictions on the given testing data set

    predict_proba(self, X, baseline_prediction=None)
        Make probability predictions on the given testing data set
    
    """

    def __init__(self, \
                 ts_transformers=[TokenLenFreqTransformer(), TokenLenSeqTransformer(), WordCntVecTransformer(), TokenFreqTransformer(), TokenFreqRankTransformer()], \
                 feature_extractor=TsfreshTSFeatureExtractor(), baseline_classifier=BOWMNB(), \
                 meta_classifier=GradientBoostingClassifier(), buffer=MemoryBuffer()):
        """Construct new LanTiSEAA object

        Parameters
        ----------
        ts_transformers : List, optional
            a list of ts_transformers. The ts_transformers will be used to map the texts and time series 
            features will be extracted from each type of the time series and then been combined and selected
            (in default the list is five transformers - TokenLenFreqTransformer, TokenLenSeqTransformer, 
            WordCntVecTransformer, TokenFreqTransformer, and TokenFreqRankTransformer - all under 
            lantiseaa.ts package).
        
        feature_extractor : lantiseaa.extractor.TSFeatureExtractor, optional
            the time series feature extractor for extracting and selecting relevant features (default is
            lantiseaa.extractor.TsfreshTSFeatureExtractor).

        baseline_classifier : Classifier, optional
            the baseline method for making baseline predictions. Ignored in fit, predict and 
            predict_proba if baseline_prediction is given in fit. If is None and baseline_prediction
            is not given, baseline will be ignored and only time series features will be used (default
            is lantiseaa.nlp.BOWMNB).

        meta_classifier : Classifier, optional
            the meta-classifier to be trained on combined features and make predictions (default is 
            GradientBoostingClassifier from sklearn. Note this is different from the XGBClassifier by XGBoost
            we used in the paper due to a generalizability issue with XGBClassifier - the number of classes
            should be defined when initializing the object. For better performance, pass in XGBClassifier 
            instead of GradientBoostingClassifier instead).

        buffer : lantiseaa.buffer.Buffer, optional
            the buffer used to store the data generated during computation (default is MemoryBuffer).

        """
        #TODO add type checkers
        self.ts_transformers_ = ts_transformers
        self.feature_extractor_ = feature_extractor
        self.baseline_classifier_ = baseline_classifier
        self.meta_classifier_ = meta_classifier
        self.buffer_ = buffer
        self.flags_ = {'baseline_prediction_given_in_fit': False}
    

    def get_combined_features(self, train_test='train', surfix=None):
        features = []
        for feature_group in self.feature_groups_:
            features.append(self.buffer_.read_feature_set(feature_group, train_test=train_test, surfix=None))
        return pd.concat(features, axis=1)

    
    def fit(self, X, y, baseline_prediction=None, classes=None, baseline_clf_fit_kwargs={}, baseline_clf_predict_proba_kwargs={}, meta_clf_fit_kwargs={}):
        """Fit on the given training data set, extract features and train the classifier

        Parameters
        ----------
        X : array-like
            the training texts
        
        y : array-like
            the true target values

        baseline_prediction : array-like, optional
            the baseline predictions to be combined with the time series features to train the
            meta-classifier. If is None, the non-None baseline_classifier will be used to make 
            baseline predictions (default is None).
        
        classes : list, optional
            the classes used as pandas.DataFrame column names when saving baseline_predictions
            made by baseline_classifier. If is None, the value will be retrieved from 
            self.baseline_classifier_.classes_ if exists (default is None).

        baseline_clf_fit_kwargs : dict, Optional
            the kwargs to be passed to the baseline classifier when calling fit. Ignored if 
            baseline_prediction is not None (default is an empty dictionary).

        baseline_clf_predict_proba_kwargs : dict, Optional
            the kwargs to be passed to the baseline classifier when calling predict_proba. 
            Ignored if baseline_prediction is not None (default is an empty dictionary).

        meta_clf_fit_kwargs : dict, Optional
            the kwargs to be passed to the meta classifier when calling fit (default is an empty 
            dictionary).
            
        """
        self.feature_groups_ = []

        # if baseline_prediction is given, ignore baseline_classifier
        if baseline_prediction is not None:
            self.flags_['baseline_prediction_given_in_fit'] = True
            # TODO check baseline_prediction shape
            if not isinstance(baseline_prediction, pd.DataFrame):
                baseline_prediction = pd.DataFrame(data=baseline_prediction)
        else:
            self.flags_['baseline_prediction_given_in_fit'] = False
            if self.baseline_classifier_ is not None: # if baseline_classifier is given
                # compute baseline method
                logging.info("Computing baseline method.")
                self.baseline_classifier_.fit(X, y, **baseline_clf_fit_kwargs)
                pred = self.baseline_classifier_.predict_proba(X, **baseline_clf_predict_proba_kwargs)
                if classes is None:
                    try:
                        classes = self.baseline_classifier_.classes_
                    except AttributeError:
                        pass
                baseline_prediction = pd.DataFrame(data=pred, columns=classes)
                self.buffer_.save_class(self.baseline_classifier_, method_name='baseline', 
                                        class_name=self.baseline_classifier_.__class__.__name__)
        
        # if at lease one of baseline_prediction or baseline_classifier is given so baseline 
        # predictions was computed
        if baseline_prediction is not None:
            self.buffer_.save_feature_set(baseline_prediction, method_name='baseline', train_test='train')
            self.buffer_.save_prediction(baseline_prediction, method_name='baseline', train_test='train')
            self.feature_groups_.append('baseline')
        
        # compute time series features
        for transformer in self.ts_transformers_:
            logging.info("Computing {} time series.".format(transformer.name))
            # map texts into time series
            time_series = transformer.transform(X)
            # extract and save time series features
            ts_features = self.feature_extractor_.extract_features(time_series)
            self.buffer_.save_feature_set(ts_features, method_name=transformer.name, train_test='train')
            self.feature_groups_.append(transformer.name)

        # combine features
        X_combined = self.get_combined_features(train_test='train', surfix=None)
        combined_name = "_".join(self.feature_groups_)

        # select and save relevant features 
        self.relevant_features_ = self.feature_extractor_.select_relevant_features(X_combined)
        self.buffer_.save_feature_relevance_table(self.relevant_features_, method_name=combined_name)
        X_relevant = X_combined[self.relevant_features_.feature]

        # train and save meta-classifier
        self.meta_classifier_.fit(X_relevant, y, **meta_clf_fit_kwargs)
        self.buffer_.save_class(self.meta_classifier_, method_name='meta_classifier', 
                                    class_name=self.meta_classifier_.__class__.__name__)
        
        return self


    def precompute_X(self, X, baseline_prediction=None, classes=None, surfix=None, baseline_clf_predict_proba_kwargs={}):
        """Precompute X, prepare a feature matrix for meta-classifier to make predictions

        Parameters
        ----------
        X : array-like
            the testing texts, ignored if precomputed_X is not None.

        baseline_prediction : array_like, optional
            the baseline predictions to be combined with the time series features for the meta-
            classifier to make predictions. If the baseline_prediction is given in fit, this 
            parameter cannot be None, or an error will be raised, otherwise this parameter is 
            ignored (default is None).

        classes : list, optional
            the classes used as pandas.DataFrame column names when saving baseline_predictions
            made by baseline_classifier. If is None, the value will be retrieved from 
            self.baseline_classifier_.classes_ if exists (default is None).

        surfix : str, optional
            the surfix to be passed to the buffer when saving data

        baseline_clf_predict_proba_kwargs : dict, Optional
            the kwargs to be passed to the baseline classifier when calling predict_proba. 
            Ignored if baseline_prediction is not None (default is an empty dictionary).

        meta_clf_predict_kwargs : dict, Optional
            the kwargs to be passed to the meta classifier when calling predict (default is an 
            empty dictionary).
        
        """
        assert (baseline_prediction is None and self.flags_['baseline_prediction_given_in_fit'] == True), "baseline_prediction cannot be None if it was used in fit."
        if 'baseline' in self.feature_groups_:
            if baseline_prediction is not None:
                # TODO check baseline_prediction shape
                if not isinstance(baseline_prediction, pd.DataFrame):
                    baseline_prediction = pd.DataFrame(data=baseline_prediction)
            else:
                # compute baseline method
                logging.info("Computing baseline method.")
                pred = self.baseline_classifier_.predict_proba(X, **baseline_clf_predict_proba_kwargs)
                if classes is None:
                    try:
                        baseline_pred_classes = self.baseline_classifier_.classes_
                    except AttributeError:
                        baseline_pred_classes = None
                else:
                    baseline_pred_classes = classes
                baseline_prediction = pd.DataFrame(data=pred, columns=baseline_pred_classes)
                
            self.buffer_.save_feature_set(baseline_prediction, method_name='baseline', train_test='test', surfix=surfix)
            self.buffer_.save_prediction(baseline_prediction, method_name='baseline', train_test='test', surfix=surfix)

        # compute time series features
        for transformer in self.ts_transformers_:
            logging.info("Computing {} time series.".format(transformer.name))
            # map texts into time series
            time_series = transformer.transform(X)
            # extract and save time series features
            ts_features = self.feature_extractor_.extract_features(time_series)
            self.buffer_.save_feature_set(ts_features, method_name=transformer.name, train_test='test', surfix=surfix)
            
        # combine features and select relevant features
        X_combined = self.get_combined_features(train_test='test', surfix=surfix)
        return X_combined[self.relevant_features_.feature]


    def predict(self, X, X_precomputed=None, baseline_prediction=None, classes=None, surfix=None, baseline_clf_predict_proba_kwargs={}, meta_clf_predict_kwargs={}):
        """Make predictions on the given testing data set

        Parameters
        ----------
        X : array-like
            the testing texts, ignored if precomputed_X is not None.

        X_precomputed : array_like, optional
            the precomputed combined feature set for the meta-classifier to make predictions,
            often used when making predictions for the training data set. Besides, when saving
            to buffer when X_precomputed is given, "train_test" will not be specified, instead, 
            surfix can be used to indicate the property (train, test, etc.) of the precomputed X.

        baseline_prediction : array_like, optional
            the baseline predictions to be combined with the time series features for the meta-
            classifier to make predictions. If the baseline_prediction is given in fit, this 
            parameter cannot be None, or an error will be raised, otherwise this parameter is 
            ignored. Also ignored if precomputed_X is not None (default of this parameter is 
            None).

        classes : list, optional
            the classes used as pandas.DataFrame column names when saving baseline_predictions
            made by baseline_classifier. If is None, the value will be retrieved from 
            self.baseline_classifier_.classes_ if exists (default is None).

        surfix : str, optional
            the surfix to be passed to the buffer when saving data

        baseline_clf_predict_proba_kwargs : dict, Optional
            the kwargs to be passed to the baseline classifier when calling predict_proba. 
            Ignored if baseline_prediction is not None (default is an empty dictionary).

        meta_clf_predict_kwargs : dict, Optional
            the kwargs to be passed to the meta classifier when calling predict (default is an 
            empty dictionary).
        
        """
        if X_precomputed is not None:
            X_relevant = X_precomputed[self.relevant_features_.feature]
            pred = self.meta_classifier_.predict(X_relevant, **meta_clf_predict_kwargs)
            self.buffer_.save_prediction(pd.DataFrame(pred, columns=['y']), method_name='meta_classifier', surfix=surfix)

            return pred

        else:
            X_relevant = self.precompute_X(X, baseline_prediction=baseline_prediction, classes=classes,
                                           surfix=surfix, baseline_clf_predict_proba_kwargs=baseline_clf_predict_proba_kwargs)
            
            pred = self.meta_classifier_.predict(X_relevant, **meta_clf_predict_kwargs)
            self.buffer_.save_prediction(pd.DataFrame(pred, columns=['y']), method_name='meta_classifier', train_test='test', surfix=surfix)

            return pred


    def predict_proba(self, X, X_precomputed=None, baseline_prediction=None, classes=None, surfix=None, baseline_clf_predict_proba_kwargs={}, meta_clf_predict_proba_kwargs={}):
        """Make probability predictions on the given testing data set

        Parameters
        ----------
        X : array-like
            the testing texts

        X_precomputed : array_like, optional
            the precomputed combined feature set for the meta-classifier to make predictions,
            often used when making predictions for the training data set. Besides, when saving
            to buffer when X_precomputed is given, "train_test" will not be specified, instead, 
            surfix can be used to indicate the property (train, test, etc.) of the precomputed X.

        baseline_prediction : array_like, optional
            the baseline predictions to be combined with the time series features for the meta-
            classifier to make predictions. If the baseline_prediction is given in fit, this 
            parameter cannot be None, or an error will be raised, otherwise this parameter is 
            ignored. Also ignored if precomputed_X is not None (default of this parameter is 
            None).

        classes : list, optional
            the classes used as pandas.DataFrame column names when saving baseline_predictions
            made by baseline_classifier and predictions made by the meta_classifier. If is None, 
            the corresponding values will be retrieved from self.baseline_classifier_.classes_ and 
            self.meta_classifier_.classes_ if exist (default is None).

        surfix : str, optional
            the surfix to be passed to the buffer when saving data

        baseline_clf_predict_proba_kwargs : dict, Optional
            the kwargs to be passed to the baseline classifier when calling predict_proba. 
            Ignored if baseline_prediction is not None (default is an empty dictionary).

        meta_clf_predict_proba_kwargs : dict, Optional
            the kwargs to be passed to the meta classifier when calling predict_proba (default is 
            an empty dictionary).

        """
        # get column names for DataFrame saving meta-classifier predictions
        if classes is None:
            try:
                meta_clf_pred_classes = self.meta_classifier_.classes_
            except AttributeError:
                meta_clf_pred_classes = None
        else:
            meta_clf_pred_classes = classes
        
        # making predictions
        if X_precomputed is not None:
            X_relevant = X_precomputed[self.relevant_features_.feature]
            pred = self.meta_classifier_.predict_proba(X_relevant, **meta_clf_predict_proba_kwargs)
            self.buffer_.save_prediction(pd.DataFrame(pred, columns=meta_clf_pred_classes), method_name='meta_classifier', surfix=surfix)

            return pred

        else:
            X_relevant = self.precompute_X(X, baseline_prediction=baseline_prediction, classes=classes,
                                           surfix=surfix, baseline_clf_predict_proba_kwargs=baseline_clf_predict_proba_kwargs)
            
            pred = self.meta_classifier_.predict_proba(X_relevant, **meta_clf_predict_proba_kwargs)
            self.buffer_.save_prediction(pd.DataFrame(pred, columns=meta_clf_pred_classes), method_name='meta_classifier', train_test='test', surfix=surfix)

            return pred



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