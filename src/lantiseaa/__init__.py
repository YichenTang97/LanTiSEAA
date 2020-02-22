# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
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
        self.ts_transformers = ts_transformers
        self.feature_extractor = feature_extractor
        self.baseline_classifier = baseline_classifier
        self.meta_classifier = meta_classifier
        self.buffer = buffer
        self.flags_ = {'baseline_prediction_given_in_fit': False}
    

    def get_combined_features(self, fold_number=None, train_test='train', surfix=None):
        features = []
        for feature_group in self.feature_groups_:
            features.append(self.buffer.read_feature_set(feature_group, fold_number=None, train_test=train_test, surfix=None))
        return pd.concat(features, axis=1)


    def get_classes(self, classes=None, clf=None):
        # get classes from self.meta_classifier in default
        if clf is None:
            clf = self.meta_classifier

        if classes is None:
            try:
                return clf.classes_
            except AttributeError:
                return None
        else:
            return classes
    

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
            self.baseline_classifier.classes_ if exists (default is None).

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
            if self.baseline_classifier is not None: # if baseline_classifier is given
                # compute baseline method
                logging.info("Computing baseline method.")
                self.baseline_classifier.fit(X, y, **baseline_clf_fit_kwargs)
                pred = self.baseline_classifier.predict_proba(X, **baseline_clf_predict_proba_kwargs)
                classes = self.get_classes(classes, self.baseline_classifier)
                baseline_prediction = pd.DataFrame(data=pred, columns=classes)
                self.buffer.save_class(self.baseline_classifier, method_name='baseline', 
                                        class_name=self.baseline_classifier.__class__.__name__)
        
        # if at lease one of baseline_prediction or baseline_classifier is given so baseline 
        # predictions was computed
        if baseline_prediction is not None:
            self.buffer.save_feature_set(baseline_prediction, method_name='baseline', train_test='train')
            self.buffer.save_prediction(baseline_prediction, method_name='baseline', train_test='train')
            self.feature_groups_.append('baseline')
        
        # compute time series features
        for transformer in self.ts_transformers:
            logging.info("Computing {} time series.".format(transformer.name))
            # map texts into time series
            time_series = transformer.transform(X)
            # extract and save time series features
            ts_features = self.feature_extractor.extract_features(time_series)
            self.buffer.save_feature_set(ts_features, method_name=transformer.name, train_test='train')
            self.feature_groups_.append(transformer.name)

        # combine features
        X_combined = self.get_combined_features(fold_number=None, train_test='train', surfix=None)
        combined_name = "_".join(self.feature_groups_)

        # select and save relevant features 
        self.relevant_features_ = self.feature_extractor.select_relevant_features(X_combined)
        self.buffer.save_feature_relevance_table(self.relevant_features_, method_name=combined_name)
        X_relevant = X_combined[self.relevant_features_.feature]

        # train and save meta-classifier
        self.meta_classifier.fit(X_relevant, y, **meta_clf_fit_kwargs)
        self.buffer.save_class(self.meta_classifier, method_name='meta_classifier', 
                                    class_name=self.meta_classifier.__class__.__name__)
        
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
            self.baseline_classifier.classes_ if exists (default is None).

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
                pred = self.baseline_classifier.predict_proba(X, **baseline_clf_predict_proba_kwargs)
                baseline_pred_classes = self.get_classes(classes, self.baseline_classifier)
                baseline_prediction = pd.DataFrame(data=pred, columns=baseline_pred_classes)
                
            self.buffer.save_feature_set(baseline_prediction, method_name='baseline', train_test='test', surfix=surfix)
            self.buffer.save_prediction(baseline_prediction, method_name='baseline', train_test='test', surfix=surfix)

        # compute time series features
        for transformer in self.ts_transformers:
            logging.info("Computing {} time series.".format(transformer.name))
            # map texts into time series
            time_series = transformer.transform(X)
            # extract and save time series features
            ts_features = self.feature_extractor.extract_features(time_series)
            self.buffer.save_feature_set(ts_features, method_name=transformer.name, train_test='test', surfix=surfix)
            
        # combine features and select relevant features
        X_combined = self.get_combined_features(fold_number=None, train_test='test', surfix=surfix)
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
            self.baseline_classifier.classes_ if exists (default is None).

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
            pred = self.meta_classifier.predict(X_relevant, **meta_clf_predict_kwargs)
            self.buffer.save_prediction(pd.DataFrame(pred, columns=['y']), method_name='meta_classifier', surfix=surfix)

            return pred

        else:
            X_relevant = self.precompute_X(X, baseline_prediction=baseline_prediction, classes=classes,
                                           surfix=surfix, baseline_clf_predict_proba_kwargs=baseline_clf_predict_proba_kwargs)
            
            pred = self.meta_classifier.predict(X_relevant, **meta_clf_predict_kwargs)
            self.buffer.save_prediction(pd.DataFrame(pred, columns=['y']), method_name='meta_classifier', train_test='test', surfix=surfix)

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
            the corresponding values will be retrieved from self.baseline_classifier.classes_ and 
            self.meta_classifier.classes_ if exist (default is None).

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
        meta_clf_pred_classes = self.get_classes(classes, self.meta_classifier)
        
        # making predictions
        if X_precomputed is not None:
            X_relevant = X_precomputed[self.relevant_features_.feature]
            pred = self.meta_classifier.predict_proba(X_relevant, **meta_clf_predict_proba_kwargs)
            self.buffer.save_prediction(pd.DataFrame(pred, columns=meta_clf_pred_classes), method_name='meta_classifier', surfix=surfix)

            return pred

        else:
            X_relevant = self.precompute_X(X, baseline_prediction=baseline_prediction, classes=classes,
                                           surfix=surfix, baseline_clf_predict_proba_kwargs=baseline_clf_predict_proba_kwargs)
            
            pred = self.meta_classifier.predict_proba(X_relevant, **meta_clf_predict_proba_kwargs)
            self.buffer.save_prediction(pd.DataFrame(pred, columns=meta_clf_pred_classes), method_name='meta_classifier', train_test='test', surfix=surfix)

            return pred



class IterativeLanTiSEAA(LanTiSEAA):
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
    feature_groups_ : List
        a list of names for the feature groups combined together
    relevant_features_ : pandas.DataFrame
        the relevant features table selected by the TSFeatureExtractor
    fold_indices_ : pandas.DataFrame
        the fold indices used to split X in fit


    Methods
    -------
    fit(X, y)
        Perform the iterative stacking procedure and fit on the complete training data set

    predict(self, X)
        Make predictions on the given testing data set

    predict_proba(self, X)
        Make probability predictions on the given testing data set
    
    """

    def __init__(self, \
                 ts_transformers=[TokenLenFreqTransformer(), TokenLenSeqTransformer(), WordCntVecTransformer(), TokenFreqTransformer(), TokenFreqRankTransformer()], \
                 feature_extractor=TsfreshTSFeatureExtractor(), baseline_classifier=BOWMNB(), \
                 meta_classifier=GradientBoostingClassifier(), cv=None, \
                 metric=log_loss, needs_proba=True, greater_is_better=False, \
                 buffer=MemoryBuffer(), random_state=None):
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

        cv : int, cross-validation generator or an iterable, optional
            the cv splitting strategy used for setting up the cvs for the iterative framework. Values can 
            be chosen from: 
            * None, to use the default 10-fold cross validation generating by a StratifiedKFold,
            * integer, to specify the number of folds in a StratifiedKFold, cannot be less than 5 to allow accurate statistical tests
            * cross-validation generator
            When None or an integer is passed in, a StratifiedKFold will be used to split the folds and 
            the random_state parameter will be used to specify the random seed for the StratifiedKFold.
        
        metric : sklearn.metric, optional
            the metric used to score the predictions (default is log_loss)
        
        needs_proba : boolean, optional
            property of metric - where it needs predict_proba to score the predictions (default is True)
        
        greater_is_better : boolean, optional
            property of metric - where a greater score is a better score (default is False)

        buffer : lantiseaa.buffer.Buffer, optional
            the buffer used to store the data generated during computation (default is MemoryBuffer).

        random_state : int, optional
            the random seed to be used when constructing cv_folds, ignored if the cv parameter given is a 
            cross-validation generator or an iterable (default is None).

        """
        self.ts_transformers = ts_transformers
        self.feature_extractor = feature_extractor
        self.baseline_classifier = baseline_classifier
        self.meta_classifier = meta_classifier
        self.metric = metric
        self.needs_proba = needs_proba
        self.greater_is_better = greater_is_better
        self.buffer = buffer
        self.random_state = random_state
        self.flags_ = {'baseline_prediction_given_in_fit': False}

        if cv is None:
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
        elif isinstance(cv, int):
            assert (cv >= 5), "Number of folds cannot be less than 5 to allow accurate statistical tests."
            cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        self.cv = cv


    def fold_train_test(self, fold, X, y):
        # split train test
        train_indices = fold[1].train
        test_indices = fold[1].test
        train = X[train_indices].reset_index(drop=True)
        test = X[test_indices].reset_index(drop=True)
        y_train = y[train_indices].reset_index(drop=True)
        y_test = y[test_indices].reset_index(drop=True)

        return train, test, y_train, y_test

    
    def fit(self, X, y, baseline_prediction=None, classes=None, \
            baseline_clf_fit_kwargs={}, baseline_clf_predict_proba_kwargs={}, \
            meta_clf_fit_kwargs={}, meta_clf_predict_proba_kwargs={}):
        """Perform the iterative stacking procedure and fit on the complete training data set

        Unlike LanTiSEAA, baseline_prediction parameter is not allowed here as the data set 
        will be split into multiple folds during iterative computation. To get the most realistic
        measurements in how the time series features will improve the baseline method, the 
        baseline method need to be fit on training data and predict on testing data under each 
        fold. A pre-defined baseline_prediction will lead to 

        Parameters
        ----------
        X : array-like
            the training texts
        
        y : array-like
            the true target values

        (deprecated) baseline_prediction : array-like, optional
            the baseline predictions to be combined with the time series features to train the
            meta-classifier. If is None, the non-None baseline_classifier will be used to make 
            baseline predictions (default is None).

            It is deprecated here as the data set will be split into multiple folds during iterative 
            computation. To get the most realistic measurements in how the time series features will 
            improve the baseline method, the baseline method need to be fit on training data and 
            predict on testing data under each fold. A pre-defined baseline_prediction will lead to 
            biased measurements and selection of the time series methods.
        
        classes : list, optional
            the classes used as pandas.DataFrame column names when saving baseline_predictions
            made by baseline_classifier and predictions made by the meta_classifier. If is None, 
            the corresponding values will be retrieved from self.baseline_classifier.classes_ and 
            self.meta_classifier.classes_ if exist (default is None).

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

        if (baseline_prediction is not None) or (self.baseline_classifier is not None):
            self.feature_groups_.append('baseline')

        # if baseline_prediction is given, ignore baseline_classifier
        if baseline_prediction is not None:
            self.flags_['baseline_prediction_given_in_fit'] = True
            # TODO check baseline_prediction shape
            if not isinstance(baseline_prediction, pd.DataFrame):
                baseline_prediction = pd.DataFrame(data=baseline_prediction)
        else:
            self.flags_['baseline_prediction_given_in_fit'] = False
        
        # compute fold indices
        fold_indices = []
        for dev_index, val_index in self.cv.split(X, y):
            fold_indices.append((dev_index, val_index))
        self.fold_indices_ = pd.DataFrame(fold_indices, columns=['train', 'test'])
        # save indices_df
        self.buffer.save_result(self.fold_indices_, 'fold_indices', data_type='indices')
        
        # prepare data
        X = pd.Series(X)
        y = pd.Series(y)
        #compute dataset independent ts features
        for transformer in self.ts_transformers:
            if isinstance(transformer, BaseDatasetIndependentTSTransformer):
                logging.info("Computing {} time series (data independent).".format(transformer.name))
                # map texts into time series
                time_series = transformer.transform(X)
                # extract and save time series features
                ts_features = self.feature_extractor.extract_features(time_series)
                self.buffer.save_feature_set(ts_features, method_name=transformer.name, train_test='train')
        # compute baseline and dataset dependent ts features
        for fold in self.fold_indices_.iterrows():
            # split train test
            X_train, X_test, y_train, _ = self.fold_train_test(fold, X, y)

            # compute baseline if needed
            if 'baseline' in self.feature_groups_:
                if baseline_prediction is not None:
                    pred_train = baseline_prediction.iloc[fold[1].train].reset_index(drop=True)
                    pred_test = baseline_prediction.iloc[fold[1].test].reset_index(drop=True)
                else:
                    # compute baseline method
                    logging.info("Computing baseline method for fold {}.".format(fold[0]))
                    self.baseline_classifier.fit(X_train, y_train, **baseline_clf_fit_kwargs)
                    baseline_pred_classes = self.get_classes(classes, self.baseline_classifier)
                    pred_train = pd.DataFrame(self.baseline_classifier.predict_proba(X_train, **baseline_clf_predict_proba_kwargs), columns=baseline_pred_classes)
                    pred_test = pd.DataFrame(self.baseline_classifier.predict_proba(X_test, **baseline_clf_predict_proba_kwargs), columns=baseline_pred_classes)

                    self.buffer.save_class(self.baseline_classifier, method_name='baseline', 
                                           class_name=self.baseline_classifier.__class__.__name__, fold_number=fold[0])
                    
                self.buffer.save_feature_set(pred_train, method_name='baseline', fold_number=fold[0], train_test='train')
                self.buffer.save_prediction(pred_train, method_name='baseline', fold_number=fold[0], train_test='train')
                self.buffer.save_feature_set(pred_test, method_name='baseline', fold_number=fold[0], train_test='test')
                self.buffer.save_prediction(pred_test, method_name='baseline', fold_number=fold[0], train_test='test')

            # compute time series features
            for transformer in self.ts_transformers:
                if isinstance(transformer, BaseDatasetDependentTSTransformer):
                    logging.info("Computing {} time series (data dependent) for fold {}.".format(transformer.name, fold[0]))
                    # map texts into time series
                    transformer.fit(X_train)
                    ts_train = transformer.transform(X_train)
                    ts_test = transformer.transform(X_test)
                    self.buffer.save_class(transformer, method_name=transformer.name, class_name=transformer.__class__.__name__, fold_number=fold[0]) # TODO test if this works with LocalBuffer
                    # extract and save time series features
                    ts_train_features = self.feature_extractor.extract_features(ts_train)
                    ts_test_features = self.feature_extractor.extract_features(ts_test)
                    self.buffer.save_feature_set(ts_train_features, method_name=transformer.name, fold_number=fold[0], train_test='train')
                    self.buffer.save_feature_set(ts_test_features, method_name=transformer.name, fold_number=fold[0], train_test='test')

        # first iteration - evaluate baseline or the first time series method
        cv_results_train = []
        cv_results_test = []
        for fold in self.fold_indices_.iterrows():
            # split train test
            _, _, y_train, y_test = self.fold_train_test(fold, X, y)

            if 'baseline' in self.feature_groups_:
                pred_train = self.buffer.read_prediction(method_name='baseline', fold_number=fold[0], train_test='train')
                pred_test = self.buffer.read_prediction(method_name='baseline', fold_number=fold[0], train_test='test')
            else:
                first_transformer = self.ts_transformers[0]
                self.feature_groups_.append(first_transformer.name)
                # get ts features
                if isinstance(first_transformer, BaseDatasetIndependentTSTransformer):
                    ts_features = self.buffer.read_feature_set(method_name=first_transformer.name, train_test='train')
                    X_train = ts_features.iloc[fold[1].train].reset_index(drop=True)
                    X_test = ts_features.iloc[fold[1].test].reset_index(drop=True)
                else:
                    X_train = self.buffer.read_feature_set(method_name=first_transformer.name, fold_number=fold[0], train_test='train')
                    X_test = self.buffer.read_feature_set(method_name=first_transformer.name, fold_number=fold[0], train_test='test')
                # select and save relevant features 
                relevant_features = self.feature_extractor.select_relevant_features(X_train)
                self.buffer.save_feature_relevance_table(relevant_features, method_name=first_transformer.name, fold_number=fold[0])
                X_train_relevant = X_train[relevant_features.feature]
                X_test_relevant = X_test[relevant_features.feature]
                # train and save meta-classifier
                self.meta_classifier.fit(X_train_relevant, y_train, **meta_clf_fit_kwargs)
                self.buffer.save_class(self.meta_classifier, method_name='meta_classifier', 
                                            class_name=self.meta_classifier.__class__.__name__, 
                                            fold_number=fold[0], suffix=first_transformer.name)
                meta_clf_pred_classes = self.get_classes(classes, self.meta_classifier)
                pred_train = pd.DataFrame(self.meta_classifier.predict_proba(X_train_relevant, **meta_clf_predict_proba_kwargs), columns=meta_clf_pred_classes)
                pred_test = pd.DataFrame(self.meta_classifier.predict_proba(X_test_relevant, **meta_clf_predict_proba_kwargs), columns=meta_clf_pred_classes)

            # evaluate
            cv_results_train.append(self.metric(y_train, pred_train))
            cv_results_test.append(self.metric(y_test, pred_test))
        # save evaluation scores
        evaluation_score = pd.DataFrame({
            'train': cv_results_train,
            'test': cv_results_test
        })
        self.buffer.save_evaluation_score(score_df=evaluation_score, method_name=self.feature_groups_[0])




################################################
        else:
            self.flags_['baseline_prediction_given_in_fit'] = False
            if self.baseline_classifier is not None: # if baseline_classifier is given
                # compute baseline method
                logging.info("Computing baseline method .")
                self.baseline_classifier.fit(X, y, **baseline_clf_fit_kwargs)
                pred = self.baseline_classifier.predict_proba(X, **baseline_clf_predict_proba_kwargs)
                if classes is None:
                    try:
                        classes = self.baseline_classifier.classes_
                    except AttributeError:
                        pass
                baseline_prediction = pd.DataFrame(data=pred, columns=classes)
                self.buffer.save_class(self.baseline_classifier, method_name='baseline', 
                                        class_name=self.baseline_classifier.__class__.__name__)
        
        # if at lease one of baseline_prediction or baseline_classifier is given so baseline 
        # predictions was computed
        if baseline_prediction is not None:
            self.buffer.save_feature_set(baseline_prediction, method_name='baseline', train_test='train')
            self.buffer.save_prediction(baseline_prediction, method_name='baseline', train_test='train')
            self.feature_groups_.append('baseline')
        
        # compute time series features
        for transformer in self.ts_transformers:
            logging.info("Computing {} time series.".format(transformer.name))
            # map texts into time series
            time_series = transformer.transform(X)
            # extract and save time series features
            ts_features = self.feature_extractor.extract_features(time_series)
            self.buffer.save_feature_set(ts_features, method_name=transformer.name, train_test='train')
            self.feature_groups_.append(transformer.name)

        # combine features
        X_combined = self.get_combined_features(fold_number=None, train_test='train', surfix=None)
        combined_name = "_".join(self.feature_groups_)

        # select and save relevant features 
        self.relevant_features_ = self.feature_extractor.select_relevant_features(X_combined)
        self.buffer.save_feature_relevance_table(self.relevant_features_, method_name=combined_name)
        X_relevant = X_combined[self.relevant_features_.feature]

        # train and save meta-classifier
        self.meta_classifier.fit(X_relevant, y, **meta_clf_fit_kwargs)
        self.buffer.save_class(self.meta_classifier, method_name='meta_classifier', 
                                    class_name=self.meta_classifier.__class__.__name__)
        
        return self


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