import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table, combine_relevance_tables
from tsfresh.feature_selection.benjamini_hochberg_test import benjamini_hochberg_test


class BaseTSFeatureExtractor(ABC):
    '''Base class for TSFeatureExtractor classes which can extract features from time series and select relevant features'''

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def extract_features(self, ts):
        '''Extract features from the given time series ts'''
        pass

    @abstractmethod
    def select_relevant_features(self, X, y):
        '''Select relevant features from the given features X using y'''
        pass


class TsfreshTSFeatureExtractor(BaseTSFeatureExtractor):
    
    def __init__(self, fdr_level=0.001):
        '''Construct a TsfreshTSFeatureExtractor

        Parameters
        ----------
        fdr_level : float, optional
            The theoretical expected percentage of irrelevant features among all selected features
        '''
        super().__init__()
        self.fdr_level = fdr_level


    def extract_features(self, ts, column_id='id', impute_function=impute, default_fc_parameters=ComprehensiveFCParameters(), n_jobs=-1, show_warnings=False, profile=False):
        '''Extract all possible features from ts using tsfresh's extract_features method'''
        return extract_features(ts, column_id=column_id, impute_function=impute_function,
                                default_fc_parameters=default_fc_parameters,
                                n_jobs=n_jobs, show_warnings=show_warnings, profile=profile)

    def select_relevant_features(self, X, y):
        '''Select statistically significant features while computing the relevance of these features.'''
        # calculate relevance tables for each binary class pair
        relevance_tables = list()
        for label in np.unique(y):
            y_binary = (y == label)
            relevance_tables.append((label, calculate_relevance_table(X, y_binary, fdr_level=self.fdr_level)))

        # concatenate relevance tables
        relevance_table_concat = pd.concat([table for (lable, table) in relevance_tables])

        # perform benjamini hochberg test
        relevance_table_benjamini = benjamini_hochberg_test(relevance_table_concat, hypotheses_independent=False, fdr_level=self.fdr_level)

        # remove irrelevant features from the table
        relevance_table_benjamini = relevance_table_benjamini[relevance_table_benjamini.relevant == True]

        # select features occurred at least twice in the table
        feature_occurrences = relevance_table_benjamini.feature.value_counts()
        relevant_features = feature_occurrences[feature_occurrences == len(y.unique())].index.values
        occurrence_counts = feature_occurrences.value_counts()
        for i in range(1, 4):
            try:
                logging.info(
                    'Number of features occurred {} time(s) in the relevant features selected after benjamini hochberg test: {}'
                        .format(i, occurrence_counts[i]))
            except (KeyError, IndexError):  # when there is no feature occur the corresponding number of times
                pass
        # build final relevance table
        relevance_table_final = pd.DataFrame({
            'feature': relevant_features,
            'p_value': [relevance_table_benjamini.loc[f].p_value.max() for f in relevant_features],
            'occurrence': [feature_occurrences[f] for f in relevant_features]
        }).sort_values(by=['p_value', 'occurrence']).reset_index(drop=True)
        logging.info("Number of relevant features for all classes: {}/{}".format(relevance_table_final.shape[0], X.shape[1]))

        return relevance_table_final
