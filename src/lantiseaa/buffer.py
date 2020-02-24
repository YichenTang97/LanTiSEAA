import os
import csv
import copy
import errno
import pandas as pd
import pymc3 as pm
from sklearn.externals import joblib
from abc import ABC, abstractmethod


class Buffer(ABC):

    def __init__(self):
        super().__init__()


    @abstractmethod
    def save_record(self, genre, type_name, name, key, fold_number, train_test, surfix):
        pass

    @abstractmethod
    def read_records(self):
        pass
    
    @abstractmethod
    def save_result(self, df, key, data_type='data', data_name=None, fold_number=None, train_test=None, surfix=None):
        pass

    @abstractmethod
    def read_result(self, key, data_type='data', data_name=None, fold_number=None, train_test=None, surfix=None):
        pass

    @abstractmethod
    def save_class(self, c, method_name, class_name, subclass_name=None, fold_number=None, surfix=None):
        pass

    @abstractmethod
    def read_class(self, method_name, class_name, subclass_name=None, fold_number=None, surfix=None):
        pass
    
    @abstractmethod
    def save_bayesian_estimation_trace(self, trace, group1_name, group2_name, surfix=None):
        pass

    @abstractmethod
    def read_bayesian_estimation_trace(self, group1_name, group2_name, surfix=None):
        pass


    def save_feature_set(self, df, method_name, key='X', fold_number=None, train_test=None, surfix=None):
        self.save_result(df, key, data_type='feature', data_name=method_name, fold_number=fold_number, train_test=train_test, surfix=surfix)


    def read_feature_set(self, method_name, key='X', fold_number=None, train_test=None, surfix=None):
        return self.read_result(key, data_type='feature', data_name=method_name, fold_number=fold_number, train_test=train_test, surfix=surfix)


    def save_intermediate_result(self, df, method_name, key, fold_number=None, train_test=None, surfix=None):
        self.save_result(df, key, data_type='intermediate_result', data_name=method_name, fold_number=fold_number, train_test=train_test, 
                         surfix=surfix)


    def read_intermediate_result(self, method_name, key, fold_number=None, train_test=None, surfix=None):
        return self.read_result(key, data_type='intermediate_result', data_name=method_name, fold_number=fold_number, train_test=train_test, 
                                    surfix=surfix)


    def save_feature_relevance_table(self, relevance_table, method_name, key='feature_relevance_table', fold_number=None, surfix=None):
        self.save_result(relevance_table, key, data_type='result', data_name=method_name, fold_number=fold_number, train_test=None, surfix=surfix)


    def read_feature_relevance_table(self, method_name, key='feature_relevance_table', fold_number=None, surfix=None):
        return self.read_result(key, data_type='result', data_name=method_name, fold_number=fold_number, train_test=None, surfix=surfix)


    def save_prediction(self, pred_df, method_name, key='prediction', fold_number=None, train_test=None, surfix=None):
        self.save_result(pred_df, key, data_type='result', data_name=method_name, fold_number=fold_number, train_test=train_test, surfix=surfix)


    def read_prediction(self, method_name, key='prediction', fold_number=None, train_test=None, surfix=None):
        return self.read_result(key, data_type='result', data_name=method_name, fold_number=fold_number, train_test=train_test, surfix=surfix)


    def save_class_attribute(self, attribute_df, method_name, class_name, attribute_name, fold_number=None, surfix=None):
        key = '{}__{}'.format(class_name, attribute_name)
        self.save_result(attribute_df, key, data_type='class_attribute', data_name=method_name, fold_number=fold_number, train_test=None, surfix=surfix)


    def read_class_attribute(self, method_name, class_name, attribute_name, fold_number=None, surfix=None):
        key = '{}__{}'.format(class_name, attribute_name)
        return self.read_result(key, data_type='class_attribute', data_name=method_name, fold_number=fold_number, train_test=None, surfix=surfix)


    def save_evaluation_score(self, score_df, method_name, surfix=None):
        key = 'cv_scores__{}'.format(method_name)
        self.save_result(score_df, key, data_type='score', data_name=None, fold_number=None, 
                             train_test=None, surfix=surfix)


    def read_evaluation_score(self, method_name, surfix=None):
        key = 'cv_scores__{}'.format(method_name)
        return self.read_result(key, data_type='score', data_name=None, fold_number=None, 
                                    train_test=None, surfix=surfix)



class LocalBuffer(Buffer):

    def __init__(self, project_path=None, subfolder=''):
        super().__init__()
        self.reset_path(project_path, subfolder)


    def reset_path(self, project_path=None, subfolder=''):
        """Reset the default project path and the subfolder name for saving/reading data and results

        If the project_path is not given, the 'LanTiSEAA' folder will be set as the project_path in default.
        If the subfolder is not given, all savings/readings will be done from the root 'data', 'results', 
        and 'results/figs' folders in default (this can be override on each save/read method).

        Parameters
        ----------
        project_path : str, optional
            The root path of the project where the data and results will be stored in (default is None).
        
        subfolder : str, optional
            The subfolder under the root 'data', 'results', and 'results/figs' folders where the data and 
            results will be saved to (default is an empty string '').

        """
        if project_path == None:
            # Assume this file is under '../workspace/LanTiSEAA/src/lantiseaa', in default set '../workspace' as the project path
            self.project_path_ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '../../..')
                                )
        else:
            if os.path.isdir(project_path):
                self.project_path_ = project_path
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), project_path)
        
        self.subfolder_ = subfolder


    def project(self, filename, folder=''):
        """Return absolute path of file within folder hierarchy"""
        return os.path.join(self.project_path_, folder, filename)


    def data(self, filename, subfolder=''):
        return self.project(filename, folder=os.path.join('data', subfolder))


    def results(self, filename, subfolder=''):
        return self.project(filename, folder=os.path.join('results', subfolder))


    def figures(self, filename, subfolder=''):
        return self.project(filename, folder=os.path.join('figures', subfolder))

    
    def classes(self, filename, subfolder=''):
        return self.project(filename, folder=os.path.join('classes', subfolder))


    def save_record(self, genre, type_name, name, key, fold_number, train_test, surfix):
        records = self.results("records.csv", subfolder=self.subfolder_)
        if not os.path.exists(records):
            with open(records, 'w', newline='') as csvfile:
                csv.writer(csvfile, delimiter=',', quotechar='|').writerow(['Genre', 'Type/Method', 'Data/Class', 'Key/Subclass', 'Fold', 'Train/Test', 'Surfix'])

        with open(records, 'a', newline='') as csvfile:
            csv.writer(csvfile, delimiter=',', quotechar='|').writerow([genre, type_name, name, key, fold_number, train_test, surfix])


    def read_records(self):
        records = self.results("records.csv", subfolder=self.subfolder_)
        if not os.path.exists(records):
            return None
        else:
            return pd.read_csv(records).sort_values(by=['Genre', 'Type/Method', 'Data/Class', 'Key/Subclass', 'Fold', 'Train/Test', 'Surfix']).reset_index(drop=True)


    def save_result(self, df, key, data_type='data', data_name=None, fold_number=None, train_test=None, surfix=None):
        # create the directory if not exist
        if not os.path.isdir(self.results('', self.subfolder_)):
            os.mkdir(self.results('', self.subfolder_))

        if data_name is not None:
            data_type += '__{}'.format(data_name)

        if fold_number is not None:
            key += '__fold_{}'.format(fold_number)
        if train_test is not None:
            key += '__{}'.format(train_test)
        if surfix is not None:
            key += '__{}'.format(surfix)

        df.to_hdf(self.results('{}.hdf'.format(data_type), self.subfolder_), key)
        self.save_record(genre='result', type_name=data_type, name=data_name, key=key, fold_number=fold_number, train_test=train_test, surfix=surfix)


    def read_result(self, key, data_type='data', data_name=None, fold_number=None, train_test=None, surfix=None):
        if data_name is not None:
            data_type += '__{}'.format(data_name)

        if fold_number is not None:
            key += '__fold_{}'.format(fold_number)
        if train_test is not None:
            key += '__{}'.format(train_test)
        if surfix is not None:
            key += '__{}'.format(surfix)
        
        return pd.read_hdf(self.results('{}.hdf'.format(data_type), self.subfolder_), key)


    def save_class(self, c, method_name, class_name, subclass_name=None, fold_number=None, surfix=None):
        # create the directory if not exist
        if not os.path.isdir(self.classes('', self.subfolder_)):
            os.mkdir(self.classes('', self.subfolder_))
        
        filename = '{}__{}'.format(method_name, class_name)

        if subclass_name is not None:
            filename += '__{}'.format(subclass_name)
        if fold_number is not None:
            filename += '__fold_{}'.format(fold_number)
        if surfix is not None:
            filename += '__{}'.format(surfix)

        joblib.dump(c, self.classes(filename, self.subfolder_))
        self.save_record(genre='class', type_name=method_name, name=class_name, key=subclass_name, fold_number=fold_number, train_test=None, surfix=surfix)


    def read_class(self, method_name, class_name, subclass_name=None, fold_number=None, surfix=None):
        filename = '{}__{}'.format(method_name, class_name)

        if subclass_name is not None:
            filename += '__{}'.format(subclass_name)
        if fold_number is not None:
            filename += '__fold_{}'.format(fold_number)
        if surfix is not None:
            filename += '__{}'.format(surfix)

        return joblib.load(self.classes(filename, self.subfolder_))

    
    def save_bayesian_estimation_trace(self, trace, group1_name, group2_name, surfix=None):
        filename = 'traces__{}__{}'.format(group1_name, group2_name)

        if surfix is not None:
            filename += '__{}'.format(surfix)
        
        joblib.dump(trace, self.classes(filename, self.subfolder_))
        self.save_record(genre='traces', type_name=filename, name=None, key=None, fold_number=None, train_test=None, surfix=surfix)


    def read_bayesian_estimation_trace(self, group1_name, group2_name, surfix=None):
        filename = 'traces__{}__{}'.format(group1_name, group2_name)

        if surfix is not None:
            filename += '__{}'.format(surfix)
        
        return joblib.load(self.classes(filename, self.subfolder_))



class MemoryBuffer(Buffer):

    def __init__(self):
        super().__init__()
        self.records_ = None
        self.results_ = {}
        self.classes_ = {}
        self.traces_ = {}


    def save_record(self, genre, type_name, name, key, fold_number, train_test, surfix):
        new_record = pd.DataFrame([[genre, type_name, name, key, fold_number, train_test, surfix]], 
                                  columns=['Genre', 'Type/Method', 'Data/Class', 'Key/Subclass', 'Fold', 'Train/Test', 'Surfix'])
        if self.records_ is None:
            self.records_ = new_record
        else:
            self.records_.append(new_record, ignore_index=True)


    def read_records(self):
        return self.records_.sort_values(by=['Genre', 'Type/Method', 'Data/Class', 'Key/Subclass', 'Fold', 'Train/Test', 'Surfix']).reset_index(drop=True)
    

    def save_result(self, df, key, data_type='data', data_name=None, fold_number=None, train_test=None, surfix=None):
        if data_name is not None:
            data_type += '__{}'.format(data_name)
        if data_type not in self.results_.keys():
            self.results_.update({data_type: {}})

        if fold_number is not None:
            key += '__fold_{}'.format(fold_number)
        if train_test is not None:
            key += '__{}'.format(train_test)
        if surfix is not None:
            key += '__{}'.format(surfix)
        
        self.results_[data_type].update({key: df})
        self.save_record(genre='result', type_name=data_type, name=data_name, key=key, fold_number=fold_number, train_test=train_test, surfix=surfix)


    def read_result(self, key, data_type='data', data_name=None, fold_number=None, train_test=None, surfix=None):
        if data_name is not None:
            data_type += '__{}'.format(data_name)

        if fold_number is not None:
            key += '__fold_{}'.format(fold_number)
        if train_test is not None:
            key += '__{}'.format(train_test)
        if surfix is not None:
            key += '__{}'.format(surfix)
        
        return self.results_[data_type][key]


    def save_class(self, c, method_name, class_name, subclass_name=None, fold_number=None, surfix=None):
        key = '{}__{}'.format(method_name, class_name)

        if subclass_name is not None:
            key += '__{}'.format(subclass_name)
        if fold_number is not None:
            key += '__fold_{}'.format(fold_number)
        if surfix is not None:
            key += '__{}'.format(surfix)

        self.classes_.update({key: copy.deepcopy(c)})
        self.save_record(genre='class', type_name=method_name, name=class_name, key=subclass_name, fold_number=fold_number, train_test=None, surfix=surfix)


    def read_class(self, method_name, class_name, subclass_name=None, fold_number=None, surfix=None):
        key = '{}__{}'.format(method_name, class_name)

        if subclass_name is not None:
            key += '__{}'.format(subclass_name)
        if fold_number is not None:
            key += '__fold_{}'.format(fold_number)
        if surfix is not None:
            key += '__{}'.format(surfix)

        return self.classes_[key]


    def save_bayesian_estimation_trace(self, trace, group1_name, group2_name, surfix=None):
        key = '{}__{}'.format(group1_name, group2_name)

        if surfix is not None:
            key += '__{}'.format(surfix)
        
        self.traces_.update({key: copy.deepcopy(trace)})
        self.save_record(genre='traces', type_name=filename, name=None, key=None, fold_number=None, train_test=None, surfix=surfix)


    def read_bayesian_estimation_trace(self, group1_name, group2_name, surfix=None):
        key = '{}__{}'.format(group1_name, group2_name)

        if surfix is not None:
            key += '__{}'.format(surfix)
        
        return self.traces_[key]
