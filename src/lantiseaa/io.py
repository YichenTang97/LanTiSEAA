import os
import errno
import pandas as pd
from sklearn.externals import joblib

class IO():

    def __init__(self, project_path=None, subfolder=''):
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
            # Assume this file is under '../LanTiSEAA/src/lantiseaa', in default set '../LanTiSEAA' as the project path
            self.project_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '../..')
                                )
        else:
            if os.path.isdir(project_path):
                self.project_path = project_path
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), project_path)
        
        self.subfolder = subfolder


    def project(self, filename, folder=''):
        """Return absolute path of file within folder hierarchy"""
        return os.path.join(self.project_path, folder, filename)


    def data(self, filename, subfolder=''):
        return self.project(filename, folder=os.path.join('data', subfolder))


    def results(self, filename, subfolder=''):
        return self.project(filename, folder=os.path.join('results', subfolder))


    def figures(self, filename, subfolder=''):
        return self.project(filename, folder=os.path.join('figures', subfolder))

    
    def classes(self, filename, subfolder=''):
        return self.project(filename, folder=os.path.join('classes', subfolder))



    def save_hdf_result(self, df, key, data_type='data', data_name=None, fold_number=None, train_test=None, surfix=None, use_default_subfolder=True, subfolder=''):
        if use_default_subfolder:
            subfolder = self.subfolder

        if data_name is not None:
            data_type += '__{}'.format(data_name)

        if fold_number is not None:
            key += '__fold_{}'.format(fold_number)
        if train_test is not None:
            key += '__{}'.format(train_test)
        if surfix is not None:
            key += '__{}'.format(surfix)

        df.to_hdf(self.results('{}.hdf'.format(data_type), subfolder), key)


    def read_hdf_result(self, key, data_type='data', data_name=None, fold_number=None, train_test=None, surfix=None, use_default_subfolder=True, subfolder=''):
        if use_default_subfolder:
            subfolder = self.subfolder

        if data_name is not None:
            data_type += '__{}'.format(data_name)

        if fold_number is not None:
            key += '__fold_{}'.format(fold_number)
        if train_test is not None:
            key += '__{}'.format(train_test)
        if surfix is not None:
            key += '__{}'.format(surfix)
        
        return pd.read_hdf(self.results('{}.hdf'.format(data_type), subfolder), key)



    def save_feature_set(self, df, method_name, key='X', fold_number=None, train_test=None, surfix=None, use_default_subfolder=True, subfolder=''):
        self.save_hdf_result(df, key, data_type='feature', data_name=method_name, fold_number=fold_number, train_test=train_test, surfix=surfix, 
                             use_default_subfolder=use_default_subfolder, subfolder=subfolder)


    def read_feature_set(self, method_name, key='X', fold_number=None, train_test=None, surfix=None, use_default_subfolder=True, subfolder=''):
        return self.read_hdf_result(key, data_type='feature', data_name=method_name, fold_number=fold_number, train_test=train_test, surfix=surfix, 
                                    use_default_subfolder=use_default_subfolder, subfolder=subfolder)



    def save_intermediate_result(self, df, method_name, key, fold_number=None, train_test=None, surfix=None, use_default_subfolder=True, subfolder=''):
        self.save_hdf_result(df, key, data_type='intermediate_result', data_name=method_name, fold_number=fold_number, train_test=train_test, 
                             surfix=surfix, use_default_subfolder=use_default_subfolder, subfolder=subfolder)


    def read_intermediate_result(self, method_name, key, fold_number=None, train_test=None, surfix=None, use_default_subfolder=True, subfolder=''):
        return self.read_hdf_result(key, data_type='intermediate_result', data_name=method_name, fold_number=fold_number, train_test=train_test, 
                                    surfix=surfix, use_default_subfolder=use_default_subfolder, subfolder=subfolder)



    def save_feature_relevance_table(self, relevance_table, method_name, key='feature_relevance_table', fold_number=None, surfix=None, use_default_subfolder=True, subfolder=''):
        self.save_hdf_result(relevance_table, key, data_type='result', data_name=method_name, fold_number=fold_number, train_test=None, surfix=surfix, 
                             use_default_subfolder=use_default_subfolder, subfolder=subfolder)


    def read_feature_relevance_table(self, method_name, key='feature_relevance_table', fold_number=None, surfix=None, use_default_subfolder=True, subfolder=''):
        return self.read_hdf_result(key, data_type='result', data_name=method_name, fold_number=fold_number, train_test=None, surfix=surfix, 
                                    use_default_subfolder=use_default_subfolder, subfolder=subfolder)



    def save_prediction(self, pred_df, method_name, fold_number, train_test, key='prediction', surfix=None, use_default_subfolder=True, subfolder=''):
        self.save_hdf_result(pred_df, key, data_type='result', data_name=method_name, fold_number=fold_number, train_test=train_test, surfix=surfix, 
                             use_default_subfolder=use_default_subfolder, subfolder=subfolder)


    def read_prediction(self, method_name, fold_number, train_test, key='prediction', surfix=None, use_default_subfolder=True, subfolder=''):
        return self.read_hdf_result(key, data_type='result', data_name=method_name, fold_number=fold_number, train_test=train_test, surfix=surfix, 
                                    use_default_subfolder=use_default_subfolder, subfolder=subfolder)



    def save_class_attribute(self, attribute_df, method_name, class_name, attribute_name, fold_number=None, surfix=None, use_default_subfolder=True, subfolder=''):
        key = '{}__{}'.format(class_name, attribute_name)
        self.save_hdf_result(attribute_df, key, data_type='class_attribute', data_name=method_name, fold_number=fold_number, train_test=None, surfix=surfix, 
                             use_default_subfolder=use_default_subfolder, subfolder=subfolder)


    def read_class_attribute(self, method_name, class_name, attribute_name, fold_number=None, surfix=None, use_default_subfolder=True, subfolder=''):
        key = '{}__{}'.format(class_name, attribute_name)
        return self.read_hdf_result(key, data_type='class_attribute', data_name=method_name, fold_number=fold_number, train_test=None, surfix=surfix, 
                                    use_default_subfolder=use_default_subfolder, subfolder=subfolder)



    def save_evaluation_score(self, score_df, method_name, surfix=None, use_default_subfolder=True, subfolder=''):
        key = 'cv_scores__{}'.format(method_name)
        self.save_hdf_result(score_df, key, data_type='score', data_name=None, fold_number=None, 
                             train_test=None, surfix=surfix, use_default_subfolder=use_default_subfolder, subfolder=subfolder)


    def read_evaluation_score(self, method_name, surfix=None, use_default_subfolder=True, subfolder=''):
        key = 'cv_scores__{}'.format(method_name)
        return self.read_hdf_result(key, data_type='score', data_name=None, fold_number=None, 
                                    train_test=None, surfix=surfix, use_default_subfolder=use_default_subfolder, subfolder=subfolder)



    def save_class(self, c, method_name, class_name, subclass_name=None, fold_number=None, surfix=None, use_default_subfolder=True, subfolder=''):
        if use_default_subfolder:
            subfolder = self.subfolder

        filename = '{}__{}'.format(method_name, class_name)

        if subclass_name is not None:
            filename += '__{}'.format(subclass_name)
        if fold_number is not None:
            filename += '__fold_{}'.format(fold_number)
        if surfix is not None:
            filename += '__{}'.format(surfix)

        joblib.dump(c, self.classes(filename, subfolder))


    def read_class(self, method_name, class_name, subclass_name=None, fold_number=None, surfix=None, use_default_subfolder=True, subfolder=''):
        if use_default_subfolder:
            subfolder = self.subfolder

        filename = '{}__{}'.format(method_name, class_name)

        if subclass_name is not None:
            filename += '__{}'.format(subclass_name)
        if fold_number is not None:
            filename += '__fold_{}'.format(fold_number)
        if surfix is not None:
            filename += '__{}'.format(surfix)

        return joblib.load(self.classes(filename, subfolder))
