
# optimize avec GP + nouvelle arborescence (un dossier par modele)

# methods needed for the model : fit, predict

import os
import shutil
import pickle
from datetime import datetime
from time import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from skopt.space import Space
from skopt.sampler import Lhs
from skopt import gp_minimize
from skopt.plots import plot_convergence

from pathlib import Path


root_path = '/data/appli_PITSI/users/targe/FastML/'

folder_data = 'data'
folder_train = 'train'
folder_test = 'test'

folder_models = 'models'
folder_accuracies = 'accuracies'
folder_prediction = 'prediction'

folder_src = 'src'


class Model:

    """
    Model ML

    :param callable model: the model to use
    :param array X_train: the train data
    :param array X_test: the test data
    :param array y_train: the train target
    :param array y_test: the test target
    :param dict parameters: parameters of the model
    :param dict metrics: metrics to evaluate the model
    :param str name: name of the model (default value is the model's creation date

    >>> model = Model(RandomForestRegressor, X_train, X_test, y_train, y_test,
                      parameters={'n_estimator': 50}, metrics={r2_score: {}, mean_squared_error: {}})
    """

    def __init__(self, model: callable, X_train, X_test, y_train, y_test, parameters={}, metrics={}, name=''):

        self.model = model(**parameters)
        self.type = model  # sklearn object
        self.parameters = parameters
        self.metrics = metrics
        self.metrics_score = {}
        self.X_train = X_train
        self.y_train = _checkDim(y_train)
        self.X_test = X_test
        self.y_test = _checkDim(y_test)
        self.y_pred = None  # =model.predict(X_test)

        self.now = datetime.now().strftime("%y%m%d_%H%M%S")
        self.name = self.now if name == '' else name
        self.folder = os.path.join(self.name, "")

        Path(os.path.join(root_path, folder_models, self.folder, folder_accuracies)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(root_path, folder_models, self.folder, folder_prediction)).mkdir(parents=True, exist_ok=True)


    def fit(self, save=True, kwargs_fit={}):
        """
        Fit the model

        :param bool save: whether to save the model or not
        :param dict kwargs_fit: arguments of the fit method of the model
        """

        self.model.fit(self.X_train, self.y_train, **kwargs_fit)
        if save: self._save()


    def predict(self, metrics={}, save=True, write=True, kwargs_predict={}):
        """
        Test the model

        :param dict metrics: metrics to evaluate the model (if metric != {}, erase the metrics passed during initialization)
        :param bool save: whether to save the model or not
        :param bool write: whether to write information in text file or not
        :param dict kwargs_predict: arguments of the predict method of the model

        >>> model.predict(metrics={accuracy_score: {}, recall_score: {'average': 'macro'}}) # classification metrics
        """

        if metrics == {}:
            metrics = self.metrics
        else:
            self.metrics = metrics

        self.y_pred = self.model.predict(self.X_test, **kwargs_predict)
        self._measure(metrics=metrics, write=write)
        if save: self._save()


    def _measure(self, metrics={}, write=True):
        """
        Evaluate the model regarding the metrics

        :param dict metrics: metrics to evaluate the model
        :param bool write: whether to write information in text file or not
        """

        for metric, kwargs in metrics.items():
            self.metrics_score[metric.__name__] = metric(self.y_test, self.y_pred, **kwargs)

        if write:
            with open(os.path.join(root_path, folder_models, self.folder, folder_accuracies, 'accuracies.txt'), 'a') as file:
                for metric, kwargs in metrics.items():
                    file.write(metric.__name__ + ': ' + str(self.metrics_score[metric.__name__]) + '\n')
                file.write('\n'*2)


    def _save(self):
        """
        Save the model into a pickle file
        """

        pickle.dump(self, open(os.path.join(root_path, folder_models, self.folder, 'model.pkl'), 'wb'))


    def process(self, metrics={}, save=True, write=True, kwargs_fit={}, kwargs_predict={}):
        """
        Compute fit and predict

        :param dict metrics: metrics to evaluate the model
        :param bool save: whether to save the model or not
        :param bool write: whether to write information in text file or not
        :param dict kwargs_fit: arguments of the fit method of the model
        :param dict kwargs_predict: arguments of the predict method of the model
        """

        self.fit(save=False, **kwargs_fit)
        if metrics == {}: metrics = self.metrics
        self.predict(metrics=metrics, save=save, write=write, **kwargs_predict)


    def inference(self, X_pred=None, file=''):
        """
        Run inference from the model
        Use either an array or a csv file

        :param array X_pred: the input data
        :param str file: data file name
        """

        if file == '':
            # X_pred passed in argument
            pass
        else:
            X_pred = pd.read_csv(os.path.join(root_path, folder_data, folder_prediction, file))
        # Here, X_pred exists

        y_pred = self.model.predict(X_pred)
        df = pd.concat([pd.DataFrame(X_pred), pd.DataFrame(y_pred, index=X_pred.index, columns=[self.y_test.name + '_pred'])], axis=1)  #, ignore_index=True)
        now = datetime.now().strftime("%y%m%d_%H%M%S")
        df.to_csv(os.path.join(root_path, folder_models, self.folder, folder_prediction, now + '.txt'), index=False)


    def optimize(self, parameters_range, metric={}, n_lhs=1, n_calls=1, min_or_max='min', refit=False, verbose=True, write=True, plot_cvg=False) -> tuple:
        """
        Run Bayesian optimization (EGO) to find the best hyperparameters of the model

        :param dict parameters_range: parameters with range
        :param dict metric: metric to optimize
        :param int n_lhs: numbers of initial points. Use LHS algorithm to find them
        :param int n_calls: numbers of calls to objective function. n_calls-n_lhs is the number of points computed by Bayesian optimization
        :param bool refit: whether to refit the model with the best parameters found or not
        :param str min_or_max: whether to find the minimum or maximum of the objective function
        :param bool verbose: whether to print information during optimization or not (advised for long optimization runs
        :param bool write: whether to write information in text file or not
        :param bool plot_cvg: whether to plot the convergence plot or not

        :return: best parameters, best score, duration of optimization
        :rtype: tuple

        >>> model.optimize(parameters_range={'n_estimators': [10, 20, 30, 40], 'min_samples_split': [2,4,6,8,10]},
                    n_lhs=10, n_calls=20, metric={r2_score: {}}, min_or_max='max', refit=True)
        """

        # parameter_range = {'param1': [1,2,3], 'param2': [15,20,25], ...}
        # discrete parameter : list of possibilities
        # continous parameter : bounds (float) of the interval

        assert len(metric) == 1, "Use one metric"
        assert min_or_max in ['min', 'max'], "min_or_max argument is not correct. use 'min' or 'max'"

        def f_obj(param: list) -> float:
            param_dico = dict(zip(parameters_name, param))
            model = Model(self.type, self.X_train, self.X_test, self.y_train, self.y_test, parameters=param_dico)
            model.process(metrics=metric, save=False, write=False)
            score = model.metrics_score[next(iter(metric)).__name__]

            shutil.rmtree(os.path.join(root_path, folder_models, model.folder))

            if min_or_max == 'min':
                return score
            elif min_or_max == 'max':
                return -score
            else:
                pass

        t0 = time()
        parameters_name = list(parameters_range.keys())

        results = gp_minimize(f_obj,                  # the function to minimize
                              list(parameters_range.values()),      # the bounds on each dimension of x
                              acq_func="EI",      # the acquisition function
                              n_calls=n_calls,         # the number of evaluations of f
                              n_random_starts=n_lhs,  # the number of random initialization points
                              random_state=None,   # the random seed
                              initial_point_generator='lhs',
                              n_jobs=-1,
                              verbose=verbose)


        t1 = round(time() - t0, 5)  # duration GP
        msg1 = "Duration of optimization: " + str(t1) + ' s'
        print(msg1)

        if plot_cvg: plot_convergence(results)

        best_param = results.x
        combinations = results.x_iters
        if min_or_max == 'min':
            scores = results.func_vals
            best_score = results.fun
        elif min_or_max == 'max':
            scores = -results.func_vals
            best_score = -results.fun
        else:
            pass

        # Sorting of scores and combinations regarding scores
        scores_combis = zip(scores, combinations)
        scores_combis = sorted(scores_combis, key=lambda x: x[0], reverse=True)
        scores_sort, combinations_sort = list(zip(*scores_combis))

        msg2 = "best score: " + str(best_score) + "\nbest parameters found: " + str(best_param)
        print(msg2)
        if write:
            with open(os.path.join(root_path, folder_models, self.folder, folder_accuracies + 'optimization.txt'), 'a') as file:
                file.write(str(parameters_name) + '\n'*2)
                file.write(msg1 + '\n')
                file.write(msg2 + '\n'*2)
                file.write(str(n_calls) + ' combinations with associated score (n_lhs=' + str(n_lhs) + ', n_ego=' + str(n_calls-n_lhs) + ')\n')
                for i in range(len(combinations_sort)):
                    file.write(str(combinations_sort[i]) + ': ' + str(scores_sort[i]) + '\n')
                file.write('\n'*2)

        if refit:
            self.model = self.type(**dict(zip(parameters_name, best_param)))
            self.process(metrics=self.metrics)
            self.parameters = best_param
        else:
            pass

        return best_param, best_score, t1



def getData(inputs: list, output: str, df_or_file, test_size=0.0, random_state=None):
    """
    Get data from either file or array

    :param list inputs: inputs name
    :param str output: target name
    :param df_or_file: array or csv file from which get the data
    :param float test_size: percentage of data used to test the model. Use test_size=0.0 to not split the data into train/test
    :param random_state: seed
    :return: X, y if test_size=0.0 else X_train, X_test, y_train, y_test
    :rtype: tuple of array

    >>> X_train, X_test, y_train, y_test = getData(inputs=['a', 'b'], output='c', df_or_file=df, test_size=0.30)

    >>> X, y = getData(inputs=['a', 'b'], output='c', df_or_file='data.txt', test_size=0.0)
    """
    if isinstance(df_or_file, str):
        return _getData_file(inputs, output, file=df_or_file, test_size=test_size, random_state=random_state)
    elif isinstance(df_or_file, pd.DataFrame):
        return _getData_df(inputs, output, df=df_or_file, test_size=test_size, random_state=random_state)


def _getData_df(inputs: list, output: str, df: pd.DataFrame, test_size=0.0, random_state=None):
    if test_size <= 1e-8:
        return df[inputs], df[output]
    else:
        return train_test_split(df[inputs], df[output], test_size=test_size, random_state=random_state)


def _getData_file(inputs: list, output: str, file: str, test_size=0.0, random_state=None):
    df = pd.read_csv(os.path.join(root_path, folder_data, file))
    return _getData_df(inputs, output, df, test_size, random_state)


def _checkDim(y):
    if y.ndim != 1:
        return np.ravel(y)
    else:
        return y



def getModel(file_name: str) -> Model:
    """
    Get the model from pickle file
    
    :param str file_name: pickle file name
    :return: The model
    :rtype: Model
    """
    return pickle.load(open(os.path.join(root_path, folder_models, file_name, 'model.pkl'), 'rb'))


def prediction(model_name: str, file_pred='', X_pred=None):
    """
    Run inference from either array or csv file

    :param str model_name: pickle file name
    :param str file_pred: data file name
    :param array X_pred: input data

    >>> prediction(model_name=model.name, X_pred=X_pred)

    >>> prediction(model_name=model.name, file_pred='data.txt')
    """
    model = getModel(file_name=model_name)
    model.inference(X_pred=X_pred, file=file_pred)

