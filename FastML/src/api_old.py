
# optimize avec LHS seulement, utilise l'ancienne arborescence

# methods needed for the model : fit, predict

import pickle
from datetime import datetime
from time import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from skopt.space import Space
from skopt.sampler import Lhs


root_path = '/data/appli_PITSI/users/targe/FastML/'

folder_data = 'data/'
folder_train = 'train/'
folder_test = 'test/'

folder_results = 'results/'
folder_models = 'models/'
folder_accuracies = 'accuracies/'
folder_prediction = 'prediction/'

folder_src = 'src/'


class Model:

    def __init__(self, model, X_train, X_test, y_train, y_test, parameters={}, name=''):

        self.model = model(**parameters)
        self.type = model  # sklearn object
        self.parameters = parameters
        self.metrics = {}
        self.X_train = X_train
        self.y_train = _checkDim(y_train)
        self.X_test = X_test
        self.y_test = _checkDim(y_test)
        self.y_pred = None  # =model.predict(X_test)

        self.now = datetime.now().strftime("%y%m%d_%H%M%S")
        self.name = self.now if name == '' else name


    def fit(self, save=True, *args):
        self.model.fit(self.X_train, self.y_train, *args)
        if save: self._save()


    def predict(self, metrics={}, save=True, write=True, *args):
        self.y_pred = self.model.predict(self.X_test, *args)
        self._measure(metrics=metrics, write=write)
        if save: self._save()


    def _measure(self, metrics={}, write=True):
        for metric, kwargs in metrics.items():
            self.metrics[metric.__name__] = metric(self.y_test, self.y_pred, **kwargs)

        if write:
            with open(root_path + folder_results + folder_accuracies + self.name + '.txt', 'a') as file:
                for metric, kwargs in metrics.items():
                    file.write(metric.__name__ + ': ' + str(self.metrics[metric.__name__]) + '\n')

                file.write('\n'*2)


    def _save(self):
        pickle.dump(self, open(root_path + folder_results + folder_models + self.name + '.pkl', 'wb'))


    def process(self, metrics={}, save=True, write=True):
        self.fit(save=False)
        self.predict(metrics=metrics, save=save, write=write)


    def inference(self, X_pred=None, file=''):
        if file == '':
            # X_pred passed in argument
            pass
        else:
            X_pred = pd.read_csv(root_path + folder_data + folder_prediction + file)
        # Here, X_pred exists

        y_pred = self.model.predict(X_pred)
        df = pd.concat([pd.DataFrame(X_pred), pd.DataFrame(y_pred, index=X_pred.index, columns=[self.y_test.name])], axis=1)  #, ignore_index=True)
        now = datetime.now().strftime("%y%m%d_%H%M%S")
        df.to_csv(root_path + folder_results + folder_prediction + self.name + '__' + now + '.txt', index=False)


    def optimize(self, parameters_range, n_lhs, metric={}, n_ego=0, refit=False, min_or_max='min', verbose=True) -> dict:
        # parameter_range = {'param1': [1,2,3], 'param2': [15,20,25], ...}
        # discrete parameter : list of possibilities
        # continous parameter : bounds (float) of the interval
        # TODO: LHS + EGO

        assert len(metric) == 1, "Use one metric"
        assert min_or_max in ['min', 'max'], "min_or_max argument is not correct. use 'min' or 'max'"

        rnd = 5
        t0 = time()
        parameters_name = list(parameters_range.keys())

        space = Space(list(parameters_range.values()))
        lhs = Lhs(criterion="maximin", iterations=n_lhs*20)
        combinations = lhs.generate(space.dimensions, n_lhs)

        unique = set(map(tuple, combinations))
        combinations = list(unique)

        t1 = time()  # duration LHS
        dt1 = round(t1 - t0, rnd)
        msg1 = "Duration of LHS algorithm: " + str(dt1) + ' s'
        if verbose: print(msg1)

        scores = []
        for combination in combinations:
            parameters = dict(zip(parameters_name, combination))
            model = Model(self.type, X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test, parameters=parameters)
            model.process(save=False, write=False, metrics=metric)
            scores.append(model.metrics[next(iter(metric)).__name__])

        t2 = time()  # duration model creation + fitting + testing
        dt2 = round(t2 - t1, rnd)
        msg2 = "Duration of model creation + fitting + testing: " + str(dt2) + ' s'
        if verbose: print(msg2)

        #####
        # EGO
        #####

        t3 = time()  # duration EGO
        dt3 = round(t3 - t2, rnd)
        msg3 = "Duration of EGO algorithm: " + str(dt3) + ' s'
        if verbose: print(msg3)

        if min_or_max == 'min':
            best_score = min(scores)
            best_param = combinations[np.argmin(scores)]
        elif min_or_max == 'max':
            best_score = max(scores)
            best_param = combinations[np.argmax(scores)]
        else:
            pass

        scores_combis = zip(scores, combinations)
        scores_combis = sorted(scores_combis, key=lambda x: x[0], reverse=True)
        scores_sort, combinations_sort = list(zip(*scores_combis))

        msg = "best score: " + str(best_score) + "\nbest parameters found: " + str(best_param)
        if verbose: print(msg)
        dttot = round(t3-t0, rnd)
        with open(root_path + folder_results + folder_accuracies + self.name + '_optimize ' + '.txt', 'a') as file:
            file.write(str(parameters_name) + '\n'*2)
            file.write(msg1 + '\n' + msg2 + '\n' + msg3 + '\n')
            file.write("Total duration: " + str(dttot) + ' s\n')
            file.write(msg + '\n'*2)
            file.write(str(n_lhs) +' combinations with associated score\n')
            for i in range(len(combinations_sort)):
                file.write(str(combinations_sort[i]) + ': ' + str(scores_sort[i]) + '\n')
            file.write('\n'*2)

        if refit:
            self.model = model(**best_param)
            self.process(metrics=self.metrics)
        else:
            pass

        return best_param, best_score, dttot





def getData(inputs: list, output: str, df_or_file, test_size=0.0, random_state=None):
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
    df = pd.read_csv(root_path + folder_data + file)
    return _getData_df(inputs, output, df, test_size, random_state)


def _checkDim(y):
    if y.ndim != 1:
        return np.ravel(y)
    else:
        return y



def getModel(file_name: str) -> Model:
    return pickle.load(open(root_path + folder_results + folder_models + file_name, 'rb'))


def prediction(model_name: str, file_pred='', X_pred=None):
    model = getModel(file_name=model_name)
    model.inference(X_pred=X_pred, file=file_pred)

