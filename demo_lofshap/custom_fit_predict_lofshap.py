import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pydantic import validate_arguments
import shap
from typing import Union

__all__ = ['custom_fit_predict']


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def custom_fit_predict(x: Union[pd.DataFrame, np.ndarray], model='random_forest', n_estimators=200, *, n_neighbors=20,
                       algorithm="auto", leaf_size=30, metric="minkowski", p=2, metric_params=None,
                       contamination="auto", novelty=False, n_jobs=None, **kwargs):
    """
        - compute the outlier status ``-1 outlier 1 good``
        - compute the margin contribution of each sensor and prints out the status ``signal ok or dict of sensors with issues``
        - for more details refer to: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
        :param x: {array-like, sparse matrix} of shape (n_samples, n_features), default=None
        :param model: options {'xgboost': xgb, 'random_forest': RandomForestClassifier, 'Histboost': HistGradientBoostingClassifier}
        :param n_estimators:
        :param n_neighbors:
        :param algorithm:
        :param leaf_size:
        :param metric:
        :param p:
        :param metric_params:
        :param contamination:
        :param novelty:
        :param n_jobs:
        :param kwargs:
        :return: Pandas dataframe, pipeline[standardscaler, model]
        """
    # global models
    _x = x.copy()
    print(_x.shape)

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, metric=metric, p=p,
                             contamination=contamination, novelty=novelty, n_jobs=n_jobs)
    pipe_lof = Pipeline([('scaler', StandardScaler()), ('LOF', lof)])

    if isinstance(_x, np.ndarray):
        columns = kwargs['_columns']
        x = pd.DataFrame(data=_x, columns=columns)
        x['STATUS'] = pipe_lof.fit_predict(np.array(x))

    elif isinstance(_x, pd.DataFrame):
        columns = _x.columns
        x = _x.copy()
        del _x
        x['STATUS'] = pipe_lof.fit_predict(np.array(x))

    print(x['STATUS'])

    if len(np.unique(x['STATUS'])) == 1 and np.unique(x['STATUS'])[0] == 1:
        print('No outliers found - according to model perfect signals but check!')
        x['STATUS_STATEMENT'] = 'signal OK!'
        return x, pipe_lof

    else:

        if model == 'xgboost':
            models = xgb.XGBClassifier(objective='binary:logistic')
        elif model == 'random_forest':
            models = RandomForestClassifier()
        elif model == 'Histgradientboost':
            models = HistGradientBoostingClassifier()
        elif model == 'Adaboost':
            models = AdaBoostClassifier()

        model_pipe = Pipeline([('scaler', StandardScaler()), (model, models)])
        y = np.where(x['STATUS'] == -1, 0, 1)
        print(np.array(y))
        model_pipe.fit(np.array(x)[:, :-1], np.array(y))

        y_pred = model_pipe.predict(np.array(x)[:, :-1])
        _performance = np.round(accuracy_score(y, y_pred), 2)
        print(f"prediction model accuracy: {_performance}")
        print(f"Loading the game model")

        _shap_input = model_pipe[0].fit_transform((np.array(x)[:, :-1]))
        if model == 'Adaboost':
            explainer = shap.Explainer(model_pipe[1].predict, _shap_input)
        else:
            explainer = shap.Explainer(model_pipe[1])

        if model == 'random_forest':
            _shap_values = explainer.shap_values(_shap_input)[1]
        elif model == 'Adaboost':
            _shap_values = explainer(_shap_input).values
        else:
            _shap_values = explainer.shap_values(_shap_input)

        driver = []
        print(len(_shap_values))
        print(np.unique(_shap_values))
        if _shap_values.any() != 0:
            for index, row in enumerate(np.abs(_shap_values)):
                _percentages = [int(np.multiply((i / (np.sum(row) + 0.0000000001)), 100)) for i in row]
                if len(x.columns) <= 4:
                    pct = 1
                if len(x.columns) >= 4 & len(x.columns) <= 20:
                    pct = 0.2
                if len(x.columns) >= 20 & len(x.columns) <= 40:
                    pct = 0.1
                else:
                    pct = 0.05
                max_value_indexes = sorted(np.argsort(_percentages)[-int(pct * len(x.columns)):], reverse=True)
                _useful_col = [{x.columns[i]: _percentages[i]} for i in max_value_indexes]
                _statement = {key: var for val in _useful_col for key, var in
                              val.items()}  # 'signal contribution to outcome! {}'.format(_useful_col)
                driver.append(_statement)
            x['STATUS_STATEMENT'] = driver
            x['STATUS_STATEMENT'] = np.where(x['STATUS'] == -1, x['STATUS_STATEMENT'], 'signal OK')
            print('process complete')
            return x, pipe_lof
        else:
            print('Nothing contributed to the model prediction of signal status! check data')
            return x, pipe_lof
