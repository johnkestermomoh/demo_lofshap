import sklearn
from sklearn import neighbors
import pytest
# from sklearn.utils._testing import assert_allclose
# from sklearn.utils._testing import assert_array_equal
from faker import Faker
import pandas as pd
from demo_lofshap.custom_fit_predict_lofshap import custom_fit_predict

# create the object
fake = Faker()
import numpy as np

fakeDataframe1 = pd.DataFrame({'stw': np.random.randint(12, 14, size=8000),
                               'shaft_power': np.random.randint(10000, 12000, size=8000),
                               'compressor_power': np.random.randint(80, 90, size=8000),
                               'sog': np.random.randint(12, 14, size=8000)})

fakeDataframe2 = pd.DataFrame({'stw': np.random.randint(1, 20, size=8000),
                               'shaft_power': np.random.randint(10000, 12000, size=8000),
                               'compressor_power': np.random.randint(80, 90, size=8000),
                               'sog': np.random.randint(1, 20, size=8000)})


@pytest.mark.parametrize("algorithm", ['xgboost', 'random_forest', 'Histgradientboost', 'Adaboost'])
@pytest.mark.parametrize("data", [fakeDataframe1, fakeDataframe2])
def test_df_input_good_data(data, algorithm):
    x = custom_fit_predict(data, model=algorithm)
    assert isinstance(x[0], pd.DataFrame)
    assert isinstance(x[1][1], sklearn.neighbors._lof.LocalOutlierFactor)
    assert len(data) == len(x[0])
    assert data.shape[1] < x[0].shape[1]
    assert type(x[1]).__name__ == 'Pipeline'


@pytest.mark.parametrize("algorithm", ['xgboost', 'random_forest', 'Histgradientboost', 'Adaboost'])
@pytest.mark.parametrize("data", [fakeDataframe1, fakeDataframe2])
def test_ndarray_input_good_data(data, algorithm):
    x = custom_fit_predict(np.array(data), _columns=data.columns.tolist())
    assert isinstance(x[0], pd.DataFrame)
    assert isinstance(x[1][1], sklearn.neighbors._lof.LocalOutlierFactor)
    assert len(data) == len(x[0])
    assert data.shape[1] < x[0].shape[1]
    assert type(x[1]).__name__ == 'Pipeline'


dommy_data = np.asarray([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2],
                         [2, 1], [5, 3], [-100, 10]])
dommy_df = pd.DataFrame(data=dommy_data, columns=['dommy_column1', 'dommy_column2'])
dommy_df


@pytest.mark.parametrize("algorithm", ['xgboost', 'random_forest', 'Histgradientboost', 'Adaboost'])
@pytest.mark.parametrize("data", [dommy_df])
def test_df_input_bad_data(data, algorithm):
    x = custom_fit_predict(data, model=algorithm)
    assert isinstance(x[0], pd.DataFrame)
    assert isinstance(x[1][1], sklearn.neighbors._lof.LocalOutlierFactor)
    assert len(data) == len(x[0])
    assert data.shape[1] < x[0].shape[1]
    assert type(x[1]).__name__ == 'Pipeline'
# %%
