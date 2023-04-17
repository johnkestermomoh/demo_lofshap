## demo_lofshap
    install the package with the following line of code
    pip install git+https://https://github.com/johnkestermomoh/demo_lofshap.git
    
**Example of application shown below**

    
    Example: Using dataframe as input
    >>> from py_lof_shap.custom_fit_predict_lofshap import custom_fit_predict
    fakeDataframe = pd.DataFrame({'stw': np.random.randint(1, 20, size=5),
                                  'shaft_power': np.random.randint(10000, 12000, size=5),
                                  'compressor_power': np.random.randint(80, 90, size=5),
                                  'sog': np.random.randint(1, 20, size=5)})
    >>> df, model = custom_fit_predict(fakeDataframe)
    (5, 4)
    0    1
    1    1
    2    1
    3    1
    4    1
    Name: STATUS, dtype: int64
    No outliers found - according to model perfect signals but check!
    >>> df['STATUS_STATEMENT']
    0    signal OK!
    1    signal OK!
    2    signal OK!
    3    signal OK!
    4    signal OK!
    Name: STATUS_STATEMENT, dtype: object

    >>> model[1].negative_outlier_factor_
    array([-1.02897676, -0.97261055, -0.97261055, -1.24121937, -1.02897676])
    
The results will contain the signal `status` and the `status statement` which specifies the marginal contribution of 
each signal to the model output. It also returns the `scikit-learn Pipeline` containing 
`StandardScaler` and the `LOF model`

    
    Example: Using numpy ndarrays as input
    >>> from py_lof_shap.custom_fit_predict_lofshap import custom_fit_predict
    >>> X = pd.DataFrame(data=np.asarray([[-1.1, 1], [0.2, 0.3], [101.1, 200], [0.3, 0.5]]),
                     columns=['sensor_a', 'sensor_b'])
    X
    >>> df, model = custom_fit_predict(X, n_neighbors=2, contamination='auto')
    (4, 2)
    0    1
    1    1
    2   -1
    3    1
    Name: STATUS, dtype: int64
    [1 1 0 1]
    prediction model accuracy: 1.0
    Loading the game model
    4
    [-0.285  -0.2075  0.1125  0.165 ]
    process complete
    >>> df['STATUS_STATEMENT']
    0                           signal OK
    1                           signal OK
    2    {'sensor_b': 42, 'sensor_a': 57}
    3                           signal OK
    Name: STATUS_STATEMENT, dtype: object
    >>> model[1].negative_outlier_factor_
    array([  -0.98680632,   -1.02710252, -101.87751967,   -0.98680632])
    
    
### Model selection options for building the signal status
  - XGBoost Classifier - Fastest but least accurate
  - Random Forest Classifier - most accurate but takes more time to compute
  - HistGradientBoostClassifier - Also fast like XGBoost
  - AdaBoostClassifier - slow
