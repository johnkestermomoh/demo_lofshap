# demo_lofshap
    install the package with the following line of code
    pip install git+https://github.com/johnkestermomoh/demo_lofshap.git
    
`Example application shown below`

    --------
    Example: Using dataframe
    >>> import numpy as np
    fakeDataframe = pd.DataFrame({'stw': np.random.randint(1, 20, size=8000),
                                  'shaft_power': np.random.randint(10000, 12000, size=8000),
                                  'compressor_power': np.random.randint(80, 90, size=8000),
                                  'sog': np.random.randint(1, 20, size=8000)})
    >>> from sklearn.neighbors import LocalOutlierFactor
    >>> df, model = custom_fit_predict(fakeDataframe)
    array([ 1,  1, -1,  1])
    >>> clf.negative_outlier_factor_
    array([ -0.9821...,  -1.0370..., -73.3697...,  -0.9821...])
    ---------

The results will contain the signal `status` and the `status statement` which specifies the marginal contribution of 
each signal to the model output. It also returns the `scikit-learn Pipeline` containing 
`StandardScaler` and the `LOF model`


    
    --------
    Example: Using numpy ndarrays 
    >>> import numpy as np
    >>> from sklearn.neighbors import LocalOutlierFactor
    >>> X = [[-1.1], [0.2], [101.1], [0.3]]
    >>> clf = LocalOutlierFactor(n_neighbors=2)
    >>> clf.fit_predict(X)
    array([ 1,  1, -1,  1])
    >>> clf.negative_outlier_factor_
    array([ -0.9821...,  -1.0370..., -73.3697...,  -0.9821...])
    ---------
    
### Model selection options for building the signal status
  - XGBoost Classifier - Fastest but least accurate
  - Random Forest Classifier - most accurate but takes more time to compute
  - HistGradientBoostClassifier - Also fast like XGBoost
  - AdaBoostClassifier - slow
