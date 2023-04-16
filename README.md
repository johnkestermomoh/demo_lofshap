# demo_lofshap

## Example application
Examples

    --------
    >>> import numpy as np
    >>> from sklearn.neighbors import LocalOutlierFactor
    >>> X = [[-1.1], [0.2], [101.1], [0.3]]
    >>> clf = LocalOutlierFactor(n_neighbors=2)
    >>> clf.fit_predict(X)
    array([ 1,  1, -1,  1])
    >>> clf.negative_outlier_factor_
    array([ -0.9821...,  -1.0370..., -73.3697...,  -0.9821...])
    ---------
kkk