# PyGML (Geographically Weighted Machine Learning in Python)
Python 3 Based Implementation of Geographically Weighted Machine Learning Models such as GXGB (Geographically Weighted XGBoost), GWMLP (Geographically Weighted Multi-layer Perceptrons) and GWRNN (Geographically Weighted RNN).
This repository contains the source code and parameter descriptions of each PyGML model, and Jupyter Notebooks and related datasets for Housing Price Prediction.

# Repository organization
 - The file "PyGML.py" is the source code of this Python-based ML models.
 - The file "Description_Parameters.pdf" explains the details of parameters in each model.
 - The folder "Notebooks" contains three Jupyter Notebooks used for implementing two Housing Price predictions cases and one Climate data modeling case. 
 - The folder "Data" contains the Housing dataset1, Housing dataset2 and Climate dataset used for application of GXGB, GWMLP and GWRNN Models respectively.
 - The folder "Test" contains the Python file for the unit test for the PyGML package. The Pytest package is needed to run this file.

# Installation
We have published PyGML as a Python package in PyPI. You can directly install it with the command "pip install PyGML".

# Potential issues and solutions
PyGML requires the pacakge esda as a dependency for computing Moran's Index. We recommend users to install esda 2.5, and then PyGML can be used smoothly with any additional action. If you use the latest version of esda 2.6, you will need to install matplotlib manually in order to import PyGRF successfully.

# Example
Below shows an example on how to fit a GXGB model and use it to make predictions.
'''

from PyGML import GXGB
from sklearn.model_selection import train_test_split

#Instantiate GXGB model with chosen parameters
model = GXGB(
    band_width=bandwidth,
    kernel="adaptive",  # or "fixed"
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    subsample=1.0,
    train_weighted=True,
    predict_weighted=True,
    random_state=42
)

#Fit model on training data
model.fit(X_train, y_train, coords_train)

#Predict on test data
local_weight = 0.5  # blend local and global predictions (adjust as needed)
y_pred, y_pred_global, y_pred_local = model.predict(X_test, coords_test, local_weight=local_weight)

#Evaluate performance
print("R2 score (combined):", r2_score(y_test, y_pred))
print("R2 score (global only):", r2_score(y_test, y_pred_global))
print("R2 score (local only):", r2_score(y_test, y_pred_local))
print("RMSE (combined):", np.sqrt(mean_squared_error(y_test, y_pred)))

#(Optional) Get local feature importances from all local models
fi_df = model.get_local_feature_importance()
print(fi_df.head())

'''

# Parameters
If you want to learn more about the major parameters in this package, please refer to the Description of Parameters.


# Authors
Moin tariq - AI for Digital Earth Lab, Shandong University, Jinan, China - Email: sci.mointariq@gmail.com, moin.tariq@mail.sdu.edu.cn , moin.bsma1810@iiu.edu.pk

# Reference
If you use the data or code from this repository, or the PyGRF package, we will really appreciate if you can cite our paper:

- Moin Tariq, Zhihua Zhang, Geographically Weighted xtreme Gradient Boost Modeling of Housing Price Determinents in Islamabad, Pakistan

