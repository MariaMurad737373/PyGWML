# Pygwml (Geographically Weighted Machine Learning in Python)
Python 3 Based Implementation of Geographically Weighted Machine Learning Models such as:
 - GXGB (Geographically Weighted XGBoost).
 - GWMLP (Geographically Weighted Multi-layer Perceptrons).
 - GWRNN (Geographically Weighted RNN).
   
This repository contains the source code and parameter descriptions of each Pygwml model, and Jupyter Notebooks and related datasets for Housing Price Prediction etc.

# Repository organization
 - The file "Pygwml.py" is the source code of this Python-based ML models.
 - The file "Description_Parameters" explains the details of parameters in each model.
 - The folder "Notebooks" contains three Jupyter Notebooks used for implementing two Housing Price predictions cases and one Climate data modeling case. 
 - The folder "Data" contains the Housing dataset1, Housing dataset2 and Climate dataset used for application of GXGB, GWMLP and GWRNN Models respectively.
 - The folder "Test" contains the Python file for the unit test for the Pygwml package. The Pytest package is needed to run this file.

# Installation
We have published Pygwml as a Python package in PyPI. You can directly install it with the command "pip install pygwml".

# Potential issues and solutions
 - Pygwml requires the pacakge esda as a dependency for computing Moran's Index. We recommend users to install esda 2.5, and then Pygwml can be used smoothly with any additional action. If you use the latest version of esda 2.6, you will need to install matplotlib manually in order to import Pygwml successfully.
 - Libpysal is used specifically for the Incremental Spatial Autocorrelation (ISA) analysis to help find the optimal spatial bandwidth (ISA_op_bw function).  libpysal >= 4.4.0 is needed to be installed for this.

# Example 1: Implementation GXGB Model
Below shows an example on how to fit a GXGB model and use it to make predictions.
```python

from pygwml import GXGB, ISA_op_bw
from sklearn.model_selection import train_test_split

#Get optimized Bandwidht
bandwidth, local_weight, p_value = ISA_op_bw(y_train,coords_train)

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

#(Optional) #Get Feature Importances
local_feature_importance=model.get_local_feature_importance()
local_feature_importance=model.global_model.feature_importances_
globally_enhanced_local_feature_importances=model.get_globally_enhanced_local_feature_importances()
```

# Parameters
If you want to learn more about the major parameters in this package, please refer to the Description of Parameters.


# Authors
 - Moin tariq - AI for Digital Earth Lab, Shandong University, Jinan, China - Email: sci.mointariq@gmail.com, moin.tariq@mail.sdu.edu.cn , moin.bsma1810@iiu.edu.pk
 - Muhammad Irfan Haider Khan - Key Laboratory of Artificial Intelligence, Optics and Electronics (iOPEN), NWPU, Xi’an, Shaanxi, China - Email: vice.haider@gmail.com, irfankhan@mail.nwpu.edu.cn
   
# Reference
If you use the data or code from this repository, or the PyGML package, we will really appreciate if you can cite our paper:

- Moin Tariq, Zhihua Zhang, Geographically Weighted xtreme Gradient Boost Modeling of Housing Price Determinents in Islamabad, Pakistan.

