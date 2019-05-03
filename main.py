#full kernel advanced regression on housing comp.(self)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
X_full=pd.read_csv("../input/train.csv")
print(X_full.shape)
X_test_full=pd.read_csv("../input/test.csv")

#remove missing target,seperate target from predictors
X_full.dropna(subset=['SalePrice'],inplace=True,axis=0)
y=X_full.SalePrice
X_full.drop(['SalePrice'],axis=1,inplace=True)
print(X_full.shape)
#split into training and validation data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2,random_state=0)

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]
                
# Select categorical columns with relatively low cardinality 
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 15 and 
                    X_train_full[cname].dtype == "object"]
                    
              

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()                 

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
model=XGBRegressor(n_estimators=1000,random_state=0,learning_rate=0.05)        

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train 
             )

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

preds_test = my_pipeline.predict(X_test)

# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index+1461,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

