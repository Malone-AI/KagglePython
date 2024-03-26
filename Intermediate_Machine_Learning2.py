"""
    Categorical Variables
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

file_path = "Kaggle/melb_data.csv"
mel_data = pd.read_csv(file_path)

y = mel_data.Price
X = mel_data.drop(["Price"],axis = 1)

train_X_full,val_X_full,train_y,val_y = train_test_split(X,y,train_size = 0.8,test_size = 0.2,random_state = 0)

col_with_missing = [col for col in train_X_full.columns if train_X_full[col].isnull().any()]
train_X_full.drop(col_with_missing,axis = 1,inplace = True)
val_X_full.drop(col_with_missing,axis = 1,inplace = True)

low_cardinality_cols = [col for col in train_X_full.columns 
                        if train_X_full[col].nunique() < 10 and train_X_full[col].dtype == "object"]
numerical_cols = [col for col in train_X_full.columns if train_X_full[col].dtype in ["int64","float64"]]
my_cols = low_cardinality_cols + numerical_cols

train_X = train_X_full[my_cols].copy()
val_X = val_X_full[my_cols].copy()

def get_mae(train_X,val_X,train_y,val_y):
    model = RandomForestRegressor(n_estimators = 100,random_state = 0)
    model.fit(train_X,train_y)
    val_y_pre = model.predict(val_X)
    mae = mean_absolute_error(val_y,val_y_pre)
    return mae

# Score from Approach 1 (Drop Categorical Variables)
drop_train_X = train_X.select_dtypes(exclude = ["object"])
drop_val_X = val_X.select_dtypes(exclude = ["object"])
print(f"MAE from Approach 1 (Drop categorical variables):\n\t\t\t{get_mae(drop_train_X,drop_val_X,train_y,val_y)}")

# Score from Approach 2 (Ordinal Encoding)
s = (train_X.dtypes == "object")
object_cols = list(s[s].index) 
label_train_X = train_X.copy()
label_val_X = val_X.copy()
ordinal_encoder = OrdinalEncoder()
label_train_X[object_cols] = ordinal_encoder.fit_transform(train_X[object_cols])
label_val_X[object_cols] = ordinal_encoder.transform(val_X[object_cols])
print(f"MAE from Approach 2 (Ordinal Encoding):\n\t\t\t{get_mae(label_train_X,label_val_X,train_y,val_y)}")

# Score from Approach 3 (One-Hot Encoding)
OH_encoder = OneHotEncoder(handle_unknown = "ignore",sparse = False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_X[object_cols]))
OH_cols_val = pd.DataFrame(OH_encoder.transform(val_X[object_cols]))

OH_cols_train.index = train_X.index
OH_cols_val.index = val_X.index

num_train_X = train_X.drop(object_cols,axis = 1)
num_val_X = val_X.drop(object_cols,axis = 1)
OH_train_X = pd.concat([num_train_X,OH_cols_train],axis = 1)
OH_val_X = pd.concat([num_val_X,OH_cols_val],axis = 1)
OH_train_X.columns = OH_train_X.columns.astype(str)
OH_val_X.columns = OH_val_X.columns.astype(str)
print(f"MAE from Approach 3 (One-Hot Encoding):\n\t\t\t{get_mae(OH_train_X,OH_val_X,train_y,val_y)}")