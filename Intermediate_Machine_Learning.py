"""
    Missing Values
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

file_path = "Kaggle/melb_data.csv"
mel_data = pd.read_csv(file_path)

y = mel_data.Price
mel_data_filter = mel_data.drop(["Price"],axis = 1)
X = mel_data_filter.select_dtypes(exclude = ["object"])

train_X,val_X,train_y,val_y = train_test_split(X,y,train_size = 0.8,test_size = 0.2,random_state = 0)

def get_mae(train_X,val_X,train_y,val_y):
    model = RandomForestRegressor(n_estimators = 10,random_state = 0)
    model.fit(train_X,train_y)
    val_y_pre = model.predict(val_X)
    mae = mean_absolute_error(val_y,val_y_pre)
    return mae

# Score from Approach 1 (Drop Columns with Missing Values)
missing_value_col = [col for col in X.columns if mel_data_filter[col].isnull().any()]
reduced_train_X = train_X.drop(missing_value_col,axis = 1)
reduced_val_X = val_X.drop(missing_value_col,axis = 1)
print(f"MAE from Approach 1 (Drop columns with missing values):\n\t\t\t{get_mae(reduced_train_X,reduced_val_X,train_y,val_y)}")

# Score from Approach 2 (Imputation)
my_imputer = SimpleImputer()
imputed_train_X = pd.DataFrame(my_imputer.fit_transform(train_X))
imputed_val_X = pd.DataFrame(my_imputer.transform(val_X))
imputed_train_X.columns = train_X.columns
imputed_val_X.columns = val_X.columns
print(f"MAE from Approach 2 (Imputation):\n\t\t\t{get_mae(imputed_train_X,imputed_val_X,train_y,val_y)}")

# Score from Approach 3 (An Extension to Imputation)
train_X_cpy = train_X.copy()
val_X_cpy = val_X.copy()
for col in missing_value_col:
    train_X_cpy[col + "_was_missing"] = train_X_cpy[col].isnull()
    val_X_cpy[col + "_was_missing"] = val_X_cpy[col].isnull()
my_imputer = SimpleImputer()
imputed_train_X_cpy = pd.DataFrame(my_imputer.fit_transform(train_X_cpy))
imputed_val_X_cpy = pd.DataFrame(my_imputer.transform(val_X_cpy))
imputed_train_X_cpy.columns = train_X_cpy.columns
imputed_val_X_cpy.columns = val_X_cpy.columns
print(f"MAE from Approach 3 (An Extension to Imputation):\n\t\t\t{get_mae(imputed_train_X_cpy,imputed_val_X_cpy,train_y,val_y)}")

print(train_X.shape)
missing_val_count_by_columns = train_X.isnull().sum()
print(missing_val_count_by_columns[missing_val_count_by_columns > 0])
print(type(missing_val_count_by_columns))