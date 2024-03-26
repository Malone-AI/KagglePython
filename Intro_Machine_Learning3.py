"""
    Underfitting and Overfitting

    Models can suffer from either:
        Overfitting: capturing spurious patterns that won't recur in the future, 
            leading to less accurate predictions, or
        Underfitting: failing to capture relevant patterns, 
            again leading to less accurate predictions.
    We use validation data, which isn't used in model training, 
    to measure a candidate model's accuracy. This lets us try 
    many candidate models and keep the best one.

"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

file_path = "Kaggle/melb_data.csv"
mel_data = pd.read_csv(file_path)
mel_data = mel_data.dropna(axis = 0)

y = mel_data.Price
mel_features = ["Rooms","Bathroom","Landsize","BuildingArea","YearBuilt","Lattitude","Longtitude"]
X = mel_data[mel_features]
train_X,val_X,train_Y,val_Y = train_test_split(X,y,random_state = 1)

def get_mae(max_leaf_node,train_X,val_X,train_Y,val_Y):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_node)
    model.fit(train_X,train_Y)
    val_predictions = model.predict(val_X)
    mae = mean_absolute_error(val_Y,val_predictions)
    return mae

for max_leaf_node in [5,50,500,5000]:
    mae = get_mae(max_leaf_node,train_X,val_X,train_Y,val_Y)
    print(f"Max leaf node : {max_leaf_node}\t\tMean Absolute Error : {mae}")
