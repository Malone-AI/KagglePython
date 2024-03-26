"""
    sns.scatterplot - Scatter plots show the relationship between two continuous variables;
        if color-coded, we can also show the relationship with a third categorical variable.

    sns.regplot - Including a regression line in the scatter plot makes it easier to see any 
        linear relationship between two variables.

    sns.lmplot - This command is useful for drawing multiple regression lines, 
        if the scatter plot contains multiple, color-coded groups.

    sns.swarmplot - Categorical scatter plots show the relationship 
        between a continuous variable and a categorical variable.
"""

import pandas as pd
from matplotlib import pyplot as plt
pd.plotting.register_matplotlib_converters()
import seaborn as sns

file_path = "Kaggle/insurance.csv"
insurance_data = pd.read_csv(file_path)
print(insurance_data.head())

sns.scatterplot(x = insurance_data["bmi"],y = insurance_data["charges"])
plt.show()

sns.regplot(x = insurance_data["bmi"],y = insurance_data["charges"])
plt.show()

sns.scatterplot(x = insurance_data["bmi"],y = insurance_data["charges"],hue = insurance_data["smoker"])
plt.show()

sns.lmplot(x = "bmi",y = "charges",hue = "smoker",data = insurance_data)
plt.show()

sns.swarmplot(x = insurance_data["smoker"],y = insurance_data["charges"])
plt.show()