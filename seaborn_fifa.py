"""
    seaborn数据可视化第一课
    Trends - A trend is defined as a pattern of change.
    sns.lineplot - Line charts are best to show trends over a period of time,
    and multiple lines can be used to show trends in more than one group.
"""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
pd.plotting.register_matplotlib_converters()
print("Set up successfully!")

file_path = "Kaggle/fifa.csv"
fifa_data = pd.read_csv(file_path,index_col = "Date",parse_dates = True)
print(fifa_data.head())

plt.figure(figsize = (14,6))
sns.lineplot(data = fifa_data)
plt.show()
