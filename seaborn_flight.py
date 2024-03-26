"""
    sns.barplot - Bar charts are useful for comparing quantities corresponding to different groups. 
    sns.heatmap - Heatmaps can be used to find color-coded patterns in tables of numbers.
"""
import pandas as pd
from matplotlib import pyplot as plt
pd.plotting.register_matplotlib_converters()
import seaborn as sns

file_path = "Kaggle/flight_delays.csv"
flight_data = pd.read_csv(file_path,index_col = "Month")
# print(flight_data)

plt.figure(figsize = (14,6))
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")
sns.barplot(x = flight_data.index,y = flight_data["NK"])
plt.ylabel("Arrival delay (in minutes)")
plt.show()

plt.figure(figsize = (14,6))
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")
sns.heatmap(data = flight_data,annot = True)
plt.xlabel("Airline")
plt.show()