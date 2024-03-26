import pandas as pd
from matplotlib import pyplot as plt
pd.plotting.register_matplotlib_converters()
import seaborn as sns

file_path = "Kaggle/spotify.csv"
spotify_data = pd.read_csv(file_path,index_col = "Date",parse_dates = True)
# print(spotify_data.head())
# print(spotify_data.tail())
plt.figure(figsize = (14,6))
plt.title("Daily Global Streams of Popular Songs in 2017-2018")
# sns.lineplot(spotify_data)
# plt.show()
sns.lineplot(spotify_data["Shape of You"],label = "Shape of You")
sns.lineplot(spotify_data["Despacito"],label = "Despacito")
plt.xlabel("Date")
plt.ylabel("")
plt.show()
