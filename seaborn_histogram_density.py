"""
    Distribution - We visualize distributions to show the possible values 
        that we can expect to see in a variable, along with how likely they are.
    sns.histplot - Histograms show the distribution of a single numerical variable.

    sns.kdeplot - KDE plots (or 2D KDE plots) show an estimated, 
        smooth distribution of a single numerical variable (or two numerical variables).

    sns.jointplot - This command is useful for simultaneously displaying a 2D KDE plot 
        with the corresponding KDE plots for each individual variable.
"""
from matplotlib import pyplot as plt
import pandas as pd
pd.plotting.register_matplotlib_converters()
import seaborn as sns

if __name__ == "__main__":
    file_path = "Kaggle/iris.csv"
    iris_data = pd.read_csv(file_path)
    print(iris_data.head())
    
    sns.histplot(x = iris_data["Petal Length (cm)"]) 
    plt.show()

    sns.kdeplot(x = iris_data["Petal Length (cm)"],fill = True)
    plt.show()

    sns.jointplot(x = iris_data["Petal Length (cm)"],y = iris_data["Sepal Width (cm)"])
    plt.show()

    sns.jointplot(x = iris_data["Petal Length (cm)"],y = iris_data["Sepal Width (cm)"],kind = "kde")
    plt.show()

    sns.histplot(data = iris_data,x = iris_data["Petal Length (cm)"],hue = iris_data["Species"])
    plt.title("Histogram of Petal Lengths, by Species")
    plt.show()

    sns.kdeplot(data = iris_data,x = iris_data["Petal Length (cm)"],fill = True,hue = iris_data["Species"])
    plt.title("Distribution of Petal Lengths, by Species")
    plt.show()