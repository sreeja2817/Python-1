import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

df = pd.read_csv('shopping_trends_updated.csv').dropna()  
print("\ndescriptive statistics:")
print(df.describe()) 
print("\ncolumns in dataset:")
print(list(df.columns))


if 'item Purchased' in df.columns:
    plt.figure(figsize=(11, 7))
    sns.scatterplot(data=df, x='age', y='item Purchased', hue='gender', palette='deep')
    plt.title('age vs. item Purchased by gender')
    plt.xlabel('age')
    plt.ylabel('item Purchased')
    plt.legend(title='gender')
    plt.show()
else:
    print(f"Column {'item Purchased'} is missing. Scatter plot can't be created.")


plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Category', hue='Frequency of Purchases', palette='muted')
plt.title('Frequency of Purchases by Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)  
plt.legend(title='Frequency of Purchases')
plt.show()


Numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[Numeric_columns].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()


AGE_DATA = df['Age'].dropna()  

M_age = AGE_DATA.mean()  #MEAN 1
V_age = AGE_DATA.var()   #VARIANCE 2
SK_age = skew(AGE_DATA)  #SKEWNESS 3
K_age = kurtosis(AGE_DATA)  #KURTOSIS 4


print("\nStatistical moments for age:")
print(f"Mean: {M_age:.2f}")
print(f"Variance: {V_age:.2f}")
print(f"Skewness: {SK_age:.2f}")
print(f"Kurtosis: {K_age:.2f}")


plt.savefig('analysis_plots.png')