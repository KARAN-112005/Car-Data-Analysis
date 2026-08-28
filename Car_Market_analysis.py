
# Importing Librarires

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
# %matplotlib inline  # Uncomment only in Jupyter/Notebook environments

# Load the Dataset and Fetching records of first 5 rows

df = pd.read_csv('/content/1776311302-P3-Car Market Trends Analysis with Car Dekho Data.csv')
df.head()

# Shape/Columns,rows in Data

df.shape

df.info()

df.isnull().sum()   # check for missing values

df.describe()        # summary statistics for numeric columns

# Create a Car_Age column (assuming the data was collected around 2020) and a Depreciation column.

df['Car_Age'] = 2020 - df['Year']
df['Depreciation'] = df['Present_Price'] - df['Selling_Price']
df['Depreciation_Pct'] = (df['Depreciation'] / df['Present_Price']) * 100
df.head()

print(df['Fuel_Type'].value_counts())
print()
print(df['Seller_Type'].value_counts())
print()
print(df['Transmission'].value_counts())
print()
print(df['Owner'].value_counts())

# Distribution of Selling Price

plt.figure(figsize=(7,5))
sns.histplot(df['Selling_Price'], bins=25, kde=True, color='#4C72B0')
plt.title('Distribution of Selling Price (Lakhs)')
plt.xlabel('Selling Price (Lakh INR)')
plt.ylabel('Number of Cars')
plt.tight_layout()
plt.show()  # Insight: Most cars in the dataset sell for under 6 lakh rupees; a small number of premium cars pull the distribution's tail to the right.

#Present Price vs Selling Price

plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x='Present_Price', y='Selling_Price', hue='Fuel_Type', alpha=0.7)
plt.title('Present Price vs Selling Price')
plt.xlabel('Present (Showroom) Price (Lakh INR)')
plt.ylabel('Selling Price (Lakh INR)')
plt.tight_layout()
plt.show()  #Insight: Selling price rises almost linearly with present (showroom) price — this is the strongest single predictor of resale value.

#Average Selling Price by Fuel Type

plt.figure(figsize=(6,5))
order = df.groupby('Fuel_Type')['Selling_Price'].mean().sort_values(ascending=False).index
sns.barplot(data=df, x='Fuel_Type', y='Selling_Price', order=order, hue='Fuel_Type', legend=False, palette='viridis', errorbar=None)
plt.title('Average Selling Price by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Avg Selling Price (Lakh INR)')
plt.tight_layout()
plt.show()  #Insight: Diesel cars have the highest average resale price, followed by CNG and Petrol.

# Average Selling Price by Transmission Type

plt.figure(figsize=(6,5))
sns.barplot(data=df, x='Transmission', y='Selling_Price', hue='Transmission', legend=False, palette='magma', errorbar=None)
plt.title('Average Selling Price by Transmission Type')
plt.xlabel('Transmission')
plt.ylabel('Avg Selling Price (Lakh INR)')
plt.tight_layout()
plt.show()  #Insight: Automatic-transmission cars resell for more than double the average price of manual cars.



#Selling Price vs Car Age

plt.figure(figsize=(7,5))
sns.regplot(data=df, x='Car_Age', y='Selling_Price', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Selling Price vs Car Age')
plt.xlabel('Car Age (Years)')
plt.ylabel('Selling Price (Lakh INR)')
plt.tight_layout()
plt.show()  #Insight: Selling price trends downward as car age increases, reflecting typical vehicle depreciation.

#Correlation Heatmap

plt.figure(figsize=(6,5))
corr = df[['Selling_Price','Present_Price','Kms_Driven','Car_Age','Owner']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()  #Insight: Present_Price has the strongest positive correlation with Selling_Price (0.88), while Car_Age has a moderate negative correlation (-0.24). Kms_Driven has almost no linear relationship with selling price (0.03).

# Average Selling Price by Seller Type

plt.figure(figsize=(6,5))
sns.barplot(data=df, x='Seller_Type', y='Selling_Price', hue='Seller_Type', legend=False, palette='crest', errorbar=None)
plt.title('Average Selling Price by Seller Type')
plt.xlabel('Seller Type')
plt.ylabel('Avg Selling Price (Lakh INR)')
plt.tight_layout()
plt.show()  #Insight: Cars sold through dealers have a much higher average selling price than those sold by individuals — dealers likely handle newer/premium inventory.

# Count of Cars by Fuel Type

plt.figure(figsize=(6,5))
sns.countplot(data=df, x='Fuel_Type', hue='Fuel_Type', legend=False, palette='pastel',
              order=df['Fuel_Type'].value_counts().index)
plt.title('Count of Cars by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Top 10 Most Listed Car Models

df['Car_Name'].value_counts().head(10)

## 15. Conclusion

# - **Present price** is the strongest driver of a used car's selling price (correlation ≈ 0.88).
# - **Fuel type** and **transmission** matter: diesel and automatic cars resell for noticeably more than petrol/manual cars.
# - **Car age** negatively affects selling price — cars depreciate steadily over time.
# - **Kilometers driven** has almost no linear effect on price in this dataset.
# - **Seller type** matters — dealer-sold cars fetch higher average prices than individual sales.

# These insights can help buyers judge fair value, help sellers price competitively, and help dealers/businesses understand demand patterns in the used-car market.