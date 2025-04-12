import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#1
df = pd.read_csv("E:/PERSONAL/amazon_product_shipment_dataset.csv")


avg_cost_by_shipment = df.groupby('Mode_of_Shipment')['Cost_of_the_Product'].mean()


plt.figure(figsize=(8, 6))
avg_cost_by_shipment.plot(kind='bar', color='skyblue')
plt.title('Average Cost of Product by Mode of Shipment')
plt.xlabel('Mode of Shipment')
plt.ylabel('Average Cost')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#2

plt.figure(figsize=(8, 6))
plt.hist(df['Discount_offered'], bins=20, color='orange', edgecolor='black')
plt.title('Distribution of Discounts Offered')
plt.xlabel('Discount Offered')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#3
correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Numerical Feature Correlations')
plt.tight_layout()
plt.show()

#4
importance_counts = df['Product_importance'].value_counts()


plt.figure(figsize=(7, 7))
importance_counts.plot.pie(
    autopct='%1.1f%%',
    startangle=140,
    colors=['gold', 'lightgreen', 'lightcoral'],
    wedgeprops={'edgecolor': 'black'}
)
plt.title('Proportion of Product Importance')
plt.ylabel('')  
plt.tight_layout()
plt.show()

#5
avg_rating_by_purchases = df.groupby('Prior_purchases')['Customer_rating'].mean()


plt.figure(figsize=(8, 6))
avg_rating_by_purchases.plot(kind='line', marker='o', color='purple')
plt.title('Average Customer Rating by Prior Purchases')
plt.xlabel('Number of Prior Purchases')
plt.ylabel('Average Customer Rating')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()