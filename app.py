import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# 1. Load the data
df = pd.read_csv('customers.csv')

# 2. Select Features (Annual Income and Spending Score)
X = df.iloc[:, [3, 4]].values

# 3. Create 5 Segments
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# 4. Show the Results
summary = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean'
}).reset_index()

print("✅ Success! You've segmented your customers.")
print(summary)
# Create a visual scatter plot of the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Cluster', palette='viridis', s=100)
plt.title('Customer Segments: Income vs Spending')
plt.savefig('customer_clusters.png')  # This saves the "Dashboard" image
print("✅ Dashboard visual saved as 'customer_clusters.png'")