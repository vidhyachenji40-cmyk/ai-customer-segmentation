import pandas as pd
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