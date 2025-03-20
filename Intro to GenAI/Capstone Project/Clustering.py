import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib

# Load the CSV data
df = pd.read_csv('data/student_data.csv')

# Preprocessing for User Story 1: Supervised Learning
# Define 'passed' as 1 if G3 >= 10 (passing grade), 0 otherwise
df['passed'] = (df['G3'] >= 10).astype(int)

# Features: studytime (hours studied), G1, G2 (past scores)
X_supervised = df[['studytime', 'G1', 'G2']]
y_supervised = df['passed']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_supervised, y_supervised, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate and display results
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"User Story 1 - Supervised Learning Results:")
print(f"Accuracy of pass/fail prediction: {accuracy:.2f}")

# Display some example predictions
test_samples = X_test.head(5)
predictions = model.predict(test_samples)
print("\nExample Predictions:")
print(test_samples.assign(predicted_pass=predictions))

# Save the model
joblib.dump(model, 'models/classification_model.pkl')

# Preprocessing for User Story 2: Clustering
# Features: G1, G2, studytime, absences
X_clustering = df[['G1', 'G2', 'studytime', 'absences']]

# Normalize the data for clustering
X_clustering = (X_clustering - X_clustering.mean()) / X_clustering.std()

# Train K-means model (3 clusters: fast, average, struggling learners)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_clustering)

# Display clustering results
print("\nUser Story 2 - Clustering Results:")
print("Cluster counts:")
print(df['cluster'].value_counts())

# Visualize clusters (using G1 vs G2 for simplicity)
plt.scatter(df['G1'], df['G2'], c=df['cluster'], cmap='viridis')
plt.xlabel('First Period Grade (G1)')
plt.ylabel('Second Period Grade (G2)')
plt.title('Student Clusters Based on G1, G2, Studytime, and Absences')
plt.colorbar(label='Cluster')
plt.show()

# Display sample students from each cluster
print("\nSample Students from Each Cluster:")
for cluster in range(3):
    print(f"\nCluster {cluster}:")
    print(df[df['cluster'] == cluster][['G1', 'G2', 'studytime', 'absences']].head(5))

# Save the clustering model
joblib.dump(kmeans, 'models/kmeans_model.pkl')
