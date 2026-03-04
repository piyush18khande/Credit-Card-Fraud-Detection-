# Credit-Card-Fraud-Detection-
Credit Card Fraud Detection using SOM + ANN
This notebook demonstrates a two-stage fraud detection approach that combines unsupervised learning (Self-Organizing Maps) with supervised deep learning (Artificial Neural Networks).

The idea is:

Use SOM to detect anomalous (suspicious) customers without labels.
Use those anomalies to create pseudo-labels.
Train an ANN to assign a fraud probability score to every customer.
📌 Dataset
File: Credit_Card_Applications.csv
Column 0: Customer ID
Columns 1–15: Customer attributes
Last column (y): Application approval (used only for visualization)
🔹 Part 1: Self-Organizing Map (Unsupervised Learning)
1. Data Preparation
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X: Input features (scaled to [0,1])
y: Approval labels (not used for training, only for visualization)
2. Feature Scaling
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)
SOMs are distance-based models, so scaling is required.

3. Training the SOM
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)
Key parameters:

x, y: SOM grid size (10×10)
input_len: Number of features
sigma: Neighborhood radius
learning_rate: Weight update rate
4. Visualizing the SOM (U-Matrix)
pcolor(som.distance_map().T)
colorbar()
Dark cells = large distance to neighbors
These represent outliers / anomalies
Customer markers:

o (red): Rejected applications
s (green): Approved applications
5. Identifying Potential Frauds
mappings = som.win_map(X)
Each SOM node maps to a list of customers assigned to it.
Selected high-distance (dark) nodes are treated as suspicious.
frauds = []
for node in [(2,7), (1,6), (2,6)]:
    if node in mappings:
        frauds.extend(mappings[node])
These customers are anomalies, not confirmed frauds.
Features are inverse-scaled back to original values.
🔹 Part 2: From Unsupervised to Supervised Learning
6. Creating Labels from SOM Output
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
SOM anomalies → is_fraud = 1
Others → is_fraud = 0
This creates pseudo-labels.

7. Preparing Data for ANN
customers = dataset.iloc[:,1:].values
customers = StandardScaler().fit_transform(customers)
Customer ID is excluded
Standardization is used for ANN
🔹 Part 3: Artificial Neural Network (Supervised Learning)
8. Building the ANN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=2, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
Output layer gives fraud probability
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
9. Training the ANN
ann.fit(customers, is_fraud, batch_size=1, epochs=15)
The ANN learns patterns that distinguish anomalous customers from normal ones.

🔹 Part 4: Fraud Probability Ranking
10. Predicting Fraud Scores
y_pred = ann.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]
Final output:

[Customer_ID, Fraud_Probability]
Sorted in ascending order of fraud probability
Most suspicious customers are at the bottom
✅ Final Outcome
✔ SOM detects anomalous customers (unsupervised) ✔ ANN assigns fraud probability scores (supervised) ✔ Customers can be ranked for manual investigation

⚠️ Important Notes
SOM anomalies ≠ confirmed fraud
This approach is ideal when true fraud labels are unavailable
In real systems, investigators validate top-ranked cases
🧠 Key Takeaway
SOM finds unusual behavior, ANN learns to score fraud risk.

This hybrid approach is commonly used in real-world fraud detection pipelines.
