# Import required libraries
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Kernels to compare
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Store results
results = []

for kernel in kernels:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results.append([kernel, acc])

# Create DataFrame
results_df = pd.DataFrame(results, columns=['Kernel', 'Accuracy'])
print("Kernel Comparison:\n")
print(results_df)

# Best kernel
best_kernel = results_df.loc[results_df['Accuracy'].idxmax()]
print("\nBest Kernel:")
print(best_kernel)