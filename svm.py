from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd

# Dataset
df = pd.read_csv("car_purchase_data.csv")
X = df[["Age", "Salary"]].values
y = df["Purchased"].values


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# SVM model
model = SVC(kernel="linear")

# Train
model.fit(X_train, y_train)