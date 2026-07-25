import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Salary_Data.csv")

# Preprocessing
df = df.drop_duplicates()
df = df.dropna()
df["YearsExperience"] = pd.to_numeric(df["YearsExperience"], errors="coerce")
df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
df = df.dropna()

# EDA
print(df.info())
print(df.describe())
print(df.corr(numeric_only=True))

sns.scatterplot(data=df, x="YearsExperience", y="Salary")
plt.title("Salary vs Years of Experience")
plt.show()

# Features and target
X = df[["YearsExperience"]]
y = df["Salary"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Linear regression training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("R² Score:", r2_score(y_test, y_pred))

# Visualization
plt.scatter(X_test, y_test, color="blue", label="Actual salary")
plt.plot(X_test, y_pred, color="red", label="Prediction line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Salary Prediction")
plt.legend()
plt.show()