

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

FILE_PATH = 'engine_data.csv'

if not os.path.exists(FILE_PATH):
    print("Dataset not found. Creating dummy engine dataset...")
    dummy_data = {
        'Engine rpm': np.random.randint(500, 1500, 100),
        'Lub oil pressure': np.random.uniform(2, 5, 100),
        'Fuel pressure': np.random.uniform(5, 15, 100),
        'Coolant pressure': np.random.uniform(1, 4, 100),
        'lub oil temp': np.random.uniform(70, 90, 100),
        'Coolant temp': np.random.uniform(75, 95, 100),
        'Engine Condition': np.random.randint(0, 2, 100)
    }
    pd.DataFrame(dummy_data).to_csv(FILE_PATH, index=False)

df = pd.read_csv(FILE_PATH)
print("Dataset loaded successfully")


print(df.head())
print(df.info())
print(df.describe())
print("Missing values:\n", df.isnull().sum())

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('Engine Condition')

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Box plot of {col}')
plt.tight_layout()
plt.savefig('boxplot_outliers.png')
plt.show()


scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
print("Numerical features scaled successfully")

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()


plt.figure(figsize=(18, 12))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='Engine Condition', y=col, data=df)
    plt.title(f'{col} vs Engine Condition')
plt.tight_layout()
plt.savefig('feature_vs_engine_condition.png')
plt.show()

plt.figure(figsize=(18, 12))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.show()

X = df.drop('Engine Condition', axis=1)
y = df['Engine Condition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Classification Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

df['Remaining Useful Life (RUL)'] = np.random.randint(10, 100, size=len(df))

X_reg = df.drop(['Engine Condition', 'Remaining Useful Life (RUL)'], axis=1)
y_reg = df['Remaining Useful Life (RUL)']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(random_state=42)
reg.fit(Xr_train, yr_train)

yr_pred = reg.predict(Xr_test)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))

print("Regression RMSE:", rmse)

clf_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=clf_imp.values, y=clf_imp.index)
plt.title('Feature Importance – Classification')
plt.savefig('classification_feature_importance.png')
plt.show()

reg_imp = pd.Series(reg.feature_importances_, index=X_reg.columns).sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=reg_imp.values, y=reg_imp.index)
plt.title('Feature Importance – Regression (RUL)')
plt.savefig('regression_feature_importance.png')
plt.show()

print("Pipeline executed successfully")
