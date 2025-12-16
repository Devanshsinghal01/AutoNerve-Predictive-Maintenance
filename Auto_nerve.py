


import pandas as pd

df = pd.read_csv('/content/engine_data.csv')
print("DataFrame loaded successfully. First 5 rows:")
print(df.head())



print("\nDataFrame Info:")
df.info()

print("\nDescriptive Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

from google.colab import drive
drive.mount('/content/drive')

import os


notebook_files = [f for f in os.listdir('/content/') if f.endswith('.ipynb')]

if notebook_files:
    notebook_name = notebook_files[0]
    !git add "{notebook_name}"
    print(f"Added notebook ({notebook_name}) to staging.")


    !git commit -m "Add Colab notebook file"
    print("New commit created for the notebook.")

  
    !git push -u origin main
    print("All committed changes pushed to GitHub.")
else:
    print("Error: No .ipynb file found in /content/. Cannot add notebook to Git.")
    print("Please ensure your notebook is saved as an .ipynb file in the /content/ directory.")

import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(15, 10))


numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in numerical_cols:
    numerical_cols.remove('Engine Condition')

for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i + 1) 
    sns.boxplot(y=df[col])
    plt.title(f'Box plot of {col}')
    plt.ylabel('')

plt.tight_layout()
plt.show()

print("Box plots generated to visualize potential outliers.")


from sklearn.preprocessing import StandardScaler


features_to_scale = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in features_to_scale:
    features_to_scale.remove('Engine Condition')


scaler = StandardScaler()


df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

print("Numerical features scaled successfully using StandardScaler.")
print("DataFrame head after scaling:")
print(df.head())



import matplotlib.pyplot as plt
import seaborn as sns

correlation_matrix = df.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Engine Sensor Data')
plt.show()

print("Correlation heatmap generated successfully.")



import matplotlib.pyplot as plt
import seaborn as sns


features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in features:
    features.remove('Engine Condition')


plt.figure(figsize=(18, 12))

for i, feature in enumerate(features):
    plt.subplot(2, 3, i + 1) 
    sns.boxplot(x='Engine Condition', y=feature, data=df)
    plt.title(f'{feature} vs. Engine Condition')
    plt.xlabel('Engine Condition')
    plt.ylabel(feature)

plt.tight_layout()
plt.show()

print("Comparison plots (box plots) of features vs. Engine Condition generated successfully.")


import matplotlib.pyplot as plt
import seaborn as sns


features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in features:
    features.remove('Engine Condition')


plt.figure(figsize=(18, 12))

for i, feature in enumerate(features):
    plt.subplot(2, 3, i + 1) 
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

print("Distribution plots for critical features generated successfully.")


X = df.drop('Engine Condition', axis=1)
y = df['Engine Condition']

print("Features (X) and target variable (y) separated.")
print("X shape:", X.shape)
print("y shape:", y.shape)

"""**Reasoning**:
Now that features (X) and target (y) have been separated, the next step is to split the dataset into training and testing sets to prepare for model training and evaluation, as per instructions 2 of the subtask.


"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Dataset split into training and testing sets.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(random_state=42)

rf_classifier.fit(X_train, y_train)

print("Random Forest Classifier instantiated and trained successfully.")


y_pred = rf_classifier.predict(X_test)

print("Predictions made on the test set.")


from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)
print("Model evaluation complete.")


import numpy as np

df['Remaining Useful Life (RUL)'] = np.random.randint(10, 100, size=len(df))

print("Dummy 'Remaining Useful Life (RUL)' column created.")
print(df[['Engine Condition', 'Remaining Useful Life (RUL)']].head())


X_reg = df.drop(['Engine Condition', 'Remaining Useful Life (RUL)'], axis=1)
y_reg = df['Remaining Useful Life (RUL)']

print("Features (X_reg) and target variable (y_reg) separated for regression.")
print(f"X_reg shape: {X_reg.shape}")
print(f"y_reg shape: {y_reg.shape}")



from sklearn.model_selection import train_test_split

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

print("Dataset split into training and testing sets for regression.")
print(f"X_reg_train shape: {X_reg_train.shape}")
print(f"X_reg_test shape: {X_reg_test.shape}")
print(f"y_reg_train shape: {y_reg_train.shape}")
print(f"y_reg_test shape: {y_reg_test.shape}")



from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(random_state=42)

rf_regressor.fit(X_reg_train, y_reg_train)

print("Random Forest Regressor instantiated and trained successfully.")



y_reg_pred = rf_regressor.predict(X_reg_test)

print("Predictions made on the regression test set.")


from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_reg_test, y_reg_pred)


rmse = np.sqrt(mse)

print(f"\nRegression Model RMSE: {rmse:.4f}")
print("Regression model evaluation complete.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

clf_feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)

clf_feature_importances = clf_feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=clf_feature_importances.values, y=clf_feature_importances.index, palette='viridis')
plt.title('Feature Importances for Engine Condition Classification')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

print("Feature importances for classification model generated and plotted.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

clf_feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)

clf_feature_importances = clf_feature_importances.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=clf_feature_importances.values, y=clf_feature_importances.index, hue=clf_feature_importances.index, palette='viridis', legend=False)
plt.title('Feature Importances for Engine Condition Classification')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

print("Feature importances for classification model generated and plotted.")

reg_feature_importances = pd.Series(rf_regressor.feature_importances_, index=X_reg.columns)

reg_feature_importances = reg_feature_importances.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=reg_feature_importances.values, y=reg_feature_importances.index, hue=reg_feature_importances.index, palette='plasma', legend=False)
plt.title('Feature Importances for RUL Regression')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

print("Feature importances for regression model generated and plotted.")


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 10))

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in numerical_cols:
    numerical_cols.remove('Engine Condition')

for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i + 1) 
    sns.boxplot(y=df[col])
    plt.title(f'Box plot of {col}')
    plt.ylabel('')

plt.tight_layout()
plt.show()
plt.savefig('boxplot_outliers.png')

print("Box plots generated and saved to 'boxplot_outliers.png'.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/engine_data.csv')

plt.figure(figsize=(15, 10))

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

if 'Engine Condition' in numerical_cols:
    numerical_cols.remove('Engine Condition')

for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i + 1) 
    sns.boxplot(y=df[col])
    plt.title(f'Box plot of {col}')
    plt.ylabel('')

plt.tight_layout()
plt.show()
plt.savefig('boxplot_outliers.png')

print("Box plots generated and saved to 'boxplot_outliers.png'.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/engine_data.csv')

plt.figure(figsize=(15, 10))

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

if 'Engine Condition' in numerical_cols:
    numerical_cols.remove('Engine Condition')

for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i + 1) 
    sns.boxplot(y=df[col])
    plt.title(f'Box plot of {col}')
    plt.ylabel('')

plt.tight_layout()
plt.show()
plt.savefig('boxplot_outliers.png')

print("Box plots generated and saved to 'boxplot_outliers.png'.")



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np


df = pd.read_csv('/content/engine_data.csv')
print("DataFrame reloaded for initial state.")

numerical_cols_initial = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in numerical_cols_initial:
    numerical_cols_initial.remove('Engine Condition')

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_initial):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Box plot of {col}')
    plt.ylabel('')
plt.tight_layout()
plt.show()
plt.savefig('boxplot_outliers.png')
print("Box plots for outlier detection generated and saved to 'boxplot_outliers.png'.")

features_to_scale = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in features_to_scale:
    features_to_scale.remove('Engine Condition')

scaler = StandardScaler()

df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
print("Numerical features scaled successfully.")

correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Engine Sensor Data (Scaled)')
plt.show()
plt.savefig('correlation_heatmap.png')
print("Correlation heatmap generated and saved to 'correlation_heatmap.png'.")


features_scaled = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in features_scaled:
    features_scaled.remove('Engine Condition')

plt.figure(figsize=(18, 12))
for i, feature in enumerate(features_scaled):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='Engine Condition', y=feature, data=df)
    plt.title(f'{feature} vs. Engine Condition')
    plt.xlabel('Engine Condition')
    plt.ylabel(feature)
plt.tight_layout()
plt.show()
plt.savefig('feature_vs_engine_condition_boxplots.png')
print("Comparison plots of features vs. Engine Condition generated and saved to 'feature_vs_engine_condition_boxplots.png'.")

plt.figure(figsize=(18, 12))
for i, feature in enumerate(features_scaled):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
plt.savefig('feature_distributions.png')
print("Distribution plots for critical features generated and saved to 'feature_distributions.png'.")

df['Remaining Useful Life (RUL)'] = np.random.randint(10, 100, size=len(df))
print("Dummy 'Remaining Useful Life (RUL)' column added.")

X = df.drop('Engine Condition', axis=1)
y = df['Engine Condition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
print("Random Forest Classifier re-trained.")


X_reg = df.drop(['Engine Condition', 'Remaining Useful Life (RUL)'], axis=1)
y_reg = df['Remaining Useful Life (RUL)']
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_reg_train, y_reg_train)
print("Random Forest Regressor re-trained.")

clf_feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
clf_feature_importances = clf_feature_importances.sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=clf_feature_importances.values, y=clf_feature_importances.index, hue=clf_feature_importances.index, palette='viridis', legend=False)
plt.title('Feature Importances for Engine Condition Classification')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
plt.savefig('classification_feature_importance.png')
print("Feature importances for classification model generated and saved to 'classification_feature_importance.png'.")


reg_feature_importances = pd.Series(rf_regressor.feature_importances_, index=X_reg.columns)
reg_feature_importances = reg_feature_importances.sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=reg_feature_importances.values, y=reg_feature_importances.index, hue=reg_feature_importances.index, palette='plasma', legend=False)
plt.title('Feature Importances for RUL Regression')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
plt.savefig('regression_feature_importance.png')
print("Feature importances for regression model generated and saved to 'regression_feature_importance.png'.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np


df = pd.read_csv('/content/engine_data.csv')
print("DataFrame reloaded for initial state.")

numerical_cols_initial = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in numerical_cols_initial:
    numerical_cols_initial.remove('Engine Condition')


plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_initial):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Box plot of {col}')
    plt.ylabel('')
plt.tight_layout()
plt.show()
plt.savefig('boxplot_outliers.png')
print("Box plots for outlier detection generated and saved to 'boxplot_outliers.png'.")

features_to_scale = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in features_to_scale:
    features_to_scale.remove('Engine Condition')


scaler = StandardScaler()


df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
print("Numerical features scaled successfully.")

correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Engine Sensor Data (Scaled)')
plt.show()
plt.savefig('correlation_heatmap.png')
print("Correlation heatmap generated and saved to 'correlation_heatmap.png'.")


features_scaled = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in features_scaled:
    features_scaled.remove('Engine Condition')

plt.figure(figsize=(18, 12))
for i, feature in enumerate(features_scaled):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='Engine Condition', y=feature, data=df)
    plt.title(f'{feature} vs. Engine Condition')
    plt.xlabel('Engine Condition')
    plt.ylabel(feature)
plt.tight_layout()
plt.show()
plt.savefig('feature_vs_engine_condition_boxplots.png')
print("Comparison plots of features vs. Engine Condition generated and saved to 'feature_vs_engine_condition_boxplots.png'.")

plt.figure(figsize=(18, 12))
for i, feature in enumerate(features_scaled):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
plt.savefig('feature_distributions.png')
print("Distribution plots for critical features generated and saved to 'feature_distributions.png'.")


df['Remaining Useful Life (RUL)'] = np.random.randint(10, 100, size=len(df))
print("Dummy 'Remaining Useful Life (RUL)' column added.")

X = df.drop('Engine Condition', axis=1)
y = df['Engine Condition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
print("Random Forest Classifier re-trained.")

X_reg = df.drop(['Engine Condition', 'Remaining Useful Life (RUL)'], axis=1)
y_reg = df['Remaining Useful Life (RUL)']
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_reg_train, y_reg_train)
print("Random Forest Regressor re-trained.")

clf_feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
clf_feature_importances = clf_feature_importances.sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=clf_feature_importances.values, y=clf_feature_importances.index, hue=clf_feature_importances.index, palette='viridis', legend=False)
plt.title('Feature Importances for Engine Condition Classification')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
plt.savefig('classification_feature_importance.png')
print("Feature importances for classification model generated and saved to 'classification_feature_importance.png'.")


reg_feature_importances = pd.Series(rf_regressor.feature_importances_, index=X_reg.columns)
reg_feature_importances = reg_feature_importances.sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=reg_feature_importances.values, y=reg_feature_importances.index, hue=reg_feature_importances.index, palette='plasma', legend=False)
plt.title('Feature Importances for RUL Regression')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
plt.savefig('regression_feature_importance.png')
print("Feature importances for regression model generated and saved to 'regression_feature_importance.png'.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np


df = pd.read_csv('/content/engine_data.csv')
print("DataFrame reloaded for initial state.")


numerical_cols_initial = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in numerical_cols_initial:
    numerical_cols_initial.remove('Engine Condition')


plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_initial):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Box plot of {col}')
    plt.ylabel('')
plt.tight_layout()
plt.show()
plt.savefig('boxplot_outliers.png')
print("Box plots for outlier detection generated and saved to 'boxplot_outliers.png'.")


features_to_scale = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in features_to_scale:
    features_to_scale.remove('Engine Condition')


scaler = StandardScaler()


df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
print("Numerical features scaled successfully.")


correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Engine Sensor Data (Scaled)')
plt.show()
plt.savefig('correlation_heatmap.png')
print("Correlation heatmap generated and saved to 'correlation_heatmap.png'.")


features_scaled = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in features_scaled:
    features_scaled.remove('Engine Condition')

plt.figure(figsize=(18, 12))
for i, feature in enumerate(features_scaled):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='Engine Condition', y=feature, data=df)
    plt.title(f'{feature} vs. Engine Condition')
    plt.xlabel('Engine Condition')
    plt.ylabel(feature)
plt.tight_layout()
plt.show()
plt.savefig('feature_vs_engine_condition_boxplots.png')
print("Comparison plots of features vs. Engine Condition generated and saved to 'feature_vs_engine_condition_boxplots.png'.")


plt.figure(figsize=(18, 12))
for i, feature in enumerate(features_scaled):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
plt.savefig('feature_distributions.png')
print("Distribution plots for critical features generated and saved to 'feature_distributions.png'.")

df['Remaining Useful Life (RUL)'] = np.random.randint(10, 100, size=len(df))
print("Dummy 'Remaining Useful Life (RUL)' column added.")

X = df.drop('Engine Condition', axis=1)
y = df['Engine Condition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
print("Random Forest Classifier re-trained.")

X_reg = df.drop(['Engine Condition', 'Remaining Useful Life (RUL)'], axis=1)
y_reg = df['Remaining Useful Life (RUL)']
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_reg_train, y_reg_train)
print("Random Forest Regressor re-trained.")


clf_feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
clf_feature_importances = clf_feature_importances.sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=clf_feature_importances.values, y=clf_feature_importances.index, hue=clf_feature_importances.index, palette='viridis', legend=False)
plt.title('Feature Importances for Engine Condition Classification')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
plt.savefig('classification_feature_importance.png')
print("Feature importances for classification model generated and saved to 'classification_feature_importance.png'.")


reg_feature_importances = pd.Series(rf_regressor.feature_importances_, index=X_reg.columns)
reg_feature_importances = reg_feature_importances.sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=reg_feature_importances.values, y=reg_feature_importances.index, hue=reg_feature_importances.index, palette='plasma', legend=False)
plt.title('Feature Importances for RUL Regression')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
plt.savefig('regression_feature_importance.png')
print("Feature importances for regression model generated and saved to 'regression_feature_importance.png'.")



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np


df = pd.read_csv('/content/engine_data.csv')
print("DataFrame reloaded for initial state.")

numerical_cols_initial = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in numerical_cols_initial:
    numerical_cols_initial.remove('Engine Condition')

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_initial):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Box plot of {col}')
    plt.ylabel('')
plt.tight_layout()
plt.show()
plt.savefig('boxplot_outliers.png')
print("Box plots for outlier detection generated and saved to 'boxplot_outliers.png'.")


features_to_scale = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in features_to_scale:
    features_to_scale.remove('Engine Condition')

scaler = StandardScaler()

df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
print("Numerical features scaled successfully.")

correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Engine Sensor Data (Scaled)')
plt.show()
plt.savefig('correlation_heatmap.png')
print("Correlation heatmap generated and saved to 'correlation_heatmap.png'.")


features_scaled = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in features_scaled:
    features_scaled.remove('Engine Condition')

plt.figure(figsize=(18, 12))
for i, feature in enumerate(features_scaled):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='Engine Condition', y=feature, data=df)
    plt.title(f'{feature} vs. Engine Condition')
    plt.xlabel('Engine Condition')
    plt.ylabel(feature)
plt.tight_layout()
plt.show()
plt.savefig('feature_vs_engine_condition_boxplots.png')
print("Comparison plots of features vs. Engine Condition generated and saved to 'feature_vs_engine_condition_boxplots.png'.")


plt.figure(figsize=(18, 12))
for i, feature in enumerate(features_scaled):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
plt.savefig('feature_distributions.png')
print("Distribution plots for critical features generated and saved to 'feature_distributions.png'.")

df['Remaining Useful Life (RUL)'] = np.random.randint(10, 100, size=len(df))
print("Dummy 'Remaining Useful Life (RUL)' column added.")


X = df.drop('Engine Condition', axis=1)
y = df['Engine Condition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
print("Random Forest Classifier re-trained.")

X_reg = df.drop(['Engine Condition', 'Remaining Useful Life (RUL)'], axis=1)
y_reg = df['Remaining Useful Life (RUL)']
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_reg_train, y_reg_train)
print("Random Forest Regressor re-trained.")

clf_feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
clf_feature_importances = clf_feature_importances.sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=clf_feature_importances.values, y=clf_feature_importances.index, hue=clf_feature_importances.index, palette='viridis', legend=False)
plt.title('Feature Importances for Engine Condition Classification')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
plt.savefig('classification_feature_importance.png')
print("Feature importances for classification model generated and saved to 'classification_feature_importance.png'.")


reg_feature_importances = pd.Series(rf_regressor.feature_importances_, index=X_reg.columns)
reg_feature_importances = reg_feature_importances.sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=reg_feature_importances.values, y=reg_feature_importances.index, hue=reg_feature_importances.index, palette='plasma', legend=False)
plt.title('Feature Importances for RUL Regression')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
plt.savefig('regression_feature_importance.png')
print("Feature importances for regression model generated and saved to 'regression_feature_importance.png'.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np


df = pd.read_csv('/content/engine_data.csv')
print("DataFrame reloaded for initial state.")


numerical_cols_initial = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in numerical_cols_initial:
    numerical_cols_initial.remove('Engine Condition')


plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_initial):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Box plot of {col}')
    plt.ylabel('')
plt.tight_layout()
plt.show()
plt.savefig('boxplot_outliers.png')
print("Box plots for outlier detection generated and saved to 'boxplot_outliers.png'.")


features_to_scale = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in features_to_scale:
    features_to_scale.remove('Engine Condition')


scaler = StandardScaler()

df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
print("Numerical features scaled successfully.")


correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Engine Sensor Data (Scaled)')
plt.show()
plt.savefig('correlation_heatmap.png')
print("Correlation heatmap generated and saved to 'correlation_heatmap.png'.")


features_scaled = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in features_scaled:
    features_scaled.remove('Engine Condition')

plt.figure(figsize=(18, 12))
for i, feature in enumerate(features_scaled):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='Engine Condition', y=feature, data=df)
    plt.title(f'{feature} vs. Engine Condition')
    plt.xlabel('Engine Condition')
    plt.ylabel(feature)
plt.tight_layout()
plt.show()
plt.savefig('feature_vs_engine_condition_boxplots.png')
print("Comparison plots of features vs. Engine Condition generated and saved to 'feature_vs_engine_condition_boxplots.png'.")


plt.figure(figsize=(18, 12))
for i, feature in enumerate(features_scaled):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
plt.savefig('feature_distributions.png')
print("Distribution plots for critical features generated and saved to 'feature_distributions.png'.")


df['Remaining Useful Life (RUL)'] = np.random.randint(10, 100, size=len(df))
print("Dummy 'Remaining Useful Life (RUL)' column added.")


X = df.drop('Engine Condition', axis=1)
y = df['Engine Condition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
print("Random Forest Classifier re-trained.")


X_reg = df.drop(['Engine Condition', 'Remaining Useful Life (RUL)'], axis=1)
y_reg = df['Remaining Useful Life (RUL)']
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_reg_train, y_reg_train)
print("Random Forest Regressor re-trained.")


clf_feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
clf_feature_importances = clf_feature_importances.sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=clf_feature_importances.values, y=clf_feature_importances.index, hue=clf_feature_importances.index, palette='viridis', legend=False)
plt.title('Feature Importances for Engine Condition Classification')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
plt.savefig('classification_feature_importance.png')
print("Feature importances for classification model generated and saved to 'classification_feature_importance.png'.")


reg_feature_importances = pd.Series(rf_regressor.feature_importances_, index=X_reg.columns)
reg_feature_importances = reg_feature_importances.sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=reg_feature_importances.values, y=reg_feature_importances.index, hue=reg_feature_importances.index, palette='plasma', legend=False)
plt.title('Feature Importances for RUL Regression')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
plt.savefig('regression_feature_importance.png')
print("Feature importances for regression model generated and saved to 'regression_feature_importance.png'.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import os


file_path = '/content/engine_data.csv'


if not os.path.exists(file_path):
    print(f"File '{file_path}' not found. Creating a dummy file for demonstration.")
    dummy_data = {
        'Engine rpm': np.random.randint(500, 1500, 100),
        'Lub oil pressure': np.random.uniform(2, 5, 100),
        'Fuel pressure': np.random.uniform(5, 15, 100),
        'Coolant pressure': np.random.uniform(1, 4, 100),
        'lub oil temp': np.random.uniform(70, 90, 100),
        'Coolant temp': np.random.uniform(75, 95, 100),
        'Engine Condition': np.random.randint(0, 2, 100)
    }
    pd.DataFrame(dummy_data).to_csv(file_path, index=False)
    print("Dummy 'engine_data.csv' created.")


df = pd.read_csv(file_path)
print("DataFrame reloaded for initial state.")


numerical_cols_initial = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in numerical_cols_initial:
    numerical_cols_initial.remove('Engine Condition')


plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_initial):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Box plot of {col}')
    plt.ylabel('')
plt.tight_layout()
plt.show()
plt.savefig('boxplot_outliers.png')
print("Box plots for outlier detection generated and saved to 'boxplot_outliers.png'.")


features_to_scale = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in features_to_scale:
    features_to_scale.remove('Engine Condition')

scaler = StandardScaler()


df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
print("Numerical features scaled successfully.")

correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Engine Sensor Data (Scaled)')
plt.show()
plt.savefig('correlation_heatmap.png')
print("Correlation heatmap generated and saved to 'correlation_heatmap.png'.")


features_scaled = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Engine Condition' in features_scaled:
    features_scaled.remove('Engine Condition')

plt.figure(figsize=(18, 12))
for i, feature in enumerate(features_scaled):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='Engine Condition', y=feature, data=df)
    plt.title(f'{feature} vs. Engine Condition')
    plt.xlabel('Engine Condition')
    plt.ylabel(feature)
plt.tight_layout()
plt.show()
plt.savefig('feature_vs_engine_condition_boxplots.png')
print("Comparison plots of features vs. Engine Condition generated and saved to 'feature_vs_engine_condition_boxplots.png'.")

plt.figure(figsize=(18, 12))
for i, feature in enumerate(features_scaled):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
plt.savefig('feature_distributions.png')
print("Distribution plots for critical features generated and saved to 'feature_distributions.png'.")


df['Remaining Useful Life (RUL)'] = np.random.randint(10, 100, size=len(df))
print("Dummy 'Remaining Useful Life (RUL)' column added.")

X = df.drop('Engine Condition', axis=1)
y = df['Engine Condition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
print("Random Forest Classifier re-trained.")


X_reg = df.drop(['Engine Condition', 'Remaining Useful Life (RUL)'], axis=1)
y_reg = df['Remaining Useful Life (RUL)']
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_reg_train, y_reg_train)
print("Random Forest Regressor re-trained.")


clf_feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
clf_feature_importances = clf_feature_importances.sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=clf_feature_importances.values, y=clf_feature_importances.index, hue=clf_feature_importances.index, palette='viridis', legend=False)
plt.title('Feature Importances for Engine Condition Classification')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
plt.savefig('classification_feature_importance.png')
print("Feature importances for classification model generated and saved to 'classification_feature_importance.png'.")


reg_feature_importances = pd.Series(rf_regressor.feature_importances_, index=X_reg.columns)
reg_feature_importances = reg_feature_importances.sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=reg_feature_importances.values, y=reg_feature_importances.index, hue=reg_feature_importances.index, palette='plasma', legend=False)
plt.title('Feature Importances for RUL Regression')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
plt.savefig('regression_feature_importance.png')
print("Feature importances for regression model generated and saved to 'regression_feature_importance.png'.")


