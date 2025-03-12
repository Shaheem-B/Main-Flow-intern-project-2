import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
file_path = '/content/Global_Superstore2.xlsx'
df = pd.read_excel(file_path, header=0)
# Drop unnamed columns that may be present due to Excel formatting issues
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Display dataset structure
print("\nDataset Columns:", df.columns)
print("Initial Dataset Shape:", df.shape)

# Handling Missing Values
def handle_missing_values(df):
    df = df.copy()
    for column in df.columns:
        if df[column].dtype == 'O':
            if df[column].isnull().sum() > 0:
                df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].mean())
    return df

df = handle_missing_values(df)
print("\nMissing Values After Handling:")
print(df.isnull().sum())
# Removing Duplicates
df = df.drop_duplicates()
print("\nDataset Shape After Removing Duplicates:", df.shape)

# Outlier Removal Functions with Column Existence Check
def remove_outliers_iqr(df, column):
    if column not in df.columns:
        print(f"Skipping {column} (not found in dataset)")
        return df
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()

# Using Z-score method
def remove_outliers_zscore(df, column, threshold=3):
    if column not in df.columns:
        print(f"Skipping {column} (not found in dataset)")
        return df
    df = df.copy()
    mean = df[column].mean()
    std = df[column].std()
    df['Z_score'] = (df[column] - mean) / std
    df = df[df['Z_score'].abs() <= threshold].copy()
    df = df.drop(columns=['Z_score'])
    return df

# Apply outlier detection only for numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df = remove_outliers_iqr(df, col)
    df = remove_outliers_zscore(df, col)
print("\nDataset Shape After Handling Outliers:", df.shape)

# Statistical Measures
print("\nBasic Statistical Measures:")
print(df.describe())

# Compute and Display Correlations (only numeric columns)
numeric_df = df.select_dtypes(include=[np.number])
if not numeric_df.empty:
    correlation_matrix = numeric_df.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Heatmap of Correlation Matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix Heatmap')
    plt.show()
else:
    print("\nNo numeric columns found for correlation analysis.")

# Save cleaned dataset locally
save_path = '/content/cleaned_dataset.xlsx'
df.to_excel(save_path, index=False)
print("\nCleaned dataset saved as cleaned_dataset.xlsx in the working directory")
# Display basic info
print("Dataset Overview:\n", df.info())
print("\nSummary Statistics:\n", df.describe())
# Select numerical columns for analysis
numerical_cols = df.select_dtypes(include=['number']).columns

# 1. Histograms for numerical data
df[numerical_cols].hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.suptitle('Histograms of Numerical Features', fontsize=16)
plt.show()

# 2. Boxplots to identify outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numerical_cols])
plt.xticks(rotation=90)
plt.title('Boxplots of Numerical Features (Outlier Detection)')
plt.show()

# 3. Heatmap for feature correlations
plt.figure(figsize=(10, 6))
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Display basic information
print("Dataset Shape:", df.shape)
# Display column-wise missing values
print("\nMissing Values:\n", df.isnull().sum())
# Display data types of columns
print("\nData Types:\n", df.dtypes)
# Display first 10 rows
print("\nFirst 10 Rows:\n", df.head(10))
# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing values
for column in df.columns:
    if df[column].dtype == 'float64' or df[column].dtype == 'int64':
        df[column].fillna(df[column].mean(), inplace=True)
    else:
        df[column].fillna(df[column].mode()[0], inplace=True)

# Convert 'Date' column to datetime (assuming the column is named 'Date')
date_column = 'Date'
if date_column in df.columns:
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
# Display the first few rows after processing
print(df.head(10))

# Save the cleaned dataset to a new Excel file
cleaned_file_path = "/content/Global_Superstore_Cleaned.xlsx"
df.to_excel(cleaned_file_path, index=False)
print(f"Cleaned dataset saved as: {cleaned_file_path}")

# Convert 'Date' column to datetime if not already done
date_column = 'Date'
if date_column in df.columns:
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
# Ensure the dataset has required columns
if 'Sales' in df.columns and date_column in df.columns:
    sales_trend = df.groupby(date_column)['Sales'].sum()

    # Plot time series graph
    plt.figure(figsize=(12, 6))
    sales_trend.plot()
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.title('Sales Trend Over Time')
    plt.grid(True)
    plt.show()

# Scatter plot for Profit vs Discount
if 'Profit' in df.columns and 'Discount' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Discount', y='Profit', alpha=0.5)
    plt.xlabel('Discount')
    plt.ylabel('Profit')
    plt.title('Profit vs Discount')
    plt.grid(True)
    plt.show()

# Sales distribution by Region using bar plot
if 'Region' in df.columns and 'Sales' in df.columns:
    plt.figure(figsize=(10, 6))
    region_sales = df.groupby('Region')['Sales'].sum().sort_values()
    region_sales.plot(kind='bar', color='skyblue')
    plt.xlabel('Region')
    plt.ylabel('Total Sales')
    plt.title('Sales Distribution by Region')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

# Sales distribution by Category using pie chart
if 'Category' in df.columns and 'Sales' in df.columns:
    plt.figure(figsize=(8, 8))
    category_sales = df.groupby('Category')['Sales'].sum()
    category_sales.plot(kind='pie', autopct='%1.1f%%', startangle=140, colormap='Set3')
    plt.ylabel('')
    plt.title('Sales Distribution by Category')
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Ensure the required columns exist
if {'Profit', 'Discount', 'Sales'}.issubset(df.columns):
    # Selecting features (X) and target variable (y)
    X = df[['Profit', 'Discount']]
    y = df['Sales']
    # Splitting the dataset into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Predict on test data
    y_pred = model.predict(X_test)

    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # 1. Scatter Plot: Actual vs Predicted Sales
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Reference line
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales (Linear Regression)")
    plt.grid(True)
    plt.show()

    # 2. Residual Plot: Checking for random error distribution
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=30, kde=True, color="green")
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel("Residual (Error)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.grid(True)
    plt.show()

    # 3. Feature Importance: Impact of Profit & Discount on Sales
    feature_importance = pd.Series(model.coef_, index=X.columns)
    plt.figure(figsize=(8, 5))
    feature_importance.plot(kind='bar', color=['darkblue', 'skyblue'])
    plt.xlabel("Features")
    plt.ylabel("Coefficient Value")
    plt.title("Feature Importance (Effect on Sales)")
    plt.grid(axis='y')
    plt.show()

    # Display model coefficients
    print(f"Intercept: {model.intercept_}")
    print(f"Coefficients: {dict(zip(X.columns, model.coef_))}")
else:
    print("Required columns (Profit, Discount, Sales) not found in the dataset.")