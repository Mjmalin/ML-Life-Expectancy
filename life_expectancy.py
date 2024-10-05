import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r'/Users/maxwellmalinofsky/Desktop/Portfolio/life_expectancy_data.csv')

pd.set_option('display.max.columns', None)

print(df)

# strip blank spaces from left and right of column names
df.columns = df.columns.str.strip()
print(df.columns)

# Correlation analysis between target variable Life expectancy and other variables
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()
le_correlation = correlation_matrix['Life expectancy'].sort_values(ascending=False)
pd.set_option('display.max.rows', None)

print(le_correlation)

# Drop null rows and load the data set
df = df[['Life expectancy', 'Schooling', 'Income composition of resources', 'BMI', 'GDP', 'Alcohol', 'percentage expenditure']].dropna()

# Checking correlations
correlation_matrix = df[['Life expectancy', 'Schooling', 'Income composition of resources', 'BMI', 'GDP', 'Alcohol', 'percentage expenditure']].corr()
print(correlation_matrix)

y = df['Life expectancy']
x = df[['Schooling', 'Income composition of resources', 'BMI', 'GDP', 'Alcohol', 'percentage expenditure']]
print(y.shape, x.shape)

# Split the dataset into training, cross-validation, and test sets
x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1)
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

# Initialize lists to save the errors
results = []

# Loop over polynomial degrees
for degree in range(1, 11):
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    # Transform the training set
    X_train_mapped = poly.fit_transform(x_train)

    # Loop over different alpha values for Ridge regression
    for alpha in [100, 10, 1, 0.5, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Scale the training set
        X_train_mapped_scaled = scaler.fit_transform(X_train_mapped)

        # Initialize the Ridge model with the current alpha value
        ridge_model = Ridge(alpha=alpha)

        # Train the model
        ridge_model.fit(X_train_mapped_scaled, y_train)

        # Compute the training MSE
        yhat_train = ridge_model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train, yhat_train) / 2

        # Transform and scale the cross-validation set
        X_cv_mapped = poly.transform(x_cv)
        X_cv_mapped_scaled = scaler.transform(X_cv_mapped)

        # Compute the cross-validation MSE
        yhat_cv = ridge_model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv, yhat_cv) / 2

        # Store the results
        results.append({
            'degree': degree,
            'alpha': alpha,
            'train_mse': train_mse,
            'cv_mse': cv_mse
        })

        # print(f"degree: {degree}, alpha: {alpha}, train_mse: {train_mse}, cv_mse: {cv_mse}")

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Find the row with the minimum cross-validation MSE
min_cv_mse_row = results_df.loc[results_df['cv_mse'].idxmin()]

print("Lowest CV MSE:")
print(min_cv_mse_row)

# Set display options to show all rows
pd.set_option('display.max_rows', None)

X_test_mapped = (PolynomialFeatures(degree=2, include_bias=False)).fit_transform(x_test)

X_test_mapped_scaled = scaler.fit_transform(X_test_mapped)

ridge_model = Ridge(alpha=0.01)

# Train the model
ridge_model.fit(X_test_mapped_scaled, y_test)

# Compute the test MSE
yhat = ridge_model.predict(X_test_mapped_scaled)
test_mse = mean_squared_error(y_test, yhat) / 2

# print(f"Training MSE: {train_mses[degree-1]:.2f}")
print(f"Cross Validation MSE: {min_cv_mse_row['cv_mse']:.2f}")
print(f"Test MSE: {test_mse:.2f}")
