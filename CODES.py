# Crop Price Prediction Project
# ============================

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Step 2: Generate sample crop price data (in real project, you would load actual data)
# Creating synthetic data for demonstration
np.random.seed(42)

# Sample data for 5 years (2020-2024) across 4 seasons for 3 crops
crops = ['Rice', 'Wheat', 'Corn']
seasons = ['Winter', 'Spring', 'Summer', 'Fall']
years = range(2020, 2025)

data = []
for crop in crops:
    base_price = np.random.randint(50, 200)  # Base price varies by crop
    for year in years:
        for season in seasons:
            # Factors affecting price
            rainfall = np.random.uniform(50, 200)  # mm
            temperature = np.random.uniform(15, 35)  # Celsius
            production = np.random.uniform(50, 150)  # thousand tons
            
            # Season effects
            season_effect = {
                'Winter': np.random.uniform(0.8, 1.0),
                'Spring': np.random.uniform(0.9, 1.1),
                'Summer': np.random.uniform(1.0, 1.2),
                'Fall': np.random.uniform(0.9, 1.1)
            }
            
            # Year trend (slight increase over years)
            year_effect = 1 + (year - 2020) * 0.03
            
            # Price calculation with some randomness
            price = base_price * season_effect[season] * year_effect * (1 + 0.01 * temperature) * (1 - 0.001 * rainfall) * (1 - 0.002 * production)
            price = price + np.random.normal(0, price * 0.05)  # Add some noise
            
            data.append({
                'Crop': crop,
                'Year': year,
                'Season': season,
                'Rainfall': rainfall,
                'Temperature': temperature,
                'Production': production,
                'Price': round(price, 2)
            })

# Create DataFrame
df = pd.DataFrame(data)

# Step 3: Exploratory Data Analysis
print("Dataset Sample:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize price distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], kde=True)
plt.title('Distribution of Crop Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Visualize price by crop
plt.figure(figsize=(12, 6))
sns.boxplot(x='Crop', y='Price', data=df)
plt.title('Price Distribution by Crop')
plt.show()

# Visualize price trends over time by crop
plt.figure(figsize=(12, 8))
for crop in crops:
    crop_data = df[df['Crop'] == crop].groupby('Year')['Price'].mean().reset_index()
    plt.plot(crop_data['Year'], crop_data['Price'], marker='o', label=crop)
plt.title('Average Price Trends by Crop (2020-2024)')
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Correlation analysis
plt.figure(figsize=(10, 8))
numerical_features = ['Rainfall', 'Temperature', 'Production', 'Price', 'Year']
correlation = df[numerical_features].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Step 4: Feature Engineering and Data Preprocessing
# Create seasonal indicators
df['Year_sin'] = np.sin(2 * np.pi * df['Year'] / df['Year'].max())
df['Year_cos'] = np.cos(2 * np.pi * df['Year'] / df['Year'].max())

# Define features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Identify numerical and categorical features
numerical_features = ['Rainfall', 'Temperature', 'Production', 'Year', 'Year_sin', 'Year_cos']
categorical_features = ['Crop', 'Season']

# Create preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

# Bundle preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Step 5: Model Building - Linear Regression
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the linear regression model
print("\nTraining Linear Regression model...")
lr_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_pipeline.predict(X_test)

# Evaluate the model
print("\nLinear Regression Model Performance:")
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Mean Absolute Error: {mae_lr:.2f}")
print(f"Mean Squared Error: {mse_lr:.2f}")
print(f"Root Mean Squared Error: {rmse_lr:.2f}")
print(f"R² Score: {r2_lr:.2f}")

# Step 6: Model Building - Random Forest Regressor
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train the Random Forest model
print("\nTraining Random Forest model...")
rf_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_pipeline.predict(X_test)

# Evaluate the model
print("\nRandom Forest Model Performance:")
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Mean Absolute Error: {mae_rf:.2f}")
print(f"Mean Squared Error: {mse_rf:.2f}")
print(f"Root Mean Squared Error: {rmse_rf:.2f}")
print(f"R² Score: {r2_rf:.2f}")

# Step 7: Hyperparameter Tuning for Random Forest
print("\nPerforming hyperparameter tuning for Random Forest...")
# Define the parameter grid (limited for demonstration)
param_grid = {
    'regressor__n_estimators': [50, 100],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5]
}

# Create the grid search
grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Get the best parameters
print(f"\nBest parameters: {grid_search.best_params_}")

# Get the best model
best_rf_model = grid_search.best_estimator_

# Evaluate the tuned model
y_pred_tuned = best_rf_model.predict(X_test)

print("\nTuned Random Forest Model Performance:")
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mse_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f"Mean Absolute Error: {mae_tuned:.2f}")
print(f"Mean Squared Error: {mse_tuned:.2f}")
print(f"Root Mean Squared Error: {rmse_tuned:.2f}")
print(f"R² Score: {r2_tuned:.2f}")

# Step 8: Feature Importance Analysis
# For the Random Forest model
if hasattr(best_rf_model['regressor'], 'feature_importances_'):
    # Get feature names after preprocessing
    feature_names = []
    
    # Get numerical feature names
    feature_names.extend(numerical_features)
    
    # Get one-hot encoded feature names
    encoder = best_rf_model['preprocessor'].transformers_[1][1]['onehot']
    encoded_features = list(encoder.get_feature_names_out(categorical_features))
    feature_names.extend(encoded_features)
    
    # Get feature importances
    importances = best_rf_model['regressor'].feature_importances_
    
    # Sort feature importances
    indices = np.argsort(importances)[::-1]
    
    # Print feature ranking
    print("\nFeature ranking:")
    for f in range(min(10, len(indices))):
        if f < len(feature_names):
            print(f"{f+1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title("Top 10 Feature Importances")
    plt.bar(range(min(10, len(indices))), 
            importances[indices[:10]], 
            align="center")
    plt.xticks(range(min(10, len(indices))), 
                [feature_names[i] for i in indices[:10]], 
                rotation=90)
    plt.tight_layout()
    plt.show()

# Step 9: Model Comparison
models = ['Linear Regression', 'Random Forest', 'Tuned Random Forest']
mae_scores = [mae_lr, mae_rf, mae_tuned]
rmse_scores = [rmse_lr, rmse_rf, rmse_tuned]
r2_scores = [r2_lr, r2_rf, r2_tuned]

# Plot model comparison
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# MAE comparison
ax[0].bar(models, mae_scores, color=['blue', 'green', 'red'])
ax[0].set_title('Mean Absolute Error (Lower is better)')
ax[0].set_ylabel('MAE')
ax[0].set_ylim(0, max(mae_scores) * 1.1)
for i, v in enumerate(mae_scores):
    ax[0].text(i, v + 0.5, f"{v:.2f}", ha='center')

# RMSE comparison
ax[1].bar(models, rmse_scores, color=['blue', 'green', 'red'])
ax[1].set_title('Root Mean Squared Error (Lower is better)')
ax[1].set_ylabel('RMSE')
ax[1].set_ylim(0, max(rmse_scores) * 1.1)
for i, v in enumerate(rmse_scores):
    ax[1].text(i, v + 0.5, f"{v:.2f}", ha='center')

# R² comparison
ax[2].bar(models, r2_scores, color=['blue', 'green', 'red'])
ax[2].set_title('R² Score (Higher is better)')
ax[2].set_ylabel('R²')
ax[2].set_ylim(0, max(r2_scores) * 1.1)
for i, v in enumerate(r2_scores):
    ax[2].text(i, v - 0.05, f"{v:.2f}", ha='center')

plt.tight_layout()
plt.show()

# Step 10: Price Prediction for Future Seasons
# Create a function for future predictions
def predict_future_prices(model, crops, seasons, years, rainfall, temperature, production):
    future_data = []
    for crop in crops:
        for year in years:
            for season in seasons:
                future_data.append({
                    'Crop': crop,
                    'Year': year,
                    'Season': season,
                    'Rainfall': rainfall,
                    'Temperature': temperature,
                    'Production': production,
                    'Year_sin': np.sin(2 * np.pi * year / max(years)),
                    'Year_cos': np.cos(2 * np.pi * year / max(years))
                })
    
    future_df = pd.DataFrame(future_data)
    predictions = model.predict(future_df)
    future_df['Predicted_Price'] = predictions
    return future_df

# Predict prices for 2025
future_years = [2025]
future_rainfall = 110  # Example value
future_temperature = 25  # Example value
future_production = 100  # Example value

future_predictions = predict_future_prices(
    best_rf_model,
    crops,
    seasons,
    future_years,
    future_rainfall,
    future_temperature,
    future_production
)

print("\nPredicted Crop Prices for 2025:")
print(future_predictions[['Crop', 'Year', 'Season', 'Predicted_Price']])

# Visualize predictions for 2025
plt.figure(figsize=(12, 8))
for crop in crops:
    crop_data = future_predictions[future_predictions['Crop'] == crop]
    plt.bar(
        [f"{crop} - {season}" for season in seasons],
        crop_data['Predicted_Price'],
        alpha=0.7,
        label=f"{crop}"
    )

plt.title('Predicted Crop Prices for 2025')
plt.xlabel('Crop - Season')
plt.ylabel('Predicted Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Step 11: Save the best model (in a real project)
# import joblib
# joblib.dump(best_rf_model, 'crop_price_prediction_model.pkl')
# print("\nModel saved as 'crop_price_prediction_model.pkl'")

# Step 12: Create function for new predictions
def predict_crop_price(model, crop, year, season, rainfall, temperature, production):
    """
    Predict crop price using the trained model
    
    Parameters:
    model: Trained model pipeline
    crop: Crop name (string)
    year: Year (int)
    season: Season name (string)
    rainfall: Rainfall in mm (float)
    temperature: Temperature in Celsius (float)
    production: Production in thousand tons (float)
    
    Returns:
    float: Predicted price
    """
    # Create input data
    input_data = pd.DataFrame({
        'Crop': [crop],
        'Year': [year],
        'Season': [season],
        'Rainfall': [rainfall],
        'Temperature': [temperature],
        'Production': [production],
        'Year_sin': [np.sin(2 * np.pi * year / 2025)],
        'Year_cos': [np.cos(2 * np.pi * year / 2025)]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    return prediction[0]

# Example usage of prediction function
example_prediction = predict_crop_price(
    best_rf_model,
    'Wheat',
    2025,
    'Summer',
    120,
    28,
    95
)

print(f"\nExample prediction for Wheat in Summer 2025: ${example_prediction:.2f}")

# Final conclusion
print("\n=== Crop Price Prediction Project Summary ===")
print(f"Best performing model: Tuned Random Forest")
print(f"Model accuracy (R²): {r2_tuned:.2f}")
print(f"Mean prediction error (RMSE): {rmse_tuned:.2f}")
print("Key influential factors:")
for f in range(min(5, len(indices))):
    if f < len(feature_names):
        print(f"- {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
