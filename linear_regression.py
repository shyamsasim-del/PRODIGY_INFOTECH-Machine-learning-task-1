import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample dataset for house prices
# Features: Square Footage, Bedrooms, Bathrooms
# Target: Price

def generate_dataset(n_samples=200):
    """
    Generate a sample dataset of house properties and prices
    """
    square_footage = np.random.uniform(1000, 5000, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.uniform(1, 4, n_samples)
    
    # Generate prices based on linear relationship with some noise
    # Price = 100 * sq_ft + 50000 * bedrooms + 30000 * bathrooms + noise
    prices = (100 * square_footage + 
              50000 * bedrooms + 
              30000 * bathrooms + 
              np.random.normal(0, 50000, n_samples))
    
    # Create DataFrame
    data = pd.DataFrame({
        'Square_Footage': square_footage,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Price': prices
    })
    
    return data

def train_model(X, y):
    """
    Train a linear regression model
    """
    model = LinearRegression()
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return y_pred, mse, rmse, r2

def predict_price(model, square_footage, bedrooms, bathrooms):
    """
    Predict the price of a house given its features
    """
    input_data = np.array([[square_footage, bedrooms, bathrooms]])
    predicted_price = model.predict(input_data)[0]
    return predicted_price

def main():
    print("="*60)
    print("Linear Regression Model for House Price Prediction")
    print("="*60)
    
    # Generate dataset
    print("\n1. Generating dataset...")
    data = generate_dataset(n_samples=200)
    print(f"   Dataset shape: {data.shape}")
    print(f"\n   Dataset Preview:")
    print(data.head(10))
    print(f"\n   Dataset Statistics:")
    print(data.describe())
    
    # Prepare features and target
    X = data[['Square_Footage', 'Bedrooms', 'Bathrooms']]
    y = data['Price']
    
    # Split data into training and testing sets
    print("\n2. Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Training set size: {X_train.shape[0]}")
    print(f"   Testing set size: {X_test.shape[0]}")
    
    # Train the model
    print("\n3. Training Linear Regression Model...")
    model = train_model(X_train, y_train)
    print("   Model trained successfully!")
    
    # Display model coefficients
    print("\n4. Model Coefficients:")
    print(f"   Intercept: ${model.intercept_:,.2f}")
    print(f"   Square Footage Coefficient: ${model.coef_[0]:,.2f} per sq ft")
    print(f"   Bedrooms Coefficient: ${model.coef_[1]:,.2f} per bedroom")
    print(f"   Bathrooms Coefficient: ${model.coef_[2]:,.2f} per bathroom")
    
    # Evaluate the model
    print("\n5. Model Evaluation on Test Set:")
    y_pred, mse, rmse, r2 = evaluate_model(model, X_test, y_test)
    print(f"   Mean Squared Error (MSE): ${mse:,.2f}")
    print(f"   Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"   R² Score: {r2:.4f}")
    
    # Make predictions on new data
    print("\n6. Predictions on New Houses:")
    test_cases = [
        (2000, 3, 2),
        (3500, 4, 3),
        (1500, 2, 1),
    ]
    
    for sq_ft, beds, baths in test_cases:
        predicted_price = predict_price(model, sq_ft, beds, baths)
        print(f"   {sq_ft} sq ft, {beds} beds, {baths} baths → ${predicted_price:,.2f}")
    
    # Visualization
    print("\n7. Creating visualizations...")
    
    # Plot 1: Actual vs Predicted Prices
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Actual vs Predicted House Prices')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = y_test - y_pred
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Price ($)')
    plt.ylabel('Residuals ($)')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('house_price_prediction.png', dpi=300, bbox_inches='tight')
    print("   Visualization saved as 'house_price_prediction.png'")
    plt.show()
    
    # Additional analysis: Feature importance visualization
    plt.figure(figsize=(8, 5))
    features = ['Square Footage', 'Bedrooms', 'Bathrooms']
    coefficients = model.coef_
    
    plt.barh(features, coefficients, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xlabel('Coefficient Value ($)')
    plt.title('Feature Importance in Price Prediction')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("   Feature importance plot saved as 'feature_importance.png'")
    plt.show()
    
    print("\n" + "="*60)
    print("Linear Regression Analysis Complete!")
    print("="*60)
    
    return model, X_test, y_test, y_pred

if __name__ == "__main__":
    model, X_test, y_test, y_pred = main()
