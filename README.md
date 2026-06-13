# Prodigy Infotech - Machine Learning Task 1: House Price Prediction using Linear Regression

This repository contains the solution for **Machine Learning Task 1** from the **Prodigy Infotech** internship program - implementing a Linear Regression model to predict house prices based on property features.

## 📚 About Prodigy Infotech

[**Visit Prodigy Infotech**](https://www.prodigyinfotech.dev/) - Learn more about the internship program and opportunities.

## 📋 Project Overview

This project focuses on building and evaluating a **Linear Regression model** that predicts house prices based on key property features:
- **Square Footage** - The area of the house in square feet
- **Number of Bedrooms** - Count of bedrooms
- **Number of Bathrooms** - Count of bathrooms

The model is trained on a dataset and evaluated using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score.

## 🎯 Task Objective

Develop a linear regression model that:
- ✅ Analyzes the relationship between house properties and their prices
- ✅ Trains on historical data (synthetic dataset with 200 samples)
- ✅ Evaluates model performance on test data
- ✅ Makes predictions on new house properties
- ✅ Visualizes the results with meaningful plots

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/shyamsasim-del/PRODIGY_INFOTECH-Machine-learning-task-1.git

# Navigate to the directory
cd PRODIGY_INFOTECH-Machine-learning-task-1

# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
PRODIGY_INFOTECH-Machine-learning-task-1/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── linear_regression.py               # Main implementation
└── [Output files generated after running]
    ├── house_price_prediction.png     # Actual vs Predicted plot + Residuals
    └── feature_importance.png         # Feature coefficients visualization
```

## 📊 How to Run

Execute the linear regression script:

```bash
python linear_regression.py
```

### Expected Output:
- Dataset statistics and preview
- Training/testing split information
- Model coefficients (intercept and feature weights)
- Performance metrics (MSE, RMSE, R² Score)
- Price predictions for sample houses
- Two visualization files saved

## 💡 Key Features

### 1. **Data Generation**
   - Generates synthetic dataset with 200 samples
   - Features: Square Footage, Bedrooms, Bathrooms
   - Realistic price values with noise

### 2. **Model Training**
   - Uses scikit-learn's LinearRegression
   - 80-20 train-test split
   - Reproducible results (random seed = 42)

### 3. **Model Evaluation**
   - **MSE (Mean Squared Error)** - Average squared prediction error
   - **RMSE (Root Mean Squared Error)** - Standard deviation of prediction errors
   - **R² Score** - Coefficient of determination (goodness of fit)

### 4. **Predictions**
   - Sample predictions on new house properties
   - Example: 2000 sq ft, 3 bedrooms, 2 bathrooms → Predicted Price

### 5. **Visualizations**
   - **Actual vs Predicted Prices** - Scatter plot with perfect prediction line
   - **Residual Plot** - Shows prediction errors
   - **Feature Importance Chart** - Bar chart of model coefficients

## 📈 Sample Results

When you run the script, you'll get model coefficients similar to:
```
Intercept: [Generated value]
Square Footage Coefficient: ~$100 per sq ft
Bedrooms Coefficient: ~$50,000 per bedroom
Bathrooms Coefficient: ~$30,000 per bathroom
```

And evaluation metrics:
```
R² Score: ~0.98-0.99 (Excellent fit)
RMSE: ~$[value] (Average prediction error)
```

## 🔧 Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python** | Programming language |
| **NumPy** | Numerical computations |
| **Pandas** | Data manipulation and analysis |
| **Scikit-learn** | Machine learning library |
| **Matplotlib** | Data visualization |
| **Seaborn** | Enhanced visualizations |

## 📚 Machine Learning Concepts Applied

- **Linear Regression** - Basic supervised learning algorithm
- **Feature Scaling** - Preparation of input features
- **Train-Test Split** - Proper model evaluation methodology
- **Performance Metrics** - MSE, RMSE, R² Score
- **Residual Analysis** - Model diagnostics
- **Data Visualization** - Communicating results

## 🎓 Learning Outcomes

After completing this task, you will understand:
- ✅ How linear regression works
- ✅ Data preprocessing and preparation
- ✅ Model training and evaluation
- ✅ Performance metrics interpretation
- ✅ Making predictions on new data
- ✅ Data visualization best practices

## 📝 License

This project is part of the **Prodigy Infotech** internship program.

## ✉️ Contact & Links

- **Prodigy Infotech Website:** [prodigyinfotech.dev](https://www.prodigyinfotech.dev/)
- **Author GitHub:** [@shyamsasim-del](https://github.com/shyamsasim-del)
- **Repository:** [PRODIGY_INFOTECH-Machine-learning-task-1](https://github.com/shyamsasim-del/PRODIGY_INFOTECH-Machine-learning-task-1)

---

**Status:** ✅ Complete and Ready for Submission

**Created:** June 2026 | **Author:** Shyam M
