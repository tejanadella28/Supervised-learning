# House Price Prediction 🏠📈 on California Dataset using Linear Regression

## Overview
This project demonstrates how to predict house prices using the California Housing Dataset and Linear Regression. The primary goal is to understand how features such as median income, population, and proximity to the ocean influence house prices in California. The project involves exploratory data analysis (EDA), feature engineering, and the implementation of a linear regression model.

## Dataset
The California Housing Dataset is a popular dataset provided by Scikit-learn. It contains information about housing in various districts of California, including:
- Median house value (target variable)
- Median income
- Population
- Average number of rooms
- Proximity to the ocean

For more details on the dataset, visit the [Scikit-learn documentation](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).

## Project Workflow
1. **Data Preprocessing**
   - Load the dataset using Scikit-learn's `fetch_california_housing()` function.
   - Handle missing values, if any.
   - Normalize/scale numerical features for better performance.

2. **Exploratory Data Analysis (EDA)**
   - Understand the data distribution and relationships between features.
   - Visualize key correlations using libraries like Matplotlib and Seaborn.

3. **Feature Engineering**
   - Select important features using correlation analysis.
   - Generate new features if necessary.

4. **Model Implementation**
   - Split the data into training and testing sets.
   - Train a linear regression model using Scikit-learn's `LinearRegression` class.
   - Evaluate model performance using metrics like Mean Squared Error (MSE) and R² score.

5. **Model Evaluation**
   - Visualize predictions versus actual values.
   - Identify any patterns in residuals.

## Prerequisites
To run the project, ensure you have the following dependencies installed:
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

You can install the required libraries using:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/tejanadella28/Supervised-learning/blob/main/House_price_prediction.ipynb.git
   ```
2. Navigate to the project directory:
   ```bash
   cd house-price-prediction
   ```
3. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook house_price_prediction.ipynb
   ```
   or
   ```bash
   python house_price_prediction.py
   ```

## Results
The trained linear regression model provides predictions for median house prices in California districts. Evaluation metrics and visualization plots are included in the notebook to assess the model's performance.

## Future Enhancements
- Experiment with more advanced regression models like Ridge, Lasso, or Gradient Boosting.
- Integrate additional external data to improve predictions.
- Deploy the model as a web application using frameworks like Flask or Django.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Author
[Nadella Teja](https://github.com/TejaNadella28)

Feel free to reach out for questions or suggestions!


