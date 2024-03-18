# Option Pricing and Modeling: Machine Learning vs. Black-Scholes Formula

## Project Overview

This project investigates the application of machine learning (ML) algorithms in option pricing, comparing their performance against the traditional Black-Scholes formula. By focusing on options, which are critical financial derivatives, this study explores innovative methodologies beyond conventional models, aiming to enhance accuracy and adaptability in option pricing.

## Team Members

- Alan Lau
- Alton Yu
- Daniel Alpizar
- Kevin Chan

## Data Sources

- **Underlying Stock Data**: Extracted from Yahoo Finance.
- **Options Data**: Sourced from Polygon.io API.
- **Risk-free Rates**: Obtained from the US Department of Treasury.

## Fundamentals of Options

Options are contracts offering the buyer the right but not the obligation to buy (call option) or sell (put option) an underlying asset at a predetermined price before or at a certain date. This project delves into the mechanics, valuation, and trading strategies associated with options.

## Objectives

- To compare the traditional Black-Scholes option pricing model with machine learning-based models in terms of accuracy and efficiency.
- To demonstrate the capabilities of ML in capturing market dynamics and complexities not addressed by the Black-Scholes model.

## Methodology

1. **Dataset Preparation**: Utilized historical data of AAPL stock prices, option prices, and expiration dates.
2. **Model Training**: Separated the dataset into calls and puts, then divided each group into an 80% training set and a 20% testing set. Employed models including Lasso Regression and Artificial Neural Networks (ANN) to predict option prices.
3. **Evaluation**: Compared the absolute errors between predicted and actual option prices in the test set to assess model performance.

## Results

- The single-layer Neural Network demonstrated superior performance in predicting option prices.
- All models exhibited similar accuracy for both calls and puts, with Lasso Regression showing a slightly larger discrepancy in predictions.
- Further model refinement and validation techniques, such as cross-validation, could potentially improve results and prevent overfitting.

## Future Directions

Future research could expand the analysis to include a wider range of stocks and underlying assets, such as commodities and energy derivatives. Additionally, exploring more complex ML models and incorporating a broader set of features may offer deeper insights and enhanced predictive capabilities.

## Tools and Technologies Used

- **Data Analysis and Modeling**: Jupyter Notebook
- **Data Visualization**: Plotly, Matplotlib
- **Reporting**: Plotly Dash

## Contact

For any inquiries or contributions, please contact [@daniel-alpizar](https://github.com/daniel-alpizar) or open an issue on this repository.
