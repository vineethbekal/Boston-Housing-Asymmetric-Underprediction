Machine Learning Project Proposal: Risk-Averse Residential Property Valuation

1. Machine Learning Task
Task Type: Supervised Regression
Objective: Predict the final sale price of residential homes (a continuous value).
Core Problem: Standard regression models (like Linear or Ridge Regression) minimize the Mean Squared Error (MSE), which treats over-predictions (estimating a home's value too high) and under-predictions (estimating it too low) equally.
Our Modified Goal (Risk Aversion): We want a model that is risk-averse regarding under-prediction. For a bank or an insurance company, severely under-predicting a property's value is a greater financial risk (e.g., loss of loan business, improper appraisal of properties, missing out on potential profits) than a small over-prediction.
Therefore, the objective is to prioritize reducing the magnitude of under-predictions over reducing the magnitude of over-predictions.

2. Real-Life Data
Dataset: The Boston Housing Price Dataset 
https://www.kaggle.com/datasets/vikrishnan/boston-house-prices
Relevance: This dataset contains dozens of continuous and categorical features (e.g., crime rate, number of rooms, distance to employment centers, property tax rate) that are causally related to a home's price, making it ideal for regression.
Feature Summary:
Input Features (): Crime rate (per capita), average number of rooms, weighted distances to five Boston employment centers, accessibility to major highways, etc.
Target Variable (): Median value of owner-occupied homes (continuous, measured in $1000s).

Distribution Analysis:
Home Prices (target variable): Approximately normal distribution with a slight right skew. The range of values is $5,000 to $50,000 with some high values over $40,000 acting as outliers.
Number of rooms: Approximately normal distribution. The range of values is 3.5 - 9 rooms with the average being 6 rooms.
Crime rate: Highly right skewed distribution. Most areas had a very low crime rate, but some outlier areas had extremely high crime rates.
Lower Status Population (%): Right skewed distribution. Most areas have a percentage under 15%, however a few outliers have a much higher percentage of 20-38%. The range of values here is 1.98% to 37.97%. 
Property Tax: Bimodal distribution that shows two distinct groups around $300 and $700.
Pupil Teacher Ratio: Left skewed distribution. Most neighbors have a relatively high ratio of around 18-20 students per teacher, however some outliers have a much lower ratio with the lowest being 12.6. 
Residential Zoning: Extremely right skewed with 67% of the values being 0. The range here is 0-100. Most towns have no large lot zoning. A few have a lot however, like one town having 100% of residential land zoned.
Non-Retail Business: Bimodal distribution with peaks around 5-7% and 17-19%. The range of values is 0.46-27.74%.
Nitric Oxide Concentration: Approximately normal distribution, with a slide right skew. The range of values is 0.38-0.87 with a mean of 0.55.
Home Age: Left skewed distribution. The range of values is 2.9-100%, with the median being 77.5% of homes being built before 1940.
Employment Distance: Right skewed distribution. The range of values is 1.13 to 12.13, with the mean being 3.80. Most homes are relatively close to employment centres instead of a few outliers.
Highway Access: Extremely right skewed distribution. Most areas ranked highway access between 1-8. Only a few had values a lot higher, like 24. The units for this are ordinal rankings with a higher number meaning better access.
Black population: Extremely left skewed distribution. The range of values is 0.32 to 396.9 with most values clustered around 390-396. This value measures how far the population is from 63% black. For example, a value of 0 means the population is exactly 63% black while a value of 396.9 means there are practically none in the population. 
   


3. Baseline Models
We will use two predictive models as our baselines: Ridge Regression and Random Forest.

Ridge Regression: This is a common choice because it minimizes the standard Mean Squared Error (MSE) while using  regularization to prevent overfitting, aligning with industry best practices for a foundational model. We can also use the coefficients calculated by this model to see which features have more/less predictive power.
Random Forest: A common choice for predictive tasks with high-dimensionality of features, which works by building/training multiple decision trees off of various subsets of the data. During testing, new samples are evaluated by all trees independently, and a majority-consensus is used to make the final prediction for that sample. The use of multiple independently-built trees reduces the risk of overfitting, which is a common issue for individual decision trees.
Baseline Objective Function (Ridge Regression)
This is a linear baseline model that minimizes the sum of the squared errors plus the squared magnitude of the coefficients for each variable.

Baseline Objective Function (Random Forest)
This baseline model’s objective function is the information gain of each decision, which is calculated as the entropy of the training data before a split, minus the summation of the weighted entropies of each subset produced by the split. This is done for every split in every tree in the forest.


4. Modified Model
To align with the objective of risk aversion against under-prediction, we will modify the standard MSE term by introducing an asymmetric cost penalty.
Modified Objective Function (Risk-Averse Regression)
We will use a parameter γ>1 (Gamma) to weigh the penalty of under-prediction more heavily than over-prediction.

Where Costi​ is the asymmetric cost for the i-th sample:

Effect of (Gamma):
If the model under-predicts (the true price is higher than the predicted price), the resulting squared error is multiplied by a penalty.
If the model over-predicts, the cost is simply the standard squared error.
This forces the optimization algorithm (Gradient Descent) to nudge the predicted value higher to avoid the severe penalty, thus making the model inherently more cautious about predicting too low.

5. Implementation Strategy and Comparison
Implementation
The modified model will be implemented using Gradient Descent (SGD) because the objective function is no longer smooth and closed-form. The gradient calculation must be derived from the asymmetric loss function.
Comparison Metrics
Since the goal is not just average accuracy, we need both a primary metric and secondary risk metrics:
Primary Metric (Risk Reduction): Compare the Root Mean Squared Under-Prediction Error (RMSUE) between the models.

Goal: The Modified Model must have a significantly lower RMSUE than the Baseline Model.
Secondary Metric (Efficiency Trade-off): Compare the overall  (Root Mean Squared Error) on the test set.
Goal: The Modified Model's  is expected to be slightly higher than the Baseline's (due to the asymmetric focus), illustrating the cost of risk aversion.
Simple financial models can be used to quantify the reduction in financial losses our model provides compared to a standard model for banks & insurance companies.
Qualitative Comparison: Compare the distribution of residuals (errors) . The modified model's residual distribution should be visibly shifted to the positive side (more over-predictions, fewer under-predictions) compared to the standard model.
6. Team Members
Benjamin Forelli: Responsible for analyzing the distribution of features in the dataset to identify skew, outliers, and scaling needs. Also will design and implement evaluation metrics on data such as Root Mean Squared Under Prediction Error and Mean Absolute Error.
Andrew Bugbee: Training & evaluating baseline models, primarily ridge regression, and evaluating using RMSUE. Will also help with EDA
Shourya Kasliwal: Implementing the custom asymmetric loss function via NumPy Gradient Descent, and statistically validating the risk-averse model's performance against the baseline.
Vineeth Bekal: Responsible for displaying and analyzing model outputs and improving model performance using log-target training, bias calibration, and simple preprocessing to provide cleaner, higher-signal inputs.
Eric Faith: Responsible for building, training and evaluating baseline models, primarily the random forest via analysis of information gain.

