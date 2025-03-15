# medical_insurance_cost_prediction
![Understanding insurance cost](hospital_file and 4 more pages - Profile 1 - Microsoft​ Edge 3_14_2025 4_42_56 PM (2).png)

## link to the model  app here (https://huggingface.co/spaces/pritex/hospital_cost_predictions_app)

# Overview
The analysis aimed to understand the key factors that influence hospital billing and medical insurance costs. Various statistical techniques and machine learning models were used to identify the relationships between factors like age, gender, BMI, number of children, smoking habits, and region with medical charges. Additionally, a predictive model was built using a Random Forest Regressor to estimate healthcare costs.

# Data Source
Data is sourced from kaggle


# Exploratory Data Analysis (EDA) Summary
The exploratory data analysis (EDA) involved various statistical techniques and visualizations to understand the factors influencing hospital costs.

## 1. Data Inspection and Cleaning
- Missing Values: There were no missing values in the dataset.
- Duplicate Rows: The dataset contained duplicate rows, which were removed to ensure data integrity.
- Dataset Structure:
The dataset contained 7 columns: age, sex, bmi, children, smoker, region, and charges.
The charges column represents the target variable (dependent variable), while the others are independent variables.
## 2. Factors Influencing Hospital Costs
A hypothesis testing approach was used to analyze the relationship between charges and various independent variables.

### 2.1. Does Age Influence Hospital Cost?
Hypothesis Test:

- Null Hypothesis (H₀): Age does not influence hospital charges.
- Alternative Hypothesis (H₁): Age influences hospital charges.
- Statistical Test: Pearson Correlation (pearsonr)

- Result: The correlation coefficient and p-value indicated a significant positive correlation between age and hospital costs.

- Key Insight: As a person ages, their medical expenses tend to increase.

- Visualization:

A barplot of age vs. charges showed that older individuals incur higher medical costs.
### 2.2. Does Gender Affect Medical Costs?
Hypothesis Test:

- Null Hypothesis (H₀): Gender affects hospital charges.
- Alternative Hypothesis (H₁): Gender does not affect hospital charges.
- Statistical Test: Independent T-test (ttest_ind)

- Result: The p-value indicated that gender does not significantly affect hospital charges.

- Key Insight: The cost of medical treatment is not influenced by gender.

- Visualization:

A histogram of charges grouped by sex showed no significant difference in cost distribution between males and females.
### 2.3. Is BMI Correlated with Medical Expenses?
Hypothesis Test:

- Null Hypothesis (H₀): BMI does not affect hospital charges.
- Alternative Hypothesis (H₁): BMI influences hospital charges.
- Statistical Test: Spearman Correlation (spearmanr)

- Result: A positive correlation was found, indicating that higher BMI values are associated with higher medical expenses.

- Key Insight: Individuals with higher BMI (overweight/obese) tend to incur higher medical costs.

- Visualization:

A line plot of bmi vs. charges showed that hospital costs increase with higher BMI values.
### 2.4. Does the Number of Children Influence Medical Costs?
Hypothesis Test:

- Null Hypothesis (H₀): Number of children does not affect hospital charges.
- Alternative Hypothesis (H₁): Number of children influences hospital charges.
- Statistical Test: Independent T-test (ttest_ind)

- Result: The p-value showed no significant relationship between the number of children and medical costs.

- Key Insight: Having more children does not increase hospital costs.

- Visualization:

A bar plot of children vs. charges confirmed that the number of dependents does not significantly impact medical costs.
### 2.5. Does Smoking Affect Medical Costs?
Hypothesis Test:

- Null Hypothesis (H₀): Smoking does not affect hospital charges.
- Alternative Hypothesis (H₁): Smoking increases hospital charges.
- Statistical Test: Independent T-test (ttest_ind)

- Result: A statistically significant difference was observed, with smokers incurring higher medical costs.

- Key Insight: Smoking is a major factor driving up hospital expenses, likely due to smoking-related health risks.

- Visualization:

A histogram of charges grouped by smoker status showed that smokers had significantly higher medical expenses.
### 2.6. Which Region Has the Highest Medical Costs?
- Analysis: Medical expenses were grouped by region to identify regional variations.
- Key Insight: The Southeast region had the highest hospital costs, while other regions showed moderate variations.
- Visualization:
A bar plot of region vs. charges confirmed that the Southeast region has the highest medical expenses.
### 3. Correlation Analysis
A heatmap of correlation coefficients revealed:
- Strongest correlation: smoker (highest positive impact on hospital costs).
- Moderate correlation: age, bmi, and region.
- Weak correlation: children and sex.

## Data Preprocessing
Before feeding the data into machine learning models, I prepared it by performing the following steps:

- 1.1 Handling Categorical Variables
The dataset had categorical columns:
sex, smoker, and region.
These were converted into numerical values using one-hot encoding:
sex: Converted to binary (0 = female, 1 = male).
smoker: Converted to binary (0 = non-smoker, 1 = smoker).
region: One-hot encoded into four separate columns (region_northeast, region_northwest, region_southeast, region_southwest).
- 1.2 Feature Scaling
Since the charges column is continuous and has a wide range of values, I scaled numerical features (age, bmi, children) using StandardScaler to improve model performance.
The charges column remained unchanged since it was the target variable.
## Model Building
- RandomforestRegressor was used with 90% accuracy.

## Key Insights
- Age and BMI are positively correlated with hospital costs – older and obese individuals pay more for medical expenses.
- Smoking is the most significant factor affecting hospital bills – smokers pay substantially higher costs.
- Gender and the number of children have no significant impact on medical costs.
- The Southeast region incurs the highest medical expenses compared to other regions.
## Recommendations
- Encourage healthy lifestyle choices – promoting weight management  programs could reduce long-term medical costs.
- Insurance premium adjustments – insurers can consider age, BMI, and smoking status when designing insurance plans.
- Targeted healthcare policies – regional variations suggest that healthcare costs in the Southeast region need further investigation to understand cost drivers.
- Public health campaigns – increasing awareness about the impact of obesity and smoking on medical expenses could help reduce overall healthcare costs.
