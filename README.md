# Bishkek House Price Prediction

## Overview

This project explores and builds multiple regression models to predict house prices based on various features such as district, micro-district, building type, condition, and other numerical variables.

## Data

* `house_price.csv`: Raw dataset containing house sale records.

## Environment Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```
2. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

> If you don’t have a `requirements.txt`, install directly:
>
> ```bash
> pip install pandas numpy matplotlib seaborn scikit-learn lightgbm catboost xgboost
> ```

## Notebook Usage

Open and run the Jupyter notebook to reproduce the analysis and results:

```bash
jupyter notebook MidtermProject.ipynb
```

### Notebook Sections

1. **Data Exploration**
   Initial inspection, descriptive statistics, and visualization of distributions.
2. **Data Cleaning**
   Handling missing values, encoding categorical variables, saving to `clean.csv`.
3. **Feature Engineering**
   Mapping district and building type codes, converting features to categorical.
4. **Model Training & Evaluation**

   * Splitting data into train/test sets.
   * Training multiple regression models:

     * Linear Regression
     * K-Nearest Neighbors
     * Decision Tree
     * Random Forest
     * Gradient Boosting
     * AdaBoost
     * XGBoost
     * CatBoost
     * LightGBM
     * Naive Bayes
   * Calculating MSE and R², and performing 5-fold cross-validation.
5. **Predictions**
   Interactive section for custom input and price prediction using trained models.

## Preprocessing Steps

* **Condition Imputation:** Rows with missing `condition` values were identified; a RandomForestClassifier was trained on non-null `condition` entries to predict and fill these missing values.
* **Micro-District Imputation:** Similarly, missing `micro_district` entries were filled by training a RandomForestClassifier on known micro-districts, with a `SimpleImputer(strategy='most_frequent')` applied to other features during training.
* **Categorical Encoding:** All categorical features (`district`, `micro_district`, `building_type`, `condition`) were converted to integer codes using `pd.Categorical(...).codes`.
* **Dataset Splitting:** The dataset was split into training and testing sets by separating rows where `condition` was originally non-null vs. null, ensuring proper imputation flow.
* **Raw Features Used:** No explicit scaling or outlier removal was applied; regression models operated on the encoded and imputed data directly.

## Results

| Model               | MSE (Train)    | R² (Train) | CV MSE         | CV R² |
| ------------------- | -------------- | ---------- | -------------- | ----- |
| Linear Regression   | 275,047,069.04 | 0.85       | 165,470,684.72 | 0.92  |
| K-Nearest Neighbors | 60,394,873.17  | 0.97       | 84,401,679.90  | 0.96  |
| Decision Tree       | 10,324,337.49  | 0.99       | 42,043,465.02  | 0.98  |
| Random Forest       | 7,475,739.00   | 1.00       | 41,042,108.15  | 0.98  |
| Gradient Boosting   | 74,916,935.81  | 0.96       | 47,204,526.78  | 0.98  |
| AdaBoost            | 395,932,061.01 | 0.79       | 367,616,465.12 | 0.81  |
| XGBoost             | 12,592,396.79  | 0.99       | 41,418,335.26  | 0.98  |
| CatBoost            | 2,939,789.25   | 1.00       | 34,620,950.39  | 0.98  |
| LightGBM            | 7,486,736.07   | 1.00       | 53,767,719.24  | 0.98  |
| Naive Bayes         | 755,716,298.16 | 0.60       | 647,005,725.57 | 0.69  |

## Insights & Key Findings

* **Best performer:** CatBoost achieved the lowest training MSE (2,939,789.25) and highest cross-validated R² (0.98), indicating robust performance on unseen data.
* **Boosting algorithms:** XGBoost and Gradient Boosting delivered high CV R² scores (\~0.98) but with slightly larger training MSEs compared to CatBoost.
* **Baseline comparison:** Linear Regression and Naive Bayes underperformed, with CV R² scores of 0.92 and 0.69 respectively, highlighting the dataset's nonlinear complexities.
* **Feature importance:** Consistent across tree-based models—`area`, `building_condition`, and `district` emerged as the most influential predictors.

## Usage

To predict custom scenarios, modify the `user_input` dictionary at the end of the notebook and rerun the prediction cells.

## Requirements

* Python 3.7+
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* lightgbm
* catboost
* xgboost

Install via:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm catboost xgboost
```

## Author

Aizhamal Zhetigenova
📧 [aizhamal.zhetigenova@gmail.com](mailto:aizhamal.zhetigenova@gmail.com)
