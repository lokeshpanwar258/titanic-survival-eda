# 🚢 Titanic Survival Analysis – Exploratory Data Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-4C72B0)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Titanic-orange)

> An in-depth Exploratory Data Analysis (EDA) on the famous Titanic dataset to uncover patterns and factors that influenced passenger survival.

---

## 📌 Project Overview

The sinking of the RMS Titanic in April 1912 is one of the most infamous maritime disasters in history. This project performs a thorough EDA on the Titanic passenger dataset to answer a key question:

**"What kinds of people were more likely to survive?"**

Using statistical analysis and visualizations, we explore how features like gender, passenger class, age, fare, and port of embarkation affected survival outcomes.

---

## 📂 Dataset

- **Source:** [Kaggle – Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- **Size:** 891 rows × 12 columns
- **Target Variable:** `Survived` (0 = No, 1 = Yes)

| Feature | Description |
|---|---|
| `Survived` | Survival (0 = No, 1 = Yes) |
| `Pclass` | Passenger class (1st, 2nd, 3rd) |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | # of siblings/spouses aboard |
| `Parch` | # of parents/children aboard |
| `Fare` | Passenger fare (£) |
| `Embarked` | Port of embarkation (C, Q, S) |

---

## 🔧 Tools & Technologies

| Tool | Purpose |
|---|---|
| Python 3.10 | Core programming language |
| Pandas | Data loading, cleaning, manipulation |
| NumPy | Numerical operations |
| Matplotlib | Base visualizations |
| Seaborn | Statistical plots and heatmaps |
| Jupyter Notebook | Interactive analysis environment |

---

## 📊 Analysis Performed

### 1. Data Cleaning & Preprocessing
- Identified and handled **177 missing Age values** (19.9%) using median imputation per class/gender group
- Dropped `Cabin` column due to 77% missing values
- Filled 2 missing `Embarked` values with mode

### 2. Univariate Analysis
- Distribution of Age, Fare, SibSp, Parch
- Count plots for categorical features (Sex, Pclass, Embarked)
- Survival rate overview: 38.4% survived

### 3. Bivariate Analysis
- Survival rate by Gender, Pclass, Age group, Fare range, Embarked port
- Cross-tabulation tables

### 4. Multivariate Analysis
- Class × Gender survival heatmap
- Correlation matrix of all numeric features
- Pairplot of key features colored by survival

---

## 📈 Key Findings

| # | Finding |
|---|---|
| 1 | **Gender** was the strongest predictor — females had **74%** survival rate vs **18.9%** for males |
| 2 | **1st class passengers** were **2.6×** more likely to survive than 3rd class passengers |
| 3 | **Children (0–10 years)** had disproportionately higher survival rates across all classes |
| 4 | **Higher fare** strongly correlated with survival (r = +0.257), reflecting class privilege |
| 5 | **Cherbourg** passengers had highest survival (55.4%) due to more 1st-class boardings |
| 6 | Being a **1st-class female** gave 96.8% survival chance vs 13.5% for a 3rd-class male |

---

## 📉 Correlation with Survival

```
Sex (Female)   →  +0.543  ████████████  Very High
Pclass         →  −0.338  ████████      High
Fare           →  +0.257  ██████        Medium
Age            →  −0.077  ██            Low
SibSp          →  −0.035  █             Very Low
```

---

## 🗂️ Project Structure

```
titanic-survival-eda/
│
├── titanic_eda.ipynb       # Main Jupyter Notebook with full analysis
├── titanic_eda.html        # Interactive HTML visualization dashboard
├── data/
│   └── train.csv           # Raw dataset from Kaggle
├── images/
│   ├── survival_by_gender.png
│   ├── survival_by_class.png
│   ├── age_distribution.png
│   └── correlation_heatmap.png
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone this repository
git clone https://github.com/yourname/titanic-survival-eda.git

# 2. Navigate to project folder
cd titanic-survival-eda

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn jupyter

# 4. Launch Jupyter Notebook
jupyter notebook titanic_eda.ipynb
```

---

## 🔮 Next Steps (Future Work)

- [ ] Build a classification model (Logistic Regression, Random Forest)
- [ ] Feature engineering: family size, title extraction from Name
- [ ] Hyperparameter tuning and cross-validation
- [ ] Deploy prediction model as a web app using Streamlit

---

## 🙏 Acknowledgements

- Dataset provided by [Kaggle](https://www.kaggle.com/c/titanic)
- Inspired by the data science community's extensive work on this classic dataset

---

*Made with ❤️ as part of a Data Science learning project*
