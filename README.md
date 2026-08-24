# 🐋 Mobile Game Whale Prediction

### Predicting Which Players Are Most Likely to Become Paying Customers

[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.0-F7931E?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas\&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Live-success)](https://whale-prediction-sagar-rokade.streamlit.app/)

> 🎮 **A Machine Learning classification system that predicts whether a mobile game player is likely to convert from a free user into a paying customer based on their early-game behaviour.**

---

## 🚀 Live Demo

### 👉 [🐋 Try the Whale Predictor](https://whale-prediction-sagar-rokade.streamlit.app/)

Enter a player's demographic information, engagement behaviour, progression, advertising activity, social behaviour and store interactions to predict their probability of becoming a paying customer.

---

## 💻 GitHub Repository

### 👉 [View the Source Code](https://github.com/sagarrokade2904/Whale-Prediction)

---

## 🎯 Project Overview

In free-to-play mobile games, the majority of players never make a purchase, while a relatively small percentage become paying users.

These high-value players are commonly referred to as **"whales"** in the gaming industry.

Identifying players who are likely to convert can help game studios make better decisions around:

* 🎁 Personalized offers
* 💰 In-game monetization
* 📣 Targeted campaigns
* 🔔 Push notifications
* 🛍️ Store recommendations
* 🎮 Player engagement
* 📈 Customer lifetime value optimization

This project uses **early player behaviour** to predict whether a player will eventually become a paying customer.

The problem is formulated as a **binary classification task**:

```text
converted_to_payer

1 → Paying customer
0 → Non-paying player
```

The project covers the complete Machine Learning workflow:

```text
Data
 ↓
EDA
 ↓
Data Cleaning
 ↓
Preprocessing
 ↓
Model Comparison
 ↓
Model Selection
 ↓
Cross-Validation
 ↓
Final Model
 ↓
Model Serialization
 ↓
Streamlit Deployment
```

---

## ✨ Key Features

* 🐋 Predict player-to-payer conversion
* 📊 Dataset containing **12,000 players**
* 🎮 Uses real-world style mobile gaming behaviour features
* 🧹 Handles missing values automatically
* 🔢 Numerical feature standardization
* 🔤 Categorical feature encoding
* 🤖 Multiple classification algorithms explored
* 📈 Accuracy, Precision, Recall, F1 and ROC-AUC evaluation
* 🔄 Stratified 5-Fold Cross-Validation
* 💾 Serialized Scikit-learn pipeline
* ⚡ Real-time prediction using Streamlit
* 📊 Displays payer probability
* ☁️ Deployed online

The notebook contains the complete modelling workflow, including comparison of Logistic Regression, KNN, SVM, Decision Tree, Bagging, Random Forest, AdaBoost, Gradient Boosting and Voting Classifier approaches.

---

# 🧠 Machine Learning Problem

## Problem Type

**Binary Classification**

The model predicts one of two outcomes:

| Prediction | Meaning                                        |
| ---------- | ---------------------------------------------- |
| `0`        | Player is unlikely to become a paying customer |
| `1`        | Player is likely to become a paying customer   |

### Target Variable

```text
converted_to_payer
```

The target is defined as:

```text
1 = Payer
0 = Non-Payer
```

The notebook explicitly frames the business problem as predicting payer conversion from early player behaviour.

---

# 📊 Dataset

The dataset contains:

### **12,000 players × 26 variables**

The notebook reports a dataset shape of `(12000, 26)`.

The data contains information about:

### 👤 Player Demographics

* Age
* Gender
* Country

### 📱 Acquisition & Device

* Acquisition channel
* Device type

### 🎮 Engagement

* Days since installation
* Sessions in the last 7 days
* Average session length
* Total playtime

### 🏆 Game Progression

* Levels completed
* Current level
* Tutorial completion
* Days active in the last 30 days
* Streak days
* Level failure rate
* Rage quit events

### 👥 Social Behaviour

* Friends connected
* Social shares

### 📢 Advertising Behaviour

* Ad views
* Rewarded ad views
* Push notification status

### 🛒 In-Game Store Behaviour

* Store visits
* Items viewed in store
* Wishlist items

### 🎯 Target

* Converted to payer

The dataset includes missing values in several numerical variables, which are handled as part of the preprocessing pipeline.

---

# 🔍 Exploratory Data Analysis

The notebook performs exploratory analysis before modelling.

Key areas investigated include:

* Dataset structure
* Missing values
* Summary statistics
* Numerical distributions
* Categorical distributions
* Target distribution
* Player engagement behaviour
* Relationships between player activity and payer conversion
* Model performance comparison

A particularly important consideration is **class imbalance**.

Only around **13% of players convert**, making accuracy alone potentially misleading. A model could achieve high accuracy simply by predicting "non-payer" for most players.

Therefore, the project emphasizes:

```text
Precision
Recall
F1 Score
ROC-AUC
```

rather than relying solely on accuracy.

---

# ⚙️ Data Preprocessing

The final model uses a Scikit-learn `Pipeline` with a `ColumnTransformer`.

## 🔢 Numerical Features

Numerical variables go through:

```text
SimpleImputer(strategy="median")
        ↓
StandardScaler()
```

This means:

1. Missing numerical values are replaced using the median.
2. Numerical variables are standardized before modelling.

## 🔤 Categorical Features

Categorical variables go through:

```text
SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")
```

This means:

1. Missing categorical values are replaced using the most frequent category.
2. Categorical variables are converted into numerical representations.
3. Previously unseen categories are handled safely during prediction.

This preprocessing architecture is embedded directly into the final model pipeline.

---

# 🤖 Machine Learning Models

Several classification algorithms were explored during the project:

| Model                  | Purpose                              |
| ---------------------- | ------------------------------------ |
| Logistic Regression    | Baseline + interpretable classifier  |
| K-Nearest Neighbors    | Distance-based classification        |
| Support Vector Machine | Margin-based classification          |
| Decision Tree          | Non-linear rule-based classification |
| Bagging                | Ensemble learning                    |
| Random Forest          | Tree-based ensemble                  |
| AdaBoost               | Boosting ensemble                    |
| Gradient Boosting      | Sequential boosting                  |
| Voting Classifier      | Ensemble combination                 |

The notebook compares these models using classification metrics and ROC curves.

---

# 🏆 Final Model

The final deployed model is:

## **Logistic Regression**

The final pipeline is structured as:

```text
                 ┌──────────────────────┐
                 │   Player Features    │
                 └──────────┬───────────┘
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
      Numerical Features           Categorical Features
             │                             │
             ▼                             ▼
      Median Imputation           Most-Frequent Imputation
             │                             │
             ▼                             ▼
       StandardScaler              One-Hot Encoding
             │                             │
             └──────────────┬──────────────┘
                            │
                            ▼
                  Logistic Regression
                            │
                            ▼
                  ┌─────────────────┐
                  │ Payer Probability│
                  └─────────────────┘
```

The final pipeline uses `LogisticRegression(max_iter=1000, random_state=42)`.

---

# 📈 Model Validation

To obtain a more reliable estimate of model performance, the final Logistic Regression model was evaluated using:

### **Stratified 5-Fold Cross-Validation**

The resulting cross-validated ROC-AUC was:

```text
ROC-AUC: 0.7302 ± 0.0204
```

This indicates that the model has meaningful ability to distinguish likely payers from non-payers, while also highlighting that payer prediction is a challenging behavioural classification problem.

---

# 🎮 Streamlit Application

The deployed application provides an interactive interface for entering player information.

## Player Information

* Age
* Gender
* Country
* Device type
* Acquisition channel

## Engagement

* Days since installation
* Sessions in last 7 days
* Average session length
* Total playtime

## Progression

* Levels completed
* Current level
* Tutorial completion
* Days active
* Streak days
* Level failure rate

## Social Behaviour

* Friends connected
* Social shares

## Monetization Signals

* Store visits
* Items viewed
* Wishlist items
* Ad views
* Rewarded ad views
* Push notifications

The application then returns:

```text
Prediction
+
Payer Probability
+
Non-Payer Probability
```

The Streamlit app loads the serialized model from `models/whale_prediction_model.pkl` and uses `predict()` and `predict_proba()` for inference.

---

# 📊 Prediction Example

A hypothetical player might have:

```text
Age                         → 25
Days Since Install          → 10
Sessions / Last 7 Days      → 7
Average Session Length      → 10 min
Total Playtime              → 5 hours
Levels Completed            → 15
Current Level               → 15
Tutorial Completed          → Yes
Friends Connected           → 2
Push Notifications          → Yes
Ad Views                    → 6
Rewarded Ad Views           → 2
Store Visits                → 2
Wishlist Items              → 1
Days Active / Last 30 Days  → 15
Streak Days                 → 4
Rage Quit Events            → 3
Level Fail Rate             → 0.60
Social Shares               → 1
```

The model processes these characteristics and produces a probability of payer conversion.

Example:

```text
🐋 Payer Probability: 72.4%

Prediction:
Likely to convert to a payer
```

*The values above are illustrative and are not a guaranteed model output.*

---

# 🗂️ Project Structure

```text
Whale-Prediction/
│
├── 📁 models/
│   └── whale_prediction_model.pkl
│
├── 🐍 app.py
│   └── Streamlit prediction application
│
├── 📊 mobile_game_whale_dataset.csv
│   └── Player behaviour dataset
│
├── 📓 mobile_game_whale_prediction.ipynb
│   └── EDA, preprocessing, modelling,
│       evaluation and model training
│
├── 📦 requirements.txt
│   └── Python dependencies
│
└── 📄 README.md
    └── Project documentation
```

The repository currently contains the Streamlit application, dataset, modelling notebook, requirements file and serialized model directory.

---

# 🛠️ Tech Stack

| Technology          | Purpose                   |
| ------------------- | ------------------------- |
| 🐍 Python           | Programming language      |
| 🐼 Pandas           | Data manipulation         |
| 🔢 NumPy            | Numerical computing       |
| 📊 Matplotlib       | Visualization             |
| 🎨 Seaborn          | Exploratory Data Analysis |
| 🤖 Scikit-learn     | Machine Learning          |
| 💾 Joblib           | Model serialization       |
| 🌐 Streamlit        | Web application           |
| 📓 Jupyter Notebook | Model development         |

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/sagarrokade2904/Whale-Prediction.git
```

## 2. Navigate to the project

```bash
cd Whale-Prediction
```

## 3. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will then be available locally in your browser.

---

# 🔬 Machine Learning Workflow

```text
                     ┌───────────────────┐
                     │   Player Dataset  │
                     │    12,000 Users    │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │       EDA         │
                     │ Missing Values    │
                     │ Distributions     │
                     │ Class Balance     │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │  Preprocessing    │
                     │ Imputation        │
                     │ Scaling           │
                     │ One-Hot Encoding  │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │ Model Comparison  │
                     │ LR / KNN / SVM    │
                     │ RF / GB / AdaBoost│
                     │ etc.              │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │ Final Logistic    │
                     │ Regression Model  │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │ 5-Fold Stratified │
                     │ Cross-Validation  │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │ Serialize Model   │
                     │     Joblib        │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │ Streamlit Cloud   │
                     │     Deployment    │
                     └───────────────────┘
```

---

# 💡 Business Applications

A model like this could be integrated into a mobile game's analytics or CRM system to identify players who may be more likely to monetize.

Potential applications include:

### 🎁 Personalized Offers

Target likely converters with relevant in-game offers.

### 📣 Marketing Optimization

Focus promotional campaigns on players with stronger conversion potential.

### 🔔 Push Notification Strategy

Use behavioural signals to determine which players may be receptive to targeted notifications.

### 🛒 Store Personalization

Recommend relevant items based on player engagement and store behaviour.

### 💰 Revenue Optimization

Improve monetization strategies by identifying high-potential players earlier in their lifecycle.

---

# ⚠️ Important Consideration: Class Imbalance

Only approximately **13% of players convert to paying users** in the dataset.

This creates an important modelling challenge.

For example, a model that predicts:

```text
Every player → Non-Payer
```

could still achieve relatively high accuracy.

However, such a model would be practically useless for identifying potential paying customers.

Therefore, this project emphasizes:

```text
              Accuracy
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
   Precision    Recall    F1
                 │
                 ▼
              ROC-AUC
```

The notebook explicitly notes that accuracy alone can be misleading for this imbalanced classification problem.

---

# 🚧 Future Improvements

Potential improvements include:

* [ ] Hyperparameter tuning using `GridSearchCV`
* [ ] Compare tuned ensemble models
* [ ] Optimize classification threshold
* [ ] Add Precision-Recall AUC
* [ ] Add SHAP-based explainability
* [ ] Display feature contribution for individual predictions
* [ ] Add player segmentation
* [ ] Predict expected customer lifetime value
* [ ] Predict purchase amount
* [ ] Build a real-time analytics dashboard
* [ ] Integrate with a game analytics API
* [ ] Add model monitoring
* [ ] Automate model retraining
* [ ] Add experiment tracking
* [ ] Deploy the model as a REST API

---

# 📚 What This Project Demonstrates

This project demonstrates practical understanding of:

### Data Science

* Exploratory Data Analysis
* Missing-value treatment
* Feature analysis
* Class imbalance

### Machine Learning

* Binary classification
* Logistic Regression
* Ensemble learning
* Model comparison
* ROC-AUC
* Precision / Recall / F1
* Cross-validation

### ML Engineering

* Scikit-learn Pipelines
* ColumnTransformer
* Model serialization
* Reusable preprocessing
* Prediction pipelines

### Deployment

* Streamlit application development
* Interactive ML inference
* Cloud deployment

---

# 🌐 Links

| Resource     | Link                                                                               |
| ------------ | ---------------------------------------------------------------------------------- |
| 🚀 Live Demo | [Whale Prediction App](https://whale-prediction-sagar-rokade.streamlit.app/)       |
| 💻 GitHub    | [Whale-Prediction Repository](https://github.com/sagarrokade2904/Whale-Prediction) |

---

# 👨‍💻 Author

## Sagar Rokade

**Machine Learning • Data Science • Python**

🔗 [GitHub Profile](https://github.com/sagarrokade2904)

🔗 [Whale Prediction Project](https://github.com/sagarrokade2904/Whale-Prediction)

🚀 [Live Application](https://whale-prediction-sagar-rokade.streamlit.app/)

---

# ⭐ Show Your Support

If you found this project interesting:

⭐ **Star the repository**

🍴 **Fork the project**

🐛 **Open an issue**

💡 **Suggest an improvement**

---

### 🐋 Predict the Player. Understand the Behaviour. Improve Monetization.

**Built with Python, Scikit-learn & Streamlit.**
