# 🌙 Sleep Debt Classification & Behavioral Feature Analysis

This project implements a Machine Learning model using supervised classification techniques to predict a user's **Sleep Debt Category** (`Severe`, `Mild`, `No Sleep Debt`) based on daily habits, screen usage before bed, and physiological sleep parameters.

---

## 📌 Project Overview

Predicting sleep debt accurately enables proactive health interventions before chronic fatigue develops. The core objective of this study was not only to maximize classification accuracy but also to systematically address **Data Leakage** and identify key behavioral indicators influencing sleep debt.

---

## 🛠️ Key Pipeline & Technical Workflow

1. **Data Preprocessing & One-Hot Encoding**:
   - Categorical variables (`gender`, `chronotype`, `primary_bedtime_app`, etc.) were encoded using `pd.get_dummies(drop_first=True)` to prevent the Dummy Variable Trap.
   - Irrelevant identifier columns (e.g., `user_id`) were dropped to prevent arbitrary memorization.

2. **Data Leakage Detection & Resolution**:
   - Initial baseline models yielded an unrealistic **100.00% accuracy**.
   - Feature analysis revealed target leakage caused by post-sleep variables (`total_sleep_hours`, `next_day_fatigue_score`).
   - Removing these leakage features ensured the model focuses strictly on **pre-sleep behaviors and actionable daily metrics**.

3. **Hyperparameter Tuning (`max_depth` Analysis)**:
   - Evaluated `DecisionTreeClassifier` across depths ranging from 2 to 20 to analyze bias-variance tradeoff:
     - `max_depth = 2`: 65.76% (Underfitting)
     - `max_depth = 7`: **74.53% (Optimal Sweet Spot)**
     - `max_depth = 20`: 68.71% (Overfitting due to noise memorization)

---

## 📊 Results & Key Findings

### Model Performance
- **Optimal Algorithm**: Decision Tree Classifier (`max_depth=7`, `random_state=30`)
- **Test Accuracy**: **74.53%**

### Top 5 Most Important Features
| Rank | Feature | Importance Weight | Insights |
| :---: | :--- | :---: | :--- |
| **1** | `morning_alarm_snoozes` | **32.96%** | Primary behavioral indicator for accumulated sleep debt |
| **2** | `bedtime_phone_minutes` | **8.29%** | Significant impact of pre-sleep screen exposure |
| **3** | `sleep_latency_min` | **6.17%** | Delay in sleep onset correlates strongly with debt |
| **4** | `chronotype_Night Owl` | **5.93%** | Circadian rhythm preference impacts sleep timing |
| **5** | `physical_activity_min` | **5.26%** | Daily activity level influences sleep quality |

---

## 🚀 How to Run

1. Clone the repository:
   git clone https://github.com/YOUR_USERNAME/sleep-debt-classification.git
   cd sleep-debt-classification

2. Install requirements:
   pip install pandas numpy scikit-learn matplotlib

3. Run the Notebook / Script:
   jupyter notebook sleep_debt_analysis.ipynb

---

## 💻 Tech Stack
- **Language**: Python
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (`DecisionTreeClassifier`, `train_test_split`, `metrics`)