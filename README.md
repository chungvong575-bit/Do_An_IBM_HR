# 🏢 IBM HR Analytics: Employee Attrition Prediction

**Author:** Chung Vong (Simon)  
**Live Web Application:** [https://doanibmhr-k92jaqyecnhmgfgfothxg9.streamlit.app/](https://doanibmhr-k92jaqyecnhmgfgfothxg9.streamlit.app/)  
**GitHub Repository:** [https://github.com/chungvong575-bit/Do_An_IBM_HR](https://github.com/chungvong575-bit/Do_An_IBM_HR)

---

## 📌 Project Overview
Employee attrition is a major cost for businesses. This project aims to predict the likelihood of an employee leaving the company using the **IBM HR Analytics Employee Attrition Dataset**. By identifying at-risk employees, the HR department can take proactive measures to improve retention.

The final product is an interactive web application built with Streamlit, allowing HR professionals to select an employee profile and instantly receive an AI-driven risk assessment.

## 🛠️ Tools & Technologies
* **Language:** Python
* **Data Processing & EDA:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (Logistic Regression, Support Vector Machine)
* **Handling Imbalanced Data:** Imbalanced-learn (SMOTE)
* **Web Deployment:** Streamlit

## 🧠 Methodology & Machine Learning Approach

### 1. Data Preprocessing & Encoding
* Removed irrelevant columns (`EmployeeCount`, `Over18`, `StandardHours`).
* Applied **Binary Encoding** for columns with two categories (e.g., Gender, OverTime).
* Applied **One-Hot Encoding** (`pd.get_dummies`) for categorical variables with multiple classes to prevent multicollinearity (`drop_first=True`).
* Scaled numerical features using `StandardScaler` to ensure distance-based algorithms perform optimally.

### 2. Handling Class Imbalance (The 84/16 Problem)
Exploratory Data Analysis revealed a significant class imbalance: **83.88%** of employees stayed, while only **16.12%** left. 
To prevent the model from becoming biased toward the majority class, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied **strictly to the training set** to synthesize new minority instances, bringing the ratio to 50/50 without causing data leakage.

### 3. Model Evaluation: Logistic Regression vs. SVM
Two models were trained and compared:
* **Support Vector Machine (SVM):** Achieved a higher overall accuracy (**83%**), but suffered from a very low Recall for the minority class. It missed 30 out of 47 leaving employees, which is highly detrimental in a real-world HR context.
* **Logistic Regression:** Selected as the **Final Model**. Although the overall accuracy was slightly lower (**77%**), it achieved a much higher **Recall (62%)**, correctly identifying 29 out of 47 leaving employees. In attrition prediction, the business cost of missing a resigning employee (False Negative) is much higher than falsely flagging a loyal employee (False Positive).

## 📁 Repository Structure
* `WA_Fn-UseC_-HR-Employee-Attrition.csv`: The raw IBM dataset.
* `EDA_va_Cleaning.ipynb`: The Google Colab notebook containing data exploration, SMOTE application, and model training/evaluation.
* `hr_attrition_model.pkl`: The exported Logistic Regression model.
* `hr_scaler.pkl`: The exported StandardScaler object.
* `app.py`: The Streamlit web application script.
* `requirements.txt`: Dependencies required to run the web app on Streamlit Cloud.
**Project Presentation:** [View Slide Deck (PDF)](./IBM_HR_Attrition_Presentation.pdf)

## 🚀 How to Run Locally
To run this project on your local machine:

1. Clone this repository:
   ```bash
   git clone [https://github.com/chungvong575-bit/Do_An_IBM_HR.git](https://github.com/chungvong575-bit/Do_An_IBM_HR.git)
