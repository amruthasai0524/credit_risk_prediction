# 📊 Credit Risk Prediction System

An end-to-end Machine Learning project that predicts whether a customer is a **high-risk or low-risk borrower** using financial and demographic data.

Built with a complete ML pipeline and deployed using **Streamlit** for real-time predictions.

---

## 🚀 Project Highlights

* ✅ End-to-End ML Workflow
* ✅ Feature Engineering (Income-to-Loan Ratio)
* ✅ Pipeline with ColumnTransformer
* ✅ OneHotEncoding for categorical features
* ✅ Model Deployment using Streamlit
* ✅ Real-time prediction dashboard

---

## 🧠 Problem Statement

Financial institutions need to assess whether a customer is likely to default on a loan.
This project helps in predicting **credit risk** using machine learning.

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Joblib

---

## 📂 Project Structure

```
├── app.py                  # Streamlit Application
├── pipeline_final.pkl      # Trained ML Pipeline
├── credit_price.ipynb      # Model Training Notebook
├── dataset.csv             # Dataset
└── README.md
```

---

## ⚙️ Machine Learning Workflow

1. Data Cleaning & Preprocessing
2. Feature Engineering
3. Train-Test Split
4. ColumnTransformer:

   * Numerical → StandardScaler
   * Categorical → OneHotEncoder
5. Model Training (Logistic Regression)
6. Pipeline Creation
7. Model Saving using Joblib

---

## 📈 Key Feature

**Income-to-Loan Ratio**

```
income_to_loan_ratio = person_income / loan_amnt
```

---

## 💻 Streamlit App Features

* Interactive user input panel
* Real-time risk prediction
* Probability visualization
* Clean UI dashboard

---

## 🎯 Output

* ✅ High Risk Customer
* ✅ Low Risk Customer
* 📊 Risk Probability Score

---

## 🧪 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🙏 Acknowledgements

Special thanks to my mentors and guides:

* Swathi Mam (Trainer)
* Rahul Sir (Mentor)
* Raghuram Sir (Programme Managing Director)
* Vishwanath Nyatani
* Kirtika Reddy Madam

for their continuous support and guidance throughout this project.

---

## 📌 Conclusion

This project demonstrates how to build a **production-ready ML system** from data preprocessing to deployment.

---

⭐ If you like this project, give it a star!
