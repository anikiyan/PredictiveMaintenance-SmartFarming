# 🚜 Predictive Maintenance for Smart Farming Machinery

![Status](https://img.shields.io/badge/Project-Complete-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)

This project demonstrates a full end-to-end machine learning pipeline to predict machinery failure and estimate remaining useful life (RUL) using high-frequency sensor data collected from agricultural equipment.

Built as a professional portfolio piece based on real-world agricultural telemetry scenarios — including vibration, torque, current, and thermal readings — aligned with my work at TAFE.

---

## 📊 Key Features

- ⏱️ High-frequency time-series simulation (1s intervals)
- 🔧 Rolling window feature engineering (mean, std, slope, lag)
- 🎯 Dual-task ML modeling:
  - **Classification** – failure event detection
  - **Regression** – RUL (remaining useful life) estimation
- 📈 ROC Curve, Confusion Matrix, and Executive KPIs
- 💼 Jupyter-based analytics with full business-ready reporting

---

## 🧱 Project Structure

```
PredictiveMaintenance-SmartFarming/
├── data/
│ ├── raw/ ← synthetic sensor logs
│ └── processed/ ← cleaned + feature-engineered datasets
├── notebooks/
│ ├── 01_data_cleaning.ipynb ← parsing & cleaning
│ ├── 02_feature_engineering.ipynb
│ ├── 03_model_training.ipynb
│ └── 04_reporting.ipynb ← visuals + executive metrics
├── src/ ← (optional future CLI tools)
├── reports/ ← auto-generated summaries or charts
├── requirements.txt
└── README.md
```


---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/anikiyan/PredictiveMaintenance-SmartFarming.git
cd PredictiveMaintenance-SmartFarming
```

2. Create environment and install dependencies:
```bash
pip install -r requirements.txt
```

3. Open Jupyter and run notebooks in order:
```bash
notebooks/
├── 01 → 02 → 03 → 04
```



📸 Sample Visualizations

✅ Confusion Matrix

![image](https://github.com/user-attachments/assets/e9a21d41-cb9d-44b4-94f7-2608061b55a3)


✅ Actual vs Predicted RUL

![image](https://github.com/user-attachments/assets/367a9fa8-7847-44a1-9329-dc15e7c81fc0)


✅ Failure Distribution by Operating Mode

![image](https://github.com/user-attachments/assets/03ee6309-8bde-4874-aa6f-af418f8f4907)




🧑‍💼 About Me

Ashkan Nikiyan, PhD.
Senior Data Scientist | ML/AI Expert, Time-Series Forecasting, Predictive Maintenance  
Expertise in Deep Learning, Data Pipelines, Cloud Deployment

[LinkedIn](https://www.linkedin.com/in/ashkan-nikiyan-ph-d-gda-ids-24aa1012b/) 




📬 Contact

For collaboration, feedback, or portfolio reviews — feel free to reach out.
