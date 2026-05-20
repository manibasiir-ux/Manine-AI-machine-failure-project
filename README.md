# Manine AI

**Industrial Predictive Maintenance Platform with Explainable AI**

Manine AI is an AI-powered predictive maintenance system designed for industrial machine failure prediction using real-time sensor data.

This project combines:

- **Machine Learning**
- **Explainable AI (XAI)**
- **Industrial Monitoring Concepts**
- **Predictive Maintenance**
- **Interactive Streamlit Dashboard**

The platform was developed as a complete end-to-end AI engineering project to demonstrate practical ML deployment, explainability integration, and professional AI product development.

---

## Project Overview

Modern industries rely heavily on predictive maintenance systems to reduce:

- machine downtime
- maintenance costs
- unexpected failures
- production interruptions

Manine AI predicts machine failure risk using industrial sensor inputs and provides explainable AI analysis to help users understand WHY the prediction happened.

The system allows users to:

- enter machine sensor values
- receive failure probability predictions
- analyze machine risk level
- view explainable AI feature impacts
- monitor industrial health indicators

---

## Features

- **AI Failure Prediction** — Predicts machine failure risk using trained machine learning models.
- **Explainable AI (SHAP)** — Provides feature impact analysis showing which sensor values influenced the prediction.
- **Industrial Monitoring** — Simulates real-world industrial machine monitoring systems.
- **Interactive Dashboard** — Professional Streamlit UI for real-time prediction and analysis.
- **Risk Analysis** — Displays: Safe probability, Failure probability, Risk level classification.
- **Rule-Based Insights** — Additional logic-based warnings for abnormal industrial conditions.

---

## Dashboard Preview

![Screenshot (77)](https://github.com/user-attachments/assets/9f5867fc-673b-4686-a1f4-7c9f4551b0b8)

---

## Technologies Used

- **Machine Learning**
- scikit-learn
- pandas
- SHAP
- joblib
- numpy

### Models Tested
During development, multiple algorithms were tested and evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

---

## Final Model Selection

Initially, some models achieved higher accuracy scores (above 0.60).

However, the dataset contained a strong class imbalance where most samples represented normal machine conditions and relatively few represented machine failures.

As a result:

- some high-accuracy models failed to correctly predict failure cases
- the models became biased toward predicting only normal states

To solve this issue, the final system used:

- **Decision Tree Classifier**

  Used for final prediction because:

  - it handled imbalanced data better
  - it successfully detected failure cases
  - class balancing improved failure sensitivity

  Although overall accuracy decreased to approximately 0.50, the model became significantly more useful for real predictive maintenance scenarios because detecting failures is more important than maximi[...]

- **Random Forest Classifier**

  Used for Explainable AI analysis with SHAP.

  This allowed the platform to generate feature importance explanations for each prediction.

---

## Explainable AI Analysis

The system includes SHAP-based explainability.

Users can understand:

- which features increased failure risk
- which features reduced failure risk
- the relative impact of each sensor input

### Interpretation

- **Positive Impact (+)** — This feature pushed the prediction TOWARD machine failure.
- **Negative Impact (-)** — This feature pushed the prediction AWAY from machine failure.

---

## Dataset Information

This project uses a research-focused industrial machine sensor dataset designed for predictive maintenance and failure risk classification.

The dataset simulates industrial monitoring environments using real-time machine sensor readings.

### Dataset Features

- **Temperature** — Machine operating temperature in Celsius
- **Vibration** — Machine vibration frequency in Hz
- **Power_Usage** — Power consumption in kilowatts
- **Humidity** — Environmental humidity percentage
- **Machine_Type** — Type of industrial machine
- **Failure_Risk** — Prediction target (Failure / Normal)

---

## Machine Types

The dataset includes multiple industrial machine categories:

- Drill
- Lathe
- Mill

---

## Purpose of the Dataset

The dataset was designed for:

- predictive maintenance research
- industrial AI experimentation
- machine health monitoring
- explainable AI applications
- educational ML projects

---

## Project Structure

```
Manine/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── machine_failure_dataset.csv
│    
├── decisiontree_machine_failure_model.pkl
├── random_forest_explainer.pkl


```

---

## Installation

1. Clone Repository

```bash
git clone https://github.com/manibasiir-ux/Manine.git
```

2. Open Project Folder

```bash
cd Manine
```

3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

1. Run dashboard

```bash
streamlit run app.py
```

After running, Streamlit automatically opens the dashboard in your browser.

---

## Example Workflow

1. Select machine type
2. Enter sensor values
3. Run AI prediction
4. Analyze failure probability
5. Review SHAP explainability
6. Monitor industrial risk indicators

---

## Future Improvements

Possible future upgrades:

- real-time IoT sensor integration
- cloud deployment
- advanced deep learning models
- database integration
- historical prediction storage
- live industrial monitoring systems
- Docker deployment
- REST API integration

---

## License

This project is licensed under the Apache License 2.0.

---

## Creator

**Mani Basir**

AI/ML Developer
Industrial AI Enthusiast
Robotics & Automation Enthusiast

**Links**

- GitHub:
  [GitHub Profile](https://github.com/maniasiir-ux.com)
- LinkedIn:
  [LinkedIn Profile](https://linkedin.com/in/mani-basir.com)
- Telegram:
  [Telegram Profile](https://t.me/manibasir.com)

---

## Acknowledgment

Special thanks to everyone who contributed guidance, explanations, and technical understanding during the development of this project. The learning process behind this system played a major role in un[...]
