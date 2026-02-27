# Customer Churn Prediction using ANN

An end-to-end deep learning project that predicts whether a customer is likely to churn (leave the bank), built with TensorFlow/Keras and deployed as an interactive **Streamlit** web application.

---

## 📌 Project Overview

Customer churn is a critical metric for banks and financial institutions. This project uses an **Artificial Neural Network (ANN)** trained on 10,000 bank customer records to predict the probability that a given customer will exit. The trained model is served through a Streamlit UI where users can fill in customer details and get an instant churn prediction.

---

## 🗂️ Project Structure

```
├── app.py                     # Streamlit web application
├── experiments.ipynb          # Model training & preprocessing notebook
├── prediction.ipynb           # Inference/prediction notebook
├── Churn_Modelling.csv        # Dataset (10,000 bank customer records)
├── model.h5                   # Saved trained ANN model
├── scaler.pkl                 # Saved StandardScaler
├── label_encode_gender.pkl    # Saved LabelEncoder for Gender
├── onehot_encoder_geo.pkl     # Saved OneHotEncoder for Geography
├── log/                       # TensorBoard training logs
└── requirements.txt           # Python dependencies
```

---

## 🧠 Model Architecture

The ANN is a **Sequential model** built with TensorFlow/Keras:

| Layer         | Type  | Neurons | Activation |
|---------------|-------|---------|------------|
| Input         | Dense | 11      | —          |
| Hidden Layer 1| Dense | 64      | ReLU       |
| Hidden Layer 2| Dense | 32      | ReLU       |
| Output        | Dense | 1       | Sigmoid    |

- **Loss**: Binary Crossentropy  
- **Optimizer**: Adam  
- **Callbacks**: EarlyStopping, TensorBoard

---

## 📊 Dataset & Features

**Dataset**: `Churn_Modelling.csv` — 10,000 rows of European bank customer data.

**Input Features** (after preprocessing):

| Feature           | Description                          | Encoding         |
|-------------------|--------------------------------------|------------------|
| CreditScore       | Customer credit score                | StandardScaler   |
| Gender            | Male / Female                        | LabelEncoder     |
| Age               | Customer age (18–92)                 | StandardScaler   |
| Tenure            | Years with the bank (0–10)           | StandardScaler   |
| Balance           | Account balance                      | StandardScaler   |
| NumOfProducts     | Number of bank products (1–4)        | StandardScaler   |
| HasCrCard         | Has credit card? (0 / 1)             | StandardScaler   |
| IsActiveMember    | Is an active member? (0 / 1)         | StandardScaler   |
| EstimatedSalary   | Estimated annual salary              | StandardScaler   |
| Geography         | France / Germany / Spain             | OneHotEncoder    |

**Target**: `Exited` — `1` (churned) or `0` (stayed)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone git@github.com:satishmachine/ANN_Project_End_to_End.git
cd ANN_Project_End_to_End
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🖥️ Streamlit App

The app accepts the following inputs via an interactive UI:

- **Geography** — dropdown (France, Germany, Spain)
- **Gender** — dropdown (Male / Female)
- **Age** — slider (18–92)
- **Balance** — numeric input
- **Credit Score** — numeric input
- **Estimated Salary** — numeric input
- **Tenure** — slider (0–10)
- **Number of Products** — slider (1–4)
- **Has Credit Card** — dropdown (0 / 1)
- **Is Active Member** — dropdown (0 / 1)

**Output**: Churn probability (%) and a human-readable verdict — *"The customer is likely to churn"* or *"The customer is not likely to churn"* (threshold: 50%).

---

## 📈 Monitor Training with TensorBoard

```bash
tensorboard --logdir log/fit
```

Then open `http://localhost:6006` in your browser.

---

## 🛠️ Tech Stack

| Tool              | Purpose                        |
|-------------------|--------------------------------|
| Python            | Core language                  |
| TensorFlow/Keras  | ANN model building & training  |
| scikit-learn      | Preprocessing (scaler/encoders)|
| Pandas / NumPy    | Data manipulation              |
| Streamlit         | Web application UI             |
| TensorBoard       | Training visualisation         |
| Pickle            | Model artifact serialisation   |

---

## 📋 Requirements

See [`requirements.txt`](requirements.txt):

```
matplotlib>=3.10.8
numpy>=2.4.2
pandas>=2.3.3
scikeras>=0.13.0
scikit-learn>=1.8.0
streamlit>=1.54.0
tensorboard>=2.20.0
tensorflow>=2.20.0
```

