# E-commerce Purchase Intent Analytics (XGBoost)

## English Version

### 📌 Overview
This repository features a machine learning pipeline and an interactive dashboard designed to predict real-time purchase intent for e-commerce visitors. The model is trained on the UCI "Online Shoppers Purchasing Intention" dataset, representing a full year (2017) of sessions from a specialized retail website.

### 🧠 Machine Learning Approach
- **Algorithm:** Extreme Gradient Boosting (XGBoost).
- **Class Imbalance:** Handled using dynamic `scale_pos_weight` to account for the low conversion rate (approx. 15%).
- **Decision Threshold:** Optimized at **0.85** to maximize precision, ensuring that "High Intent" alerts are highly reliable for marketing actions.
- **Feature Engineering:** Implementation of a custom "Time Per Product Page" metric and one-hot encoding for categorical variables (Months, Visitor Types).

### 📊 Interactive Dashboard
The Streamlit application allows users to:
1. Load predefined personas (New Prospect, Returning Customer, High-Value Buyer).
2. Manually adjust behavioral metrics (PageValue, ExitRates, Duration).
3. Analyze model biases (e.g., why returning visitors require stronger signals to be classified as "High Intent").

### 📂 Project Structure
- `app.py`: Streamlit dashboard for real-time inference.
- `train_ecommerce.py`: Full training pipeline (preprocessing, training, evaluation, export).
- `inspect_data.py`: Exploratory Data Analysis (EDA) script.
- `test_client.py`: CLI script for quick model testing.
- `sale_model.pkl`: Trained XGBoost model.
- `datas/`: Directory containing the CSV dataset.

### 🛠️ Setup
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   
   python inspect_data.py (data analysis)

   python train_ecommerce.py (model training generates sale_model.pkl)

   python test_client.py (testing case)

   streamlit run app.py (run web app several cases to test)
   
   
   
