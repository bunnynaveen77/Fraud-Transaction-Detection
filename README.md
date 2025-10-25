Fraudulent Transactions Detection using Machine Learning
 
Overview

Financial fraud is one of the most challenging issues in the digital economy. This project leverages Machine Learning (ML) techniques to detect fraudulent financial transactions by identifying suspicious behavior patterns.
The goal is to build a reliable model that can classify transactions as Fraud or Non-Fraud, enabling financial institutions to proactively reduce risk and enhance security.

🧠 Key Objectives

Analyze transactional behavior to uncover fraud patterns.

Engineer domain-specific features to capture transaction anomalies.

Compare performance across multiple models — Logistic Regression, Random Forest, and XGBoost.

Provide actionable insights for fraud prevention strategies.

📊 Dataset Information

Dataset Name: fraud.csv
Source: Simulated financial transactions dataset used for machine learning research.

Column Name	Description
step	Time step of the transaction (hourly sequence)
type	Type of transaction (CASH_OUT, TRANSFER, PAYMENT, etc.)
amount	Amount of money transferred
nameOrig	ID of the sender
oldbalanceOrg	Sender’s balance before transaction
newbalanceOrig	Sender’s balance after transaction
nameDest	ID of the recipient
oldbalanceDest	Receiver’s balance before transaction
newbalanceDest	Receiver’s balance after transaction
isFraud	Target variable (1 = Fraud, 0 = Non-Fraud)
isFlaggedFraud	Flagged suspicious transaction indicator

Dataset Shape:

Training data: (1,261,680, 14)

Testing data: (315,421, 14)

🧩 Project Workflow
1. Data Preprocessing

Loaded and explored the dataset to understand distributions and detect missing values.

Encoded categorical variables using One-Hot Encoding.

Scaled numeric features for uniform representation.

Balanced class imbalance using Random Under-Sampling.

2. Feature Engineering

Added several custom features to enhance detection power:

amount_log: Log transformation of transaction amount.

orig_balance_change: Change in sender’s balance.

dest_balance_change: Change in receiver’s balance.

isHighAmount: Flag for unusually high transactions (>200K).

hour_of_day: Extracted hour from transaction step.

isMerchantDest: Indicator if recipient is a merchant.

isSelfTransfer: Flag for self-transfers (same sender and receiver).

3. Model Training

Compared multiple models:

Logistic Regression – Baseline linear classifier.

Random Forest – Ensemble of decision trees for robust prediction.

XGBoost – Gradient-boosted trees for high accuracy.

4. Model Evaluation

Metrics used:

Confusion Matrix

Precision, Recall, F1-Score

ROC-AUC and PR Curves

Average Precision Score

5. Feature Importance

Analyzed the most influential predictors contributing to fraud detection using Random Forest and XGBoost feature importance plots.

🧪 Results & Insights
Model	ROC-AUC	Avg Precision	Notes
Logistic Regression	~0.94	0.78	Good baseline; fast to train
Random Forest	~0.99	0.92	Excellent recall; interpretable
XGBoost	~0.995	0.95	Best overall performance

Top Predictive Features:

amount_log

orig_balance_change

dest_balance_change

isHighAmount

hour_of_day

🧭 Business Insights & Actionable Plan

Fraud Patterns: High-value transfers between non-merchant accounts and self-transfers were common red flags.

Recommendations:

Implement real-time flagging for high-amount self-transfers.

Combine rule-based thresholds with ML model predictions for hybrid fraud prevention.

Periodically retrain models to adapt to evolving fraud behavior.

Establish feedback loops to collect verified outcomes and improve accuracy.

Integrate advanced graph-based features (network links between users) for future work.

🧰 Tech Stack
Category	Tools / Libraries
Language	Python
Data Handling	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Machine Learning	Scikit-Learn, XGBoost
Sampling	imbalanced-learn
Workflow	Jupyter Notebook / Colab
🚀 How to Run
# Clone this repository
git clone https://github.com/yourusername/Fraudulent-Transactions-Detection.git
cd Fraudulent-Transactions-Detection

# Install dependencies
pip install -r requirements.txt

# Run the notebook or Python file
jupyter notebook Detecting_Fraud.ipynb

🧠 Future Enhancements

Integrate Deep Learning architectures (LSTM, Autoencoders) for sequential fraud detection.

Deploy the model as a REST API for real-time transaction monitoring.

Apply Graph Neural Networks (GNNs) to detect community-level fraud behavior.
