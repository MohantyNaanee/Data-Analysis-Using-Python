# ---------------------------------------------
# Data Analysis Using Python
# Technologies: pandas, matplotlib, seaborn, scikit-learn
# ---------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import warnings

# -----------------------------
# Suppress FutureWarnings
# -----------------------------
warnings.simplefilter(action='ignore', category=FutureWarning)

# -----------------------------
# 0️⃣ Create directories to save outputs
# -----------------------------
base_dir = r"C:\Users\aditya\3D Objects\Data-Analysis-Customer-Churn-Python"
plots_dir = os.path.join(base_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# -----------------------------
# 1️⃣ Load Data
# -----------------------------
dataset_path = os.path.join(base_dir, "customer_churn1.csv")
df = pd.read_csv(dataset_path)

print(df.info())
print(df.describe(include='all'))

# -----------------------------
# 2️⃣ Data Cleaning
# -----------------------------
# Convert numeric columns
for col in ['MonthlyCharges', 'tenure', 'TotalCharges']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Impute missing numeric values
num_cols = ['MonthlyCharges', 'tenure', 'TotalCharges']
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Convert categorical columns
cat_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod", "Churn"]
for col in cat_cols:
    df[col] = df[col].astype('category')

# -----------------------------
# 3️⃣ Gender Analysis
# -----------------------------
df['gender_num'] = df['gender'].apply(lambda x: 1 if x=='Male' else 0)
gender_counts = df['gender_num'].value_counts()
print("Male:", gender_counts.get(1,0), "Female:", gender_counts.get(0,0))

plt.figure(figsize=(6,4))
sns.barplot(x=['Male','Female'], y=[gender_counts.get(1,0), gender_counts.get(0,0)],
            palette=['blue','pink'])
plt.title("Gender Distribution")
plt.ylabel("Count")
plt.savefig(os.path.join(plots_dir, "gender_distribution.png"))
plt.close()

# -----------------------------
# 4️⃣ Senior Citizens Gender Analysis
# -----------------------------
df['SeniorCitizen'] = df['SeniorCitizen'].apply(lambda x: 'Yes' if x==1 else 'No')
senior_df = df[df['SeniorCitizen']=='Yes']
senior_male = sum(senior_df['gender_num']==1)
senior_female = sum(senior_df['gender_num']==0)
print("Senior Male:", senior_male, "Senior Female:", senior_female)

plt.figure(figsize=(6,4))
plt.bar(['Male','Female'], [senior_male, senior_female], color=['blue','pink'])
plt.title("Senior Citizens Gender Distribution")
plt.ylabel("Count")
plt.savefig(os.path.join(plots_dir, "senior_gender_distribution.png"))
plt.close()

# -----------------------------
# 5️⃣ Internet Service Distribution
# -----------------------------
no_internet = sum(df['InternetService']=='No')
has_internet = len(df) - no_internet
print("No Internet:", no_internet, "Has Internet:", has_internet)

plt.figure(figsize=(6,6))
plt.pie([no_internet, has_internet],
        labels=[f"No Internet\n{no_internet} ({no_internet/len(df)*100:.1f}%)",
                f"Has Internet\n{has_internet} ({has_internet/len(df)*100:.1f}%)"],
        colors=['red','green'], autopct='%1.1f%%')
plt.title("Internet Service Distribution")
plt.savefig(os.path.join(plots_dir, "internet_service_distribution.png"))
plt.close()

# -----------------------------
# 6️⃣ Phone Service Analysis
# -----------------------------
no_phone = sum(df['PhoneService']=='No')
has_phone = sum(df['PhoneService']=='Yes')
both_phone_multi = sum((df['PhoneService']=='Yes') & (df['MultipleLines']=='Yes'))
print("No Phone:", no_phone, "Phone without Multi:", (has_phone-both_phone_multi),
      "Phone with Multi:", both_phone_multi)

phone_counts = [no_phone, has_phone-both_phone_multi, both_phone_multi]
phone_labels = ["No Phone Service", "Has Phone Without Multiple Lines", "Has Phone With Multiple Lines"]
plt.figure(figsize=(7,5))
sns.barplot(x=phone_labels, y=phone_counts, palette=['lightblue','lightgreen','lightcoral'])
plt.title("Phone Service Distribution")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig(os.path.join(plots_dir, "phone_service_distribution.png"))
plt.close()

# -----------------------------
# 7️⃣ Payment Method Distribution
# -----------------------------
payment_counts = df['PaymentMethod'].value_counts()
print(payment_counts)

plt.figure(figsize=(8,5))
sns.barplot(x=payment_counts.index, y=payment_counts.values, palette="Set3")
plt.title("Payment Methods Distribution")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig(os.path.join(plots_dir, "payment_methods_distribution.png"))
plt.close()

# -----------------------------
# 8️⃣ TotalCharges Analysis
# -----------------------------
df['TotalChargesRange'] = pd.cut(df['TotalCharges'], bins=np.arange(0,10001,1000))
totalcharges_summary = df.groupby('TotalChargesRange').size().reset_index(name='Count')
totalcharges_summary['TotalChargesRange'] = totalcharges_summary['TotalChargesRange'].astype(str)

print(totalcharges_summary)

plt.figure(figsize=(8,5))
sns.lineplot(x='TotalChargesRange', y='Count', data=totalcharges_summary, marker='o', color='blue')
plt.xticks(rotation=45)
plt.title("TotalCharges Distribution")
plt.ylabel("Count")
plt.xlabel("TotalCharges Range")
plt.savefig(os.path.join(plots_dir, "totalcharges_distribution.png"))
plt.close()

# -----------------------------
# 9️⃣ Summary Outputs
# -----------------------------
print("Max TotalCharges:", df['TotalCharges'].max())
print("Min TotalCharges:", df['TotalCharges'].min())
print("Mean TotalCharges:", df['TotalCharges'].mean())
print("Median TotalCharges:", df['TotalCharges'].median())

# -----------------------------
# 🔟 Save Cleaned Data
# -----------------------------
df.to_csv(os.path.join(base_dir, "customerchurn_cleaned.csv"), index=False)
print("Cleaned dataset saved!")
