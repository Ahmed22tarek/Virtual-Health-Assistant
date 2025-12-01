import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# load data
df = pd.read_csv('mtsamples.csv')

# Data Cleaning & Preparation
df_clean = df.dropna(subset=['transcription']).copy()

# Focus on top 5 specialties
top_specialties = df_clean['medical_specialty'].value_counts().head(5).index
df_filtered = df_clean[df_clean['medical_specialty'].isin(top_specialties)].copy()

print(f"\n chosen specialties : {list(top_specialties)}")
print(f" no of samples : {df_filtered.shape[0]}")

# 3. استخراج الميزات (Feature Engineering)
# تحويل النصوص إلى أرقام باستخدام TF-IDF (أهم 1000 كلمة)
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
X = tfidf.fit_transform(df_filtered['transcription'])
y = df_filtered['medical_specialty']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
print("\n يتم تدريب الغالي")
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("-" * 30)
print(f"Accuracy : {acc:.4f}")
print(f"F1 Score : {f1:.4f}")
print("-" * 30)
print("\n (Classification Report):\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix - Baseline Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()