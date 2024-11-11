# DECISION TREE

# Import Library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import joblib

# Baca dan Bersihkan Data
df = pd.read_csv('Source\Data\Monkey_Pox_Cases_Worldwide.csv')
df = df.dropna()
df

print(len(df))

# Menentukan risk level
median_cases = df['Confirmed_Cases'].median()
df['Risk_Level'] = df['Confirmed_Cases'].apply(lambda x: 'High' if x > median_cases else 'Low')

# Memilih fitur-fitur untuk prediksi (x) dan target variabel (y)
X = df[['Suspected_Cases', 'Hospitalized', 'Travel_History_Yes', 'Travel_History_No']]
y = df['Risk_Level']

# Membagi data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model
model = DecisionTreeClassifier(random_state=42)

# Melatih model
model.fit(X_train, y_train)

# Visualisasi Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(model, filled=True, feature_names=X.columns, class_names=['Low', 'High'], rounded=True)
plt.show()

# Hasil Prediksi
y_pred = model.predict(X_test)

# Akurasi Model
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi model:", accuracy)

# Metrik Evaluasi
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=['Low', 'High'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

df['Predicted_Risk'] = model.predict(X)  # X adalah semua data fitur (tanpa split)

# Menampilkan data prediksi = High
high_risk_countries = df[df['Predicted_Risk'] == 'High']
print(len(high_risk_countries))
print(high_risk_countries[['Country', 'Confirmed_Cases', 'Hospitalized', 'Travel_History_Yes', 'Predicted_Risk']])

# Menampilkan data prediksi = Low
low_risk_countries = df[df['Predicted_Risk'] == 'Low']
print(len(low_risk_countries))
print(low_risk_countries[['Country', 'Confirmed_Cases', 'Hospitalized', 'Travel_History_Yes', 'Predicted_Risk']])

# Menyimpan model
joblib.dump(model, "decision_tree_model.pkl")

# Menyimpan data
df.to_csv("Source\Data\Monkey_Pox_Cases_Worldwide_Predict.csv", index=False)