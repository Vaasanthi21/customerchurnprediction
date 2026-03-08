import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


data = pd.read_csv('customerchurn.csv')
print("Dataset Overview:")
print(data.head())

print("Checking for Missing Values:")
print(data.isnull().sum())
data= data.dropna()


categorical_columns=data.select_dtypes(include=['object']).columns
label_encoders={}
for column in categorical_columns:
    le=LabelEncoder()
    data[column]=le.fit_transform(data[column])
    label_encoders[column]=le

numerical_columns=data.select_dtypes(include=['int64','float64']).columns
numerical_columns=numerical_columns[numerical_columns!='Churn']
scaler=StandardScaler()
data[numerical_columns]=scaler.fit_transform(data[numerical_columns])

X=data.drop(columns=['Churn'])
y=data['Churn']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test,y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("Accuracy Score:",accuracy_score(y_test,y_pred))

feature_importance=model.feature_importances_
sorted_indices=np.argsort(feature_importance)[::-1]
print("Most Important Features:")
for i in sorted_indices:
    print(X.columns[i],":",feature_importance[i])

plt.figure(figsize=(10,6))
plt.bar(range(X.shape[1]),feature_importance[sorted_indices],align='center')
plt.xticks(range(X.shape[1]),X.columns[sorted_indices],rotation=90)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Random Forest")
plt.show()