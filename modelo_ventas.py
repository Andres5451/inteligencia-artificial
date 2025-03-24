import pandas as pd

df = pd.read_csv("vgsales.csv")

print(df.head())

X = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]  
y = df['Global_Sales']  

df.dropna(inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

score = model.score(X_test_scaled, y_test)
print(f"Precisión del modelo: {score:.4f}")

import joblib

joblib.dump(model, 'modelo_ventas_videojuegos.pkl')

joblib.dump(scaler, 'scaler.pkl')

print("Modelo y scaler guardados correctamente.")
