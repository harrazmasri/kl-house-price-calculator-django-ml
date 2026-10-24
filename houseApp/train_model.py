from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

CURRENT_DIR = Path(__file__).resolve().parent

file_path = CURRENT_DIR / "ml_model" / "house_data.csv"

if not file_path.exists():
    file_path = CURRENT_DIR.parent / "templates" / "houseApp" / "ml_model" / "house_data.csv"

print(f"File exists: {file_path.absolute()}" if file_path.exists() else "No file exist")

data = pd.read_csv(file_path)

X = data[['bedrooms', 'bathrooms', 'size_sqft', 'age']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

model_output_path = CURRENT_DIR / "ml_model" / "house_price_model.pkl"
joblib.dump(model, model_output_path)

print('Model trained successfully')
print(f"Model R squared score: {model.score(X_test, y_test)}")