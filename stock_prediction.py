import yfinance as yf
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print("=" * 60)
print("STOCK PRICE PREDICTION USING MACHINE LEARNING")
print("=" * 60)

stock_symbol = "AAPL"

print(f"\nDownloading {stock_symbol} stock data...")

data = yf.download(
    stock_symbol,
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=True
)

data["Next_Close"] = data["Close"].shift(-1)
data.dropna(inplace=True)

X = data[["Open", "High", "Low", "Volume"]]
y = data["Next_Close"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

print("Training Random Forest Model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

print("Generating Predictions...")

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("-" * 30)
print(f"Mean Absolute Error : {mae:.2f}")
print(f"R² Score            : {r2:.4f}")

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

print("\nSample Predictions")
print(results.head(10))

results.to_csv("Predicted_Stock_Prices.csv", index=False)

print("\nPrediction file saved successfully!")
print("File Name: Predicted_Stock_Prices.csv")

print("\nProject Completed Successfully!")