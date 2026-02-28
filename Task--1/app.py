# ======================================
# IMPORT LIBRARIES
# ======================================
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

# ======================================
# LOAD DATA
# ======================================
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

df['Order Date'] = pd.to_datetime(df['Order Date'])
df = df.sort_values('Order Date')

# Monthly aggregation
monthly = df.groupby(
    pd.Grouper(key='Order Date', freq='ME')
)[['Sales','Profit']].sum().reset_index()

monthly['Month_Number'] = np.arange(len(monthly))
monthly['Month'] = monthly['Order Date'].dt.month
monthly['Year'] = monthly['Order Date'].dt.year
monthly['Quarter'] = monthly['Order Date'].dt.quarter

monthly['Lag1'] = monthly['Sales'].shift(1)
monthly['Lag2'] = monthly['Sales'].shift(2)
monthly['Lag3'] = monthly['Sales'].shift(3)
monthly['RollingMean3'] = monthly['Sales'].rolling(3).mean()

monthly.dropna(inplace=True)
monthly.reset_index(drop=True, inplace=True)

features = [
    'Month_Number','Month','Year','Quarter',
    'Lag1','Lag2','Lag3','RollingMean3'
]

X = monthly[features]
y_sales = monthly['Sales']
y_profit = monthly['Profit']

# ======================================
# TRAIN-TEST SPLIT
# ======================================
X_train, X_test, y_train_sales, y_test_sales = train_test_split(
    X, y_sales, test_size=0.2, shuffle=False)

_, _, y_train_profit, y_test_profit = train_test_split(
    X, y_profit, test_size=0.2, shuffle=False)

# ======================================
# TRAIN MODELS
# ======================================
sales_model = RandomForestRegressor(n_estimators=400, max_depth=10, random_state=42)
profit_model = RandomForestRegressor(n_estimators=400, max_depth=10, random_state=42)

sales_model.fit(X_train, y_train_sales)
profit_model.fit(X_train, y_train_profit)

# ======================================
# MODEL EVALUATION
# ======================================
sales_pred_test = sales_model.predict(X_test)
profit_pred_test = profit_model.predict(X_test)

sales_mae = mean_absolute_error(y_test_sales, sales_pred_test)
sales_rmse = np.sqrt(mean_squared_error(y_test_sales, sales_pred_test))
sales_r2 = r2_score(y_test_sales, sales_pred_test)

# ======================================
# CONFIDENCE FUNCTION
# ======================================
def predict_with_confidence(model, X_row):
    tree_preds = np.array([tree.predict(X_row.values)[0] for tree in model.estimators_])
    mean_pred = tree_preds.mean()
    std_pred = tree_preds.std()
    confidence = max(0, 100 - (std_pred / abs(mean_pred))*100) if mean_pred!=0 else 0
    return mean_pred, confidence


# ======================================
# HOME ROUTE
# ======================================
@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        date_input = request.form["date"]
        future_date = pd.to_datetime(date_input)
        last_date = monthly['Order Date'].max()

        months_to_predict = (
            (future_date.year - last_date.year) * 12 +
            (future_date.month - last_date.month)
        )

        if months_to_predict <= 0:
            return render_template("index.html",
                                   error="Please enter a future date.")

        last_row = monthly.tail(1).copy()

        for _ in range(months_to_predict):

            new_row = last_row.copy()

            new_row['Month_Number'] += 1
            new_row['Order Date'] += pd.DateOffset(months=1)

            new_row['Month'] = new_row['Order Date'].dt.month
            new_row['Year'] = new_row['Order Date'].dt.year
            new_row['Quarter'] = new_row['Order Date'].dt.quarter

            new_row['Lag3'] = new_row['Lag2']
            new_row['Lag2'] = new_row['Lag1']
            new_row['Lag1'] = new_row['Sales']

            new_row['RollingMean3'] = new_row[['Lag1','Lag2','Lag3']].mean(axis=1)

            X_future = new_row[features]

            sales_pred, conf_s = predict_with_confidence(sales_model, X_future)
            profit_pred, conf_p = predict_with_confidence(profit_model, X_future)

            new_row['Sales'] = sales_pred
            new_row['Profit'] = profit_pred

            last_row = new_row

        status = "PROFIT ✅" if profit_pred >= 0 else "LOSS ❌"

        # Graph
        plt.figure(figsize=(8,4))
        plt.plot(monthly['Month_Number'], monthly['Sales'])
        plt.scatter(new_row['Month_Number'], sales_pred)
        plt.title("Sales Forecast")
        plt.xlabel("Month Index")
        plt.ylabel("Sales")

        graph_path = os.path.join("static", "forecast.png")
        plt.savefig(graph_path)
        plt.close()

        return render_template("index.html",
                               sales=round(sales_pred,2),
                               profit=round(profit_pred,2),
                               status=status,
                               conf_s=round(conf_s,2),
                               conf_p=round(conf_p,2),
                               mae=round(sales_mae,2),
                               rmse=round(sales_rmse,2),
                               r2=round(sales_r2,2),
                               graph=True)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
