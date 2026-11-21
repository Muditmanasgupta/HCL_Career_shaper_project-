
import pandas as pd
import numpy as np


df = pd.read_csv("21_retail_cleaned.csv", parse_dates=["Order_Date"])
print("Loaded cleaned data:", df.shape)


df["Order_Value"] = df["Quantity"] * df["Unit_Price"]
print("\n=== TASK 6: ORDER_VALUE COLUMN ===")
print(df[["Order_ID", "Quantity", "Unit_Price", "Order_Value"]].head())


total_orders = df["Order_ID"].nunique()
total_units = df["Quantity"].sum()
total_revenue = df["Order_Value"].sum()

print("\n=== TASK 5: OVERALL TOTALS ===")
print("Total orders       :", total_orders)
print("Total units sold   :", int(total_units))
print(f"Total sales revenue: {total_revenue:,.2f}")


product_revenue = (
    df.groupby("Product_Name")["Order_Value"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print("\n=== TASK 7: TOP 10 PRODUCTS BY REVENUE ===")
print("Unique products in the data:", df["Product_Name"].nunique())
print(product_revenue.round(2))


category_summary = df.groupby("Category").agg(
    Total_Quantity=("Quantity", "sum"),
    Total_Revenue=("Order_Value", "sum"),
)
category_summary["Quantity_Rank"] = category_summary["Total_Quantity"].rank(ascending=False).astype(int)
category_summary["Revenue_Rank"] = category_summary["Total_Revenue"].rank(ascending=False).astype(int)
category_summary = category_summary.sort_values("Total_Revenue", ascending=False)

print("\n=== TASK 8: CATEGORY RANKING ===")
print(category_summary.round(2))

payment_aov = (
    df.groupby("Payment_Method")["Order_Value"]
    .agg(Orders="count", Average_Order_Value="mean")
    .sort_values("Average_Order_Value", ascending=False)
)
print("\n=== TASK 9: AVERAGE ORDER VALUE BY PAYMENT METHOD ===")
print(payment_aov.round(2))


weekday_summary = df.groupby("Weekday").agg(
    Orders=("Order_ID", "count"),
    Revenue=("Order_Value", "sum"),
).sort_values("Orders", ascending=False)
print("\n=== TASK 10a: BUSIEST WEEKDAYS ===")
print(weekday_summary.round(2))


daily_orders = df.groupby("Order_Date")["Order_ID"].count().sort_values(ascending=False)
print("\nTop 5 busiest individual dates (orders):")
print(daily_orders.head(5))

df["Year_Month"] = df["Order_Date"].dt.to_period("M")
monthly = df.groupby("Year_Month")["Order_Value"].sum()
print("\n=== TASK 10b: MONTHLY SALES TREND (Year-Month) ===")
print(monthly.round(2))
print("\nPeak sales month:", monthly.idxmax(), "with revenue", f"{monthly.max():,.2f}")

month_total = df.groupby("Month_Name")["Order_Value"].sum().sort_values(ascending=False)
print("\nTotal revenue by calendar month (both years combined):")
print(month_total.round(2))

df.drop(columns=["Year_Month"]).to_csv("21_retail_cleaned.csv", index=False)
print("\nSaved 21_retail_cleaned.csv with the Order_Value column.")
