import pandas as pd
import numpy as np


df = pd.read_csv("21_retail_customer_orders.csv")

print("=== TASK 1: LOAD AND INSPECT ===")
print("Dimensions (rows, columns):", df.shape)
print("\nColumn names:", list(df.columns))
print("\nData types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary of numeric columns:")
print(df.describe())

print("\n=== TASK 2: MISSING VALUES AND DUPLICATES ===")
print("Missing values per column:")
print(df.isnull().sum())

print("\nDuplicate Order_ID count:", df["Order_ID"].duplicated().sum())
print("Fully identical duplicate rows:", df.duplicated().sum())


dups = df[df["Order_ID"].duplicated(keep=False)].sort_values("Order_ID")
print("\nSample of duplicated orders:")
print(dups.head(6))

rows_before = len(df)
df = df.drop_duplicates(subset="Order_ID", keep="first").reset_index(drop=True)
print(f"\nRows before: {rows_before}, after removing duplicates: {len(df)}")


print("\n=== TASK 3: HANDLE MISSING VALUES ===")
print("Missing Quantity :", df["Quantity"].isnull().sum())
print("Missing Unit_Price:", df["Unit_Price"].isnull().sum())


qty_median = round(df["Quantity"].median())
df["Quantity"] = df["Quantity"].fillna(qty_median)

df["Unit_Price"] = df["Unit_Price"].fillna(
    df.groupby("Product_Name")["Unit_Price"].transform("median")
)

df["Unit_Price"] = df["Unit_Price"].fillna(df["Unit_Price"].median())

print("Quantity filled with:", qty_median)
print("\nMissing values after cleaning:")
print(df.isnull().sum())


print("\n=== TASK 4: DATE FEATURES ===")
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Month_Name"] = df["Order_Date"].dt.month_name()
df["Weekday"] = df["Order_Date"].dt.day_name()

print("Order_Date type:", df["Order_Date"].dtype)
print("Date range:", df["Order_Date"].min().date(), "to", df["Order_Date"].max().date())
print(df[["Order_ID", "Order_Date", "Year", "Month", "Month_Name", "Weekday"]].head())


df.to_csv("21_retail_cleaned.csv", index=False)
print("\nSaved cleaned data:", df.shape)
