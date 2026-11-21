import pandas as pd
import numpy as np

df = pd.read_csv("21_retail_customer_orders.csv")

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.dtypes)
print(df.head())
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())
print("Duplicate Order_ID:", df["Order_ID"].duplicated().sum())
print("Identical duplicate rows:", df.duplicated().sum())

rows_before = len(df)
df = df.drop_duplicates(subset="Order_ID", keep="first").reset_index(drop=True)
print(f"Rows before: {rows_before}, after: {len(df)}")

qty_median = round(df["Quantity"].median())
df["Quantity"] = df["Quantity"].fillna(qty_median)
df["Unit_Price"] = df["Unit_Price"].fillna(
    df.groupby("Product_Name")["Unit_Price"].transform("median")
)
df["Unit_Price"] = df["Unit_Price"].fillna(df["Unit_Price"].median())
print("\nMissing after cleaning:")
print(df.isnull().sum())

df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Month_Name"] = df["Order_Date"].dt.month_name()
df["Weekday"] = df["Order_Date"].dt.day_name()
df["Order_Value"] = df["Quantity"] * df["Unit_Price"]

print("\nTotal orders:", df["Order_ID"].nunique())
print("Total units:", int(df["Quantity"].sum()))
print(f"Total revenue: {df['Order_Value'].sum():,.2f}")

print("\nTop products by revenue:")
print(df.groupby("Product_Name")["Order_Value"].sum().sort_values(ascending=False).head(10).round(2))

category_summary = df.groupby("Category").agg(
    Total_Quantity=("Quantity", "sum"),
    Total_Revenue=("Order_Value", "sum"),
)
category_summary["Quantity_Rank"] = category_summary["Total_Quantity"].rank(ascending=False).astype(int)
category_summary["Revenue_Rank"] = category_summary["Total_Revenue"].rank(ascending=False).astype(int)
print("\nCategory ranking:")
print(category_summary.sort_values("Total_Revenue", ascending=False).round(2))

print("\nAverage order value by payment method:")
print(df.groupby("Payment_Method")["Order_Value"].agg(Orders="count", Average_Order_Value="mean")
      .sort_values("Average_Order_Value", ascending=False).round(2))

weekday_summary = df.groupby("Weekday").agg(
    Orders=("Order_ID", "count"),
    Revenue=("Order_Value", "sum"),
).sort_values("Orders", ascending=False)
print("\nOrders by weekday:")
print(weekday_summary.round(2))

monthly = df.groupby(df["Order_Date"].dt.to_period("M"))["Order_Value"].sum()
print("\nMonthly sales:")
print(monthly.round(2))
print("Peak month:", monthly.idxmax(), f"{monthly.max():,.2f}")

category_sales_dict = df.groupby("Category")["Order_Value"].sum().round(2).to_dict()
print("\nSales by category:")
for category, sales in category_sales_dict.items():
    print(f"  {category}: {sales:,.2f}")

unique_categories = set(df["Category"])
print("Unique categories:", unique_categories)


def category_sales(category_name, data=df):
    if category_name not in set(data["Category"]):
        return None
    return data.loc[data["Category"] == category_name, "Order_Value"].sum()


for name in ["Office", "Home", "Toys"]:
    result = category_sales(name)
    print(f"category_sales('{name}'):", "not found" if result is None else f"{result:,.2f}")

values = df["Order_Value"].to_numpy()
print(f"\nMean: {np.mean(values):,.2f}")
print(f"Median: {np.median(values):,.2f}")
print(f"Std: {np.std(values):,.2f}")

q1 = np.percentile(values, 25)
q3 = np.percentile(values, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
print(f"\nQ1: {q1:,.2f}  Q3: {q3:,.2f}  IQR: {iqr:,.2f}")
print(f"Bounds: {lower_bound:,.2f} to {upper_bound:,.2f}")

high_outliers = df[df["Order_Value"] > upper_bound]
low_outliers = df[df["Order_Value"] < lower_bound]
print("High outliers:", len(high_outliers), " Low outliers:", len(low_outliers))
print(high_outliers.sort_values("Order_Value", ascending=False)
      [["Order_ID", "Product_Name", "Quantity", "Unit_Price", "Order_Value"]])
high_outliers.to_csv("21_high_value_outliers.csv", index=False)

df["Order_Band"] = np.select(
    [df["Order_Value"] < q1, df["Order_Value"] > q3],
    ["Low", "High"],
    default="Medium",
)
band_summary = df.groupby("Order_Band").agg(
    Orders=("Order_ID", "count"),
    Total_Revenue=("Order_Value", "sum"),
)
band_summary["Revenue_Share_%"] = band_summary["Total_Revenue"] / band_summary["Total_Revenue"].sum() * 100
band_summary = band_summary.reindex(["Low", "Medium", "High"])
print("\nOrder bands:")
print(band_summary.round(2))


class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price

    def describe(self):
        return f"{self.name}: average unit price {self.price:,.2f}"


avg_prices = df.groupby("Product_Name")["Unit_Price"].mean()
products = [Product(name, price) for name, price in avg_prices.items()]
print()
for p in products:
    print(p.describe())

product_rev = df.groupby("Product_Name")["Order_Value"].sum().sort_values(ascending=False)
cat_qty = df.groupby("Category")["Quantity"].sum().sort_values(ascending=False)
cat_rev = df.groupby("Category")["Order_Value"].sum().sort_values(ascending=False)
pay_aov = df.groupby("Payment_Method")["Order_Value"].mean().sort_values(ascending=False)
weekday_orders = df.groupby("Weekday")["Order_ID"].count().sort_values(ascending=False)
high_share = band_summary.loc["High", "Revenue_Share_%"]
high_orders = band_summary.loc["High", "Orders"]

print(f"""
Recommendations
1. {product_rev.index[0]} earns the most revenue ({product_rev.iloc[0]:,.0f}) and {product_rev.index[-1]} the least ({product_rev.iloc[-1]:,.0f}).
2. {cat_qty.index[0]} sells the most units ({cat_qty.iloc[0]:,.0f}) but {cat_rev.index[0]} earns the most revenue ({cat_rev.iloc[0]:,.0f}).
3. {pay_aov.index[0]} orders average {pay_aov.iloc[0]:,.0f}; {pay_aov.index[-1]} orders average {pay_aov.iloc[-1]:,.0f}.
4. {weekday_orders.index[0]} has the most orders ({weekday_orders.iloc[0]}), {weekday_orders.index[-1]} the fewest ({weekday_orders.iloc[-1]}); peak month is {monthly.idxmax()}.
5. The {high_orders} High-band orders (above {q3:,.0f}) bring {high_share:.1f}% of revenue.
""")

df.to_csv("21_retail_cleaned.csv", index=False)
print("Saved 21_retail_cleaned.csv:", df.shape)