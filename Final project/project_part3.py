import pandas as pd
import numpy as np

df = pd.read_csv("21_retail_cleaned.csv", parse_dates=["Order_Date"])
print("Loaded cleaned data:", df.shape)

print("\n=== TASK 11: DICTIONARY AND SET ===")


category_sales_dict = df.groupby("Category")["Order_Value"].sum().round(2).to_dict()
print("Sales by category (dictionary):")
for category, sales in category_sales_dict.items():
    print(f"  {category}: {sales:,.2f}")

unique_categories = set(df["Category"])
print("\nUnique categories (set):", unique_categories)
print("Number of unique categories:", len(unique_categories))

print("\n=== TASK 12: FUNCTION ===")

def category_sales(category_name, data=df):
    """Return total sales (Order_Value) for the given category name."""
    if category_name not in set(data["Category"]):
        return None  
    return data.loc[data["Category"] == category_name, "Order_Value"].sum()

for name in ["Office", "Home", "Toys"]:
    result = category_sales(name)
    if result is None:
        print(f"category_sales('{name}') -> category not found")
    else:
        print(f"category_sales('{name}') -> {result:,.2f}")


print("\n=== TASK 13: NUMPY STATISTICS OF ORDER_VALUE ===")
values = df["Order_Value"].to_numpy()

mean_value = np.mean(values)
median_value = np.median(values)
std_value = np.std(values)

print(f"Mean              : {mean_value:,.2f}")
print(f"Median            : {median_value:,.2f}")
print(f"Standard deviation: {std_value:,.2f}")


print("\n=== TASK 14: IQR OUTLIERS AND ORDER-VALUE BANDS ===")

q1 = np.percentile(values, 25)
q3 = np.percentile(values, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print(f"Q1 = {q1:,.2f}")
print(f"Q3 = {q3:,.2f}")
print(f"IQR = {iqr:,.2f}")
print(f"Lower bound = {lower_bound:,.2f}")
print(f"Upper bound = {upper_bound:,.2f}")

high_outliers = df[df["Order_Value"] > upper_bound]
low_outliers = df[df["Order_Value"] < lower_bound]
print("\nUnusually high orders:", len(high_outliers))
print("Unusually low orders :", len(low_outliers))
if len(high_outliers) > 0:
    print(high_outliers.sort_values("Order_Value", ascending=False)
          [["Order_ID", "Product_Name", "Quantity", "Unit_Price", "Order_Value"]].head(10))
    high_outliers.to_csv("21_high_value_outliers.csv", index=False)


conditions = [df["Order_Value"] < q1, df["Order_Value"] > q3]
df["Order_Band"] = np.select(conditions, ["Low", "High"], default="Medium")

band_summary = df.groupby("Order_Band").agg(
    Orders=("Order_ID", "count"),
    Total_Revenue=("Order_Value", "sum"),
)
band_summary["Revenue_Share_%"] = (band_summary["Total_Revenue"] / band_summary["Total_Revenue"].sum() * 100)
band_summary = band_summary.reindex(["Low", "Medium", "High"])
print("\nOrder-value bands:")
print(band_summary.round(2))


print("\n=== TASK 15: PRODUCT CLASS ===")

class Product:
    def __init__(self, name, price):
        self.name = name       
        self.price = price      

    def describe(self):
        return f"{self.name}: average unit price {self.price:,.2f}"

avg_prices = df.groupby("Product_Name")["Unit_Price"].mean()
products = [Product(name, price) for name, price in avg_prices.items()]
for p in products:
    print(p.describe())


product_rev = df.groupby("Product_Name")["Order_Value"].sum().sort_values(ascending=False)
cat_qty = df.groupby("Category")["Quantity"].sum().sort_values(ascending=False)
cat_rev = df.groupby("Category")["Order_Value"].sum().sort_values(ascending=False)
pay_aov = df.groupby("Payment_Method")["Order_Value"].mean().sort_values(ascending=False)
weekday_orders = df.groupby("Weekday")["Order_ID"].count().sort_values(ascending=False)
monthly = df.groupby(df["Order_Date"].dt.to_period("M"))["Order_Value"].sum()
high_share = band_summary.loc["High", "Revenue_Share_%"]
high_orders = band_summary.loc["High", "Orders"]

print("\n=== FIVE SALES RECOMMENDATIONS ===")
print(f"""
1. Promote the top products. {product_rev.index[0]} earns the most revenue
   ({product_rev.iloc[0]:,.0f}), while {product_rev.index[-1]} earns the least
   ({product_rev.iloc[-1]:,.0f}). Give the leaders priority in ads and homepage space,
   and test discounts or bundles on the weakest product.

2. Use category strengths differently. {cat_qty.index[0]} sells the most units
   ({cat_qty.iloc[0]:,.0f}) but {cat_rev.index[0]} earns the most revenue
   ({cat_rev.iloc[0]:,.0f}). Use high-volume categories to attract customers and
   high-revenue categories for upselling.

3. Encourage higher-value payment channels. {pay_aov.index[0]} orders have the highest
   average value ({pay_aov.iloc[0]:,.0f}) and {pay_aov.index[-1]} the lowest
   ({pay_aov.iloc[-1]:,.0f}). Offer small rewards on {pay_aov.index[0]} or other
   digital payments to lift average order value.

4. Time campaigns to demand. {weekday_orders.index[0]} has the most orders
   ({weekday_orders.iloc[0]}) and {weekday_orders.index[-1]} the fewest
   ({weekday_orders.iloc[-1]}). The peak month was {monthly.idxmax()}
   ({monthly.max():,.0f}). Schedule promotions and stock before peaks and run
   offers on slow days.

5. Protect high-value customers. The {high_orders} orders in the High band
   (above {q3:,.0f}) bring {high_share:.1f}% of revenue. Give these customers
   priority delivery, bundle offers and follow-ups to keep them buying.
""")


df.to_csv("21_retail_cleaned.csv", index=False)
print("Saved 21_retail_cleaned.csv with the Order_Band column.")