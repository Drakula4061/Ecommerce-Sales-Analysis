# E-COMMERCE SALES ANALYSIS
# Beginner Data Analytics Project
# Libraries: pandas, numpy, matplotlib, seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. LOAD THE DATA
# ---------------------------------------------------------
df = pd.read_csv("ecommerce_sales_analysis (1).csv")

print("First 5 rows:")
print(df.head())

print("\nShape of dataset:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
print(df.info())

# ---------------------------------------------------------
# 2. CHECK FOR MISSING VALUES AND DUPLICATES
# ---------------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

# Remove duplicate rows if any
df = df.drop_duplicates()

# ---------------------------------------------------------
# 3. DATA TYPE CONVERSION
# ---------------------------------------------------------

df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df.to_csv("ecommerce_sales_cleaned.csv", index=False)

print("\nData types after conversion:")
print(df.dtypes)

# ---------------------------------------------------------
# 4. BASIC STATISTICS
# ---------------------------------------------------------

print("\nStatistical summary:")
print(df.describe())

# ---------------------------------------------------------
# 5. CREATE NEW COLUMNS
# ---------------------------------------------------------

df["Month"] = df["Order_Date"].dt.month
df["Month_Name"] = df["Order_Date"].dt.strftime("%B")
df["Month_Year"] = df["Order_Date"].dt.to_period("M").astype(str)

# Profit Margin tells us what percentage of Net Sales becomes profit
df["Profit_Margin"] = (df["Profit"] / df["Net_Sales"]) * 100

print("\nDataset with new columns:")
print(df.head())

# ---------------------------------------------------------
# 6. KEY BUSINESS KPIs
# ---------------------------------------------------------

total_orders = df["Order_ID"].nunique()
total_quantity = df["Quantity"].sum()
total_gross_sales = df["Gross_Sales"].sum()
total_discount = df["Discount_Amount"].sum()
total_net_sales = df["Net_Sales"].sum()
total_cost = df["Cost"].sum()
total_profit = df["Profit"].sum()
average_order_value = df["Net_Sales"].mean()
average_profit_margin = df["Profit_Margin"].mean()

print("\n========== KEY PERFORMANCE INDICATORS ==========")
print("Total Orders:", total_orders)
print("Total Quantity Sold:", total_quantity)
print("Total Gross Sales:", round(total_gross_sales, 2))
print("Total Discount:", round(total_discount, 2))
print("Total Net Sales:", round(total_net_sales, 2))
print("Total Cost:", round(total_cost, 2))
print("Total Profit:", round(total_profit, 2))
print("Average Order Value:", round(average_order_value, 2))
print("Average Profit Margin:", round(average_profit_margin, 2), "%")

# ---------------------------------------------------------
# 7. SALES BY CATEGORY
# ---------------------------------------------------------

category_sales = (
    df.groupby("Category")["Net_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales by Category:")
print(category_sales)

plt.figure(figsize=(10, 6))
category_sales.plot(kind="bar")
plt.title("Net Sales by Category")
plt.xlabel("Category")
plt.ylabel("Net Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 8. PROFIT BY CATEGORY
# ---------------------------------------------------------

category_profit = (
    df.groupby("Category")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\nProfit by Category:")
print(category_profit)

plt.figure(figsize=(10, 6))
category_profit.plot(kind="bar")
plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 9. TOP 10 PRODUCTS BY SALES
# ---------------------------------------------------------

top_products = (
    df.groupby("Product")["Net_Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Products by Sales:")
print(top_products)

plt.figure(figsize=(10, 6))
top_products.sort_values().plot(kind="barh")
plt.title("Top 10 Products by Net Sales")
plt.xlabel("Net Sales")
plt.ylabel("Product")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 10. MONTHLY SALES TREND
# ---------------------------------------------------------

monthly_sales = (
    df.groupby("Month_Year")["Net_Sales"]
    .sum()
)

print("\nMonthly Sales:")
print(monthly_sales)

plt.figure(figsize=(12, 6))
monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Net Sales Trend")
plt.xlabel("Month")
plt.ylabel("Net Sales")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 11. MONTHLY PROFIT TREND
# ---------------------------------------------------------

monthly_profit = (
    df.groupby("Month_Year")["Profit"]
    .sum()
)

plt.figure(figsize=(12, 6))
monthly_profit.plot(kind="line", marker="o")
plt.title("Monthly Profit Trend")
plt.xlabel("Month")
plt.ylabel("Profit")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 12. SALES BY CITY
# ---------------------------------------------------------

city_sales = (
    df.groupby("City")["Net_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales by City:")
print(city_sales)

plt.figure(figsize=(10, 6))
city_sales.plot(kind="bar")
plt.title("Net Sales by City")
plt.xlabel("City")
plt.ylabel("Net Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 13. SALES BY CUSTOMER SEGMENT
# ---------------------------------------------------------

segment_sales = (
    df.groupby("Customer_Segment")["Net_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales by Customer Segment:")
print(segment_sales)

plt.figure(figsize=(8, 5))
segment_sales.plot(kind="bar")
plt.title("Net Sales by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Net Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 14. PAYMENT METHOD ANALYSIS
# ---------------------------------------------------------

payment_sales = (
    df.groupby("Payment_Method")["Net_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales by Payment Method:")
print(payment_sales)

plt.figure(figsize=(9, 5))
payment_sales.plot(kind="bar")
plt.title("Net Sales by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Net Sales")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 15. ORDER STATUS ANALYSIS
# ---------------------------------------------------------

status_count = df["Order_Status"].value_counts()

print("\nOrder Status:")
print(status_count)

plt.figure(figsize=(8, 5))
status_count.plot(kind="bar")
plt.title("Orders by Order Status")
plt.xlabel("Order Status")
plt.ylabel("Number of Orders")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 16. CORRELATION ANALYSIS
# ---------------------------------------------------------

numeric_columns = [
    "Quantity",
    "Unit_Price",
    "Discount_Percent",
    "Gross_Sales",
    "Discount_Amount",
    "Net_Sales",
    "Cost",
    "Profit",
    "Profit_Margin"
]

correlation = df[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 17. BUSINESS QUESTIONS
# ---------------------------------------------------------

print("\n========== BUSINESS INSIGHTS ==========")

best_category = category_sales.idxmax()
best_city = city_sales.idxmax()
best_product = top_products.idxmax()

print("Highest sales category:", best_category)
print("Highest sales city:", best_city)
print("Top product by sales:", best_product)

print("\nAnalysis completed successfully.")
