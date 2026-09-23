"""
E-Commerce Sales Data Analyzer & Comparison Dashboard
------------------------------------------------------
An upgraded, generic CSV analytics dashboard built with Streamlit, Pandas,
NumPy and Matplotlib. Works with the built-in E-Commerce sample dataset or
any user-uploaded CSV file.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="E-Commerce Sales Data Analyzer",
    page_icon="🛒",
    layout="wide",
)

RNG = np.random.default_rng(42)


# ============================================================
# 1. SAMPLE DATASET GENERATION
# ============================================================
def generate_sample_dataset(n=220):
    """Generate a realistic, varied E-Commerce sales sample dataset."""
    categories = {
        "Electronics": ["Smartphone", "Laptop", "Headphones", "Smartwatch", "Tablet", "Camera"],
        "Clothing": ["T-Shirt", "Jeans", "Jacket", "Dress", "Shoes", "Cap"],
        "Home & Kitchen": ["Mixer", "Cookware Set", "Vacuum Cleaner", "Lamp", "Bedsheet", "Chair"],
        "Beauty": ["Face Cream", "Perfume", "Lipstick", "Shampoo", "Hair Dryer", "Sunscreen"],
        "Sports": ["Yoga Mat", "Dumbbells", "Cricket Bat", "Football", "Running Shoes", "Cycle"],
    }
    cities_regions = {
        "Mumbai": "West", "Delhi": "North", "Bengaluru": "South", "Chennai": "South",
        "Kolkata": "East", "Hyderabad": "South", "Pune": "West", "Ahmedabad": "West",
        "Jaipur": "North", "Lucknow": "North",
    }
    customer_types = ["New", "Returning", "Premium"]
    payment_methods = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Cash on Delivery"]

    base_price = {
        "Electronics": (2000, 60000), "Clothing": (300, 4000),
        "Home & Kitchen": (500, 12000), "Beauty": (150, 3000), "Sports": (300, 15000),
    }

    rows = []
    start_date = pd.Timestamp("2024-01-01")
    for i in range(1, n + 1):
        category = RNG.choice(list(categories.keys()))
        product = RNG.choice(categories[category])
        city = RNG.choice(list(cities_regions.keys()))
        region = cities_regions[city]
        customer_type = RNG.choice(customer_types, p=[0.45, 0.4, 0.15])
        payment = RNG.choice(payment_methods)

        low, high = base_price[category]
        unit_price = round(float(RNG.uniform(low, high)), 2)
        quantity = int(RNG.integers(1, 8))
        discount = round(float(RNG.choice([0, 0.05, 0.1, 0.15, 0.2, 0.25], p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05])), 2)

        gross = unit_price * quantity
        sales = round(gross * (1 - discount), 2)
        cost_ratio = float(RNG.uniform(0.55, 0.8))
        cost = round(gross * cost_ratio, 2)
        profit = round(sales - cost, 2)

        rating = round(float(np.clip(RNG.normal(4.0, 0.7), 1, 5)), 1)
        order_date = start_date + pd.Timedelta(days=int(RNG.integers(0, 365)))

        rows.append({
            "Order_ID": f"ORD{1000 + i}",
            "Order_Date": order_date.strftime("%Y-%m-%d"),
            "Product": product,
            "Category": category,
            "Sub_Category": product,
            "City": city,
            "Region": region,
            "Customer_Type": customer_type,
            "Quantity": quantity,
            "Unit_Price": unit_price,
            "Discount": discount,
            "Sales": sales,
            "Cost": cost,
            "Profit": profit,
            "Payment_Method": payment,
            "Rating": rating,
        })

    return pd.DataFrame(rows)


# ============================================================
# 2. LOAD / DETECT DATASET
# ============================================================
def load_dataset(uploaded_file):
    """Safely load a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("The uploaded CSV file is empty.")
            return None
        return df
    except Exception as e:
        st.error(f"Could not read the CSV file. Error: {e}")
        return None


def detect_column_types(df):
    """Return lists of numeric, categorical and date-like columns."""
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    non_numeric = [c for c in df.columns if c not in numeric_cols]

    date_cols = []
    categorical_cols = []
    for c in non_numeric:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
        except (ValueError, TypeError):
            parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.7:
            date_cols.append(c)
        else:
            categorical_cols.append(c)

    return numeric_cols, categorical_cols, date_cols


# ============================================================
# 3. DATA CLEANING
# ============================================================
def clean_dataset(df, remove_dupes, drop_na, fill_num_method, fill_cat_mode):
    cleaned = df.copy()
    if remove_dupes:
        cleaned = cleaned.drop_duplicates()
    if drop_na:
        cleaned = cleaned.dropna()
    else:
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        cat_cols = [c for c in cleaned.columns if c not in num_cols]
        if fill_num_method == "Mean":
            for c in num_cols:
                cleaned[c] = cleaned[c].fillna(cleaned[c].mean())
        elif fill_num_method == "Median":
            for c in num_cols:
                cleaned[c] = cleaned[c].fillna(cleaned[c].median())
        if fill_cat_mode:
            for c in cat_cols:
                mode_vals = cleaned[c].mode(dropna=True)
                if not mode_vals.empty:
                    cleaned[c] = cleaned[c].fillna(mode_vals.iloc[0])
    return cleaned


def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# ============================================================
# 4. HELPERS
# ============================================================
def safe_col(df, name):
    return name in df.columns


def corr_strength(r):
    r_abs = abs(r)
    if np.isnan(r_abs):
        return "Undefined"
    if r_abs < 0.2:
        return "Very Weak"
    elif r_abs < 0.4:
        return "Weak"
    elif r_abs < 0.6:
        return "Moderate"
    elif r_abs < 0.8:
        return "Strong"
    else:
        return "Very Strong"


def get_first_available(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ============================================================
# 5. KPI DASHBOARD
# ============================================================
def show_kpis(df, numeric_cols):
    st.subheader("📌 Key Performance Indicators")

    sales_col = get_first_available(df, ["Sales", "Total_Sales", "Revenue", "Amount"])
    profit_col = get_first_available(df, ["Profit", "Net_Profit", "Margin"])
    qty_col = get_first_available(df, ["Quantity", "Qty", "Units"])
    rating_col = get_first_available(df, ["Rating", "Rank", "Score"])
    city_col = get_first_available(df, ["City", "Location", "Customer", "Customer_ID"])

    cols = st.columns(4)
    cols[0].metric("Total Orders", f"{len(df):,}")

    if sales_col:
        total_sales = df[sales_col].sum()
        cols[1].metric("Total Sales", f"{total_sales:,.2f}")
        avg_order = df[sales_col].mean()
    else:
        cols[1].metric("Total Sales", "N/A")
        avg_order = None

    if profit_col:
        cols[2].metric("Total Profit", f"{df[profit_col].sum():,.2f}")
    else:
        cols[2].metric("Total Profit", "N/A")

    if qty_col:
        cols[3].metric("Total Quantity", f"{df[qty_col].sum():,.0f}")
    else:
        cols[3].metric("Total Quantity", "N/A")

    cols2 = st.columns(4)
    cols2[0].metric("Avg Order Value", f"{avg_order:,.2f}" if avg_order is not None else "N/A")
    cols2[1].metric("Avg Rating", f"{df[rating_col].mean():,.2f}" if rating_col else "N/A")
    cols2[2].metric("Unique Locations", f"{df[city_col].nunique():,}" if city_col else "N/A")
    cols2[3].metric("Numeric Columns", f"{len(numeric_cols)}")


# ============================================================
# 6. OVERVIEW TAB
# ============================================================
def show_overview(df):
    st.subheader("📋 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Cells", int(df.isna().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))

    with st.expander("🔎 First 5 Rows"):
        st.dataframe(df.head(), use_container_width=True)
    with st.expander("🔎 Last 5 Rows"):
        st.dataframe(df.tail(), use_container_width=True)

    st.markdown("**Data Types**")
    dtype_df = pd.DataFrame({"Column": df.columns, "Data Type": df.dtypes.astype(str).values})
    st.dataframe(dtype_df, use_container_width=True)

    st.markdown("**Missing Value Summary**")
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Column": df.columns, "Missing Values": missing.values,
                                "Missing %": missing_pct.values})
    st.dataframe(missing_df, use_container_width=True)


# ============================================================
# 7. DATA CLEANING TAB
# ============================================================
def data_cleaning_tab(df):
    st.subheader("🧹 Data Cleaning")
    st.caption("Clean the dataset and download the result. Cleaning here does not overwrite the data used in other tabs unless applied.")

    c1, c2 = st.columns(2)
    remove_dupes = c1.checkbox("Remove duplicate rows")
    drop_na = c2.checkbox("Remove rows with missing values")

    fill_num_method = st.selectbox("Fill numeric missing values using:", ["None", "Mean", "Median"], disabled=drop_na)
    fill_cat_mode = st.checkbox("Fill categorical missing values using Mode", disabled=drop_na)

    cleaned = clean_dataset(df, remove_dupes, drop_na, fill_num_method, fill_cat_mode)

    st.markdown("### Before vs After Cleaning")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Before**")
        st.write(f"Rows: {df.shape[0]} | Missing: {int(df.isna().sum().sum())} | Duplicates: {int(df.duplicated().sum())}")
        st.dataframe(df.head(), use_container_width=True)
    with b2:
        st.markdown("**After**")
        st.write(f"Rows: {cleaned.shape[0]} | Missing: {int(cleaned.isna().sum().sum())} | Duplicates: {int(cleaned.duplicated().sum())}")
        st.dataframe(cleaned.head(), use_container_width=True)

    st.download_button("⬇️ Download Cleaned CSV", df_to_csv_bytes(cleaned), "cleaned_dataset.csv", "text/csv")
    return cleaned


# ============================================================
# 8. STATISTICS TAB
# ============================================================
def show_statistics(df, numeric_cols):
    st.subheader("📊 Statistical Analysis")
    if not numeric_cols:
        st.warning("No numerical columns available for statistical analysis.")
        return

    col = st.selectbox("Select a numerical column", numeric_cols)
    series = df[col].dropna()

    if series.empty:
        st.warning("Selected column has no valid values.")
        return

    mode_val = series.mode()
    mode_display = mode_val.iloc[0] if not mode_val.empty else np.nan

    stats = {
        "Count": series.count(), "Mean": series.mean(), "Median": series.median(),
        "Mode": mode_display, "Minimum": series.min(), "Maximum": series.max(),
        "Range": series.max() - series.min(), "Std Dev": series.std(),
        "Variance": series.var(), "Sum": series.sum(),
    }

    cols = st.columns(5)
    keys = list(stats.keys())
    for i, k in enumerate(keys):
        val = stats[k]
        cols[i % 5].metric(k, f"{val:,.2f}" if isinstance(val, (int, float, np.number)) else str(val))

    st.dataframe(pd.DataFrame(stats.items(), columns=["Statistic", "Value"]), use_container_width=True)


# ============================================================
# 9. COMPARATIVE ANALYSIS TAB
# ============================================================
def comparative_analysis(df, categorical_cols, numeric_cols):
    st.subheader("⚖️ Comparative Analysis")
    if not categorical_cols or not numeric_cols:
        st.warning("Need at least one categorical and one numerical column for comparative analysis.")
        return

    c1, c2 = st.columns(2)
    cat_col = c1.selectbox("Category column", categorical_cols, key="cmp_cat")
    num_col = c2.selectbox("Numerical column", numeric_cols, key="cmp_num")

    grouped = df.groupby(cat_col)[num_col].agg(["count", "sum", "mean", "min", "max", "median", "std"]).reset_index()
    grouped.columns = ["Category", "Orders", "Total", "Average", "Minimum", "Maximum", "Median", "Std Dev"]
    grouped = grouped.sort_values("Average", ascending=False)

    st.dataframe(grouped, use_container_width=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(grouped["Category"], grouped["Average"], color="steelblue")
    axes[0].set_title(f"Average {num_col} by {cat_col}")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(grouped["Category"], grouped["Total"], color="seagreen")
    axes[1].set_title(f"Total {num_col} by {cat_col}")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    groups_data = [df[df[cat_col] == cat][num_col].dropna() for cat in grouped["Category"]]
    ax2.boxplot(groups_data, tick_labels=grouped["Category"], vert=True)
    ax2.set_title(f"Distribution of {num_col} across {cat_col}")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    top_cat = grouped.iloc[0]["Category"]
    st.info(f"💡 Among the selected categories, **{top_cat}** has the highest average {num_col}.")

    st.download_button("⬇️ Download Comparison Table", df_to_csv_bytes(grouped), "comparison_table.csv", "text/csv")


# ============================================================
# 10. PRODUCT COMPARISON TAB
# ============================================================
def product_analysis(df, categorical_cols, numeric_cols):
    st.subheader("🛍️ Product Comparison")
    product_col = get_first_available(df, ["Product", "Item", "Product_Name"]) or (categorical_cols[0] if categorical_cols else None)
    if not product_col or not numeric_cols:
        st.warning("Need a product-like column and a numerical column for this analysis.")
        return

    metric = st.selectbox("Select metric", numeric_cols, key="prod_metric")
    grouped = df.groupby(product_col)[metric].agg(["sum", "mean", "count"]).reset_index()
    grouped.columns = ["Product", "Total", "Average", "Orders"]
    grouped = grouped.sort_values("Total", ascending=False)

    top10 = grouped.head(10)
    bottom10 = grouped.tail(10).sort_values("Total")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Top 10 Products by {metric}**")
        st.dataframe(top10, use_container_width=True)
    with c2:
        st.markdown(f"**Bottom 10 Products by {metric}**")
        st.dataframe(bottom10, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(top10["Product"], top10["Total"], color="darkorange")
    ax.set_title(f"Top 10 Products by Total {metric}")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    best = grouped.iloc[0]
    worst = grouped.iloc[-1]
    st.info(f"💡 Highest performing product: **{best['Product']}** ({best['Total']:,.2f}). "
            f"Lowest performing product: **{worst['Product']}** ({worst['Total']:,.2f}).")


# ============================================================
# 11. REGIONAL ANALYSIS TAB
# ============================================================
def regional_analysis(df, categorical_cols, numeric_cols):
    st.subheader("🌍 Regional Analysis")
    geo_candidates = [c for c in ["City", "Region", "State", "Country"] if c in df.columns]
    if not geo_candidates or not numeric_cols:
        st.warning("No location-type column (City/Region/State/Country) found, or no numerical columns available.")
        return

    geo_col = st.selectbox("Compare by", geo_candidates, key="geo_col")
    metric = st.selectbox("Select metric", numeric_cols, key="geo_metric")

    grouped = df.groupby(geo_col).agg(Total=(metric, "sum"), Average=(metric, "mean"), Orders=(metric, "count")).reset_index()
    grouped = grouped.sort_values("Total", ascending=False)

    st.dataframe(grouped, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(grouped[geo_col], grouped["Total"], color="mediumpurple")
    ax.set_title(f"Total {metric} by {geo_col}")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    top = grouped.iloc[0]
    low = grouped.iloc[-1]
    st.info(f"💡 Top performing {geo_col}: **{top[geo_col]}** ({top['Total']:,.2f}). "
            f"Lowest performing {geo_col}: **{low[geo_col]}** ({low['Total']:,.2f}).")


# ============================================================
# 12. CATEGORY + METRIC (FLEXIBLE) COMPARISON
# ============================================================
def flexible_comparison(df, categorical_cols, numeric_cols):
    st.subheader("🧮 Category + Metric Comparison")
    if not categorical_cols or not numeric_cols:
        st.warning("Need at least one categorical and one numerical column.")
        return

    c1, c2, c3 = st.columns(3)
    cat_col = c1.selectbox("Category", categorical_cols, key="flex_cat")
    metric = c2.selectbox("Metric", numeric_cols, key="flex_metric")
    agg = c3.selectbox("Aggregation", ["Mean", "Sum", "Count", "Minimum", "Maximum", "Median"], key="flex_agg")

    agg_map = {"Mean": "mean", "Sum": "sum", "Count": "count", "Minimum": "min", "Maximum": "max", "Median": "median"}
    result = df.groupby(cat_col)[metric].agg(agg_map[agg]).reset_index()
    result.columns = [cat_col, f"{agg} of {metric}"]
    result = result.sort_values(result.columns[1], ascending=False)

    st.dataframe(result, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(result[cat_col], result[result.columns[1]], color="teal")
    ax.set_title(f"{agg} of {metric} by {cat_col}")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    top_row = result.iloc[0]
    st.info(f"💡 **{top_row[cat_col]}** has the highest {agg.lower()} {metric} ({top_row[result.columns[1]]:,.2f}).")


# ============================================================
# 13. MULTI-METRIC COMPARISON
# ============================================================
def multi_metric_comparison(df, categorical_cols, numeric_cols):
    st.subheader("📈 Multi-Metric Comparison")
    if not categorical_cols or len(numeric_cols) < 2:
        st.warning("Need one categorical column and at least two numerical columns.")
        return

    cat_col = st.selectbox("Category", categorical_cols, key="multi_cat")
    metrics = st.multiselect("Select metrics", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))], key="multi_metrics")

    if not metrics:
        st.info("Select at least one metric to compare.")
        return

    grouped = df.groupby(cat_col)[metrics].mean().reset_index()
    st.dataframe(grouped, use_container_width=True)

    normalized = grouped.copy()
    for m in metrics:
        rng = grouped[m].max() - grouped[m].min()
        normalized[m] = (grouped[m] - grouped[m].min()) / rng if rng != 0 else 0.0

    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(grouped))
    width = 0.8 / len(metrics)
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, normalized[m], width, label=m)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(grouped[cat_col], rotation=45, ha="right")
    ax.set_title(f"Normalized Comparison of Metrics by {cat_col}")
    ax.legend()
    st.pyplot(fig)
    st.caption("Values are min-max normalized (0-1) because selected metrics may have different scales.")


# ============================================================
# 14. SALES & PROFIT ANALYSIS
# ============================================================
def sales_profit_analysis(df, categorical_cols):
    st.subheader("💰 Sales & Profit Analysis")
    sales_col = get_first_available(df, ["Sales", "Revenue", "Total_Sales", "Amount"])
    profit_col = get_first_available(df, ["Profit", "Net_Profit", "Margin"])

    if not sales_col or not profit_col:
        st.warning("This analysis needs both a Sales-like and a Profit-like column.")
        return
    if not categorical_cols:
        st.warning("No categorical column available to group by.")
        return

    cat_col = st.selectbox("Group by", categorical_cols, key="sp_cat")
    grouped = df.groupby(cat_col)[[sales_col, profit_col]].sum().reset_index()
    grouped = grouped.sort_values(sales_col, ascending=False)

    st.dataframe(grouped, use_container_width=True)

    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(grouped))
    ax.bar(x - 0.2, grouped[sales_col], width=0.4, label=sales_col, color="steelblue")
    ax.bar(x + 0.2, grouped[profit_col], width=0.4, label=profit_col, color="orangered")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped[cat_col], rotation=45, ha="right")
    ax.legend()
    ax.set_title(f"{sales_col} vs {profit_col} by {cat_col}")
    st.pyplot(fig)

    best = grouped.loc[grouped[profit_col].idxmax()]
    worst = grouped.loc[grouped[profit_col].idxmin()]
    st.info(f"💡 Highest profit: **{best[cat_col]}** ({best[profit_col]:,.2f}). "
            f"Lowest profit: **{worst[cat_col]}** ({worst[profit_col]:,.2f}).")


# ============================================================
# 15. DISCOUNT ANALYSIS
# ============================================================
def discount_analysis(df, numeric_cols):
    st.subheader("🏷️ Discount Analysis")
    discount_col = get_first_available(df, ["Discount", "Discount_Rate", "Discount_Percent"])
    if not discount_col:
        st.warning("No Discount-like column found in this dataset.")
        return

    other_numeric = [c for c in numeric_cols if c != discount_col]
    if not other_numeric:
        st.warning("Need at least one other numerical column to compare against discount.")
        return

    target = st.selectbox("Compare discount against", other_numeric, key="disc_target")

    valid = df[[discount_col, target]].dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(valid[discount_col], valid[target], alpha=0.6, color="crimson")
    ax.set_xlabel(discount_col)
    ax.set_ylabel(target)
    ax.set_title(f"{discount_col} vs {target}")
    st.pyplot(fig)

    if len(valid) > 1 and valid[discount_col].std() > 0 and valid[target].std() > 0:
        r = valid[discount_col].corr(valid[target])
        st.metric(f"Correlation ({discount_col} vs {target})", f"{r:.3f}")
        direction = "positive" if r > 0 else "negative"
        st.info(f"💡 There is a **{corr_strength(r).lower()} {direction}** relationship between "
                f"{discount_col} and {target} (r = {r:.3f}).")
    else:
        st.warning("Not enough variation in the data to calculate a reliable correlation.")


# ============================================================
# 16. CORRELATION ANALYSIS
# ============================================================
def correlation_analysis(df, numeric_cols):
    st.subheader("🔗 Correlation Analysis")
    if len(numeric_cols) < 2:
        st.warning("Need at least two numerical columns for correlation analysis.")
        return

    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(min(1.2 * len(numeric_cols) + 2, 12), min(1.2 * len(numeric_cols) + 2, 10)))
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    st.markdown("### Pairwise Correlation")
    c1, c2 = st.columns(2)
    col_a = c1.selectbox("Column A", numeric_cols, key="corr_a")
    col_b = c2.selectbox("Column B", numeric_cols, index=min(1, len(numeric_cols) - 1), key="corr_b")

    if col_a == col_b:
        st.info("Select two different columns to calculate correlation.")
    else:
        r = df[col_a].corr(df[col_b])
        if np.isnan(r):
            st.warning("Correlation could not be calculated for the selected columns.")
        else:
            st.metric(f"Correlation: {col_a} vs {col_b}", f"{r:.3f}")
            direction = "positive" if r > 0 else "negative"
            st.info(f"💡 This is a **{corr_strength(r).lower()} {direction}** relationship.")


# ============================================================
# 17. RANKING ANALYSIS
# ============================================================
def ranking_analysis(df, numeric_cols, categorical_cols):
    st.subheader("🏆 Ranking Analysis")
    if not numeric_cols:
        st.warning("No numerical columns available for ranking.")
        return

    metric = st.selectbox("Metric", numeric_cols, key="rank_metric")
    label_col = get_first_available(df, ["Product", "City", "Region"]) or (categorical_cols[0] if categorical_cols else None)

    n = st.slider("Select Top / Bottom N", 3, 20, 5)

    if label_col:
        base = df.groupby(label_col)[metric].sum().reset_index().sort_values(metric, ascending=False)
    else:
        base = df[[metric]].reset_index().rename(columns={"index": "Row"}).sort_values(metric, ascending=False)
        label_col = "Row"

    top_n = base.head(n)
    bottom_n = base.tail(n).sort_values(metric)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Top {n}**")
        st.dataframe(top_n, use_container_width=True)
    with c2:
        st.markdown(f"**Bottom {n}**")
        st.dataframe(bottom_n, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(top_n[label_col].astype(str), top_n[metric], color="darkgreen")
    ax.set_title(f"Top {n} by {metric}")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)


# ============================================================
# 18. TIME-BASED ANALYSIS
# ============================================================
def time_analysis(df, date_cols, numeric_cols):
    st.subheader("📅 Time Analysis")
    if not date_cols:
        st.info("No date column detected in this dataset — Time Analysis is unavailable.")
        return

    date_col = st.selectbox("Date column", date_cols, key="time_date")
    freq_label = st.radio("Aggregate by", ["Daily", "Monthly", "Yearly"], horizontal=True)
    freq_map = {"Daily": "D", "Monthly": "MS", "Yearly": "YS"}

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])

    if temp.empty:
        st.warning("Could not parse valid dates from this column.")
        return

    sales_col = get_first_available(temp, ["Sales", "Revenue", "Total_Sales", "Amount"])
    profit_col = get_first_available(temp, ["Profit", "Net_Profit", "Margin"])

    grouped = temp.set_index(date_col).resample(freq_map[freq_label])
    agg_dict = {"Orders": (temp.columns[0], "count")}
    result = grouped.size().reset_index(name="Orders")

    if sales_col:
        sales_series = temp.set_index(date_col)[sales_col].resample(freq_map[freq_label]).sum().reset_index()
        result = result.merge(sales_series, on=date_col, how="left")
    if profit_col:
        profit_series = temp.set_index(date_col)[profit_col].resample(freq_map[freq_label]).sum().reset_index()
        result = result.merge(profit_series, on=date_col, how="left")

    st.dataframe(result, use_container_width=True)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(result[date_col], result["Orders"], marker="o", label="Orders")
    if sales_col:
        ax2 = ax.twinx()
        ax2.plot(result[date_col], result[sales_col], marker="s", color="orangered", label=sales_col)
        ax2.set_ylabel(sales_col)
    ax.set_ylabel("Orders")
    ax.set_title(f"Trend over Time ({freq_label})")
    plt.xticks(rotation=45)
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    st.pyplot(fig)


# ============================================================
# 19. VISUALIZATION CENTER
# ============================================================
def create_visualization(df, numeric_cols, categorical_cols, date_cols):
    st.subheader("📈 Visualization Center")
    chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Pie", "Scatter", "Histogram", "Box Plot"])
    all_cols = list(df.columns)

    try:
        if chart_type in ["Bar", "Line", "Pie"]:
            x_col = st.selectbox("Category (X-axis)", categorical_cols or all_cols, key="viz_x")
            y_col = st.selectbox("Value (Y-axis)", numeric_cols, key="viz_y") if numeric_cols else None
            if y_col is None:
                st.warning("No numerical column available for the Y-axis.")
                return
            grouped = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(10, 5))
            if chart_type == "Bar":
                ax.bar(grouped.index.astype(str), grouped.values, color="cornflowerblue")
                plt.xticks(rotation=45, ha="right")
            elif chart_type == "Line":
                ax.plot(grouped.index.astype(str), grouped.values, marker="o", color="seagreen")
                plt.xticks(rotation=45, ha="right")
            else:  # Pie
                ax.pie(grouped.values, labels=grouped.index.astype(str), autopct="%1.1f%%")
                ax.axis("equal")
            ax.set_title(f"{y_col} by {x_col}")
            st.pyplot(fig)

        elif chart_type == "Scatter":
            if len(numeric_cols) < 2:
                st.warning("Need at least two numerical columns for a scatter plot.")
                return
            x_col = st.selectbox("X-axis", numeric_cols, key="scat_x")
            y_col = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols) - 1), key="scat_y")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(df[x_col], df[y_col], alpha=0.6, color="purple")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{x_col} vs {y_col}")
            st.pyplot(fig)

        elif chart_type == "Histogram":
            if not numeric_cols:
                st.warning("No numerical columns available.")
                return
            col = st.selectbox("Column", numeric_cols, key="hist_col")
            bins = st.slider("Bins", 5, 50, 20)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df[col].dropna(), bins=bins, color="teal", edgecolor="black")
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

        elif chart_type == "Box Plot":
            if not numeric_cols:
                st.warning("No numerical columns available.")
                return
            col = st.selectbox("Column", numeric_cols, key="box_col")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.boxplot(df[col].dropna())
            ax.set_title(f"Box Plot of {col}")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Could not generate the chart with the selected options. Error: {e}")


# ============================================================
# 20. AUTOMATIC INSIGHTS
# ============================================================
def generate_insights(df, numeric_cols, categorical_cols, date_cols):
    st.subheader("🤖 Automatic Business Insights")
    insights = []

    sales_col = get_first_available(df, ["Sales", "Revenue", "Total_Sales", "Amount"])
    profit_col = get_first_available(df, ["Profit", "Net_Profit", "Margin"])
    rating_col = get_first_available(df, ["Rating", "Score"])
    payment_col = get_first_available(df, ["Payment_Method", "Payment"])
    city_col = get_first_available(df, ["City", "Region", "Location"])
    product_col = get_first_available(df, ["Product", "Item"])
    category_col = get_first_available(df, ["Category", "Product_Category"])

    if sales_col:
        insights.append(f"Total Sales across the dataset is **{df[sales_col].sum():,.2f}**.")
        insights.append(f"Average order value is **{df[sales_col].mean():,.2f}**.")
        if category_col:
            top_cat = df.groupby(category_col)[sales_col].sum().idxmax()
            insights.append(f"**{top_cat}** is the category with the highest total Sales.")

    if profit_col:
        insights.append(f"Total Profit across the dataset is **{df[profit_col].sum():,.2f}**.")
        if category_col:
            top_profit_cat = df.groupby(category_col)[profit_col].sum().idxmax()
            low_profit_cat = df.groupby(category_col)[profit_col].sum().idxmin()
            insights.append(f"**{top_profit_cat}** generates the highest total Profit, while "
                             f"**{low_profit_cat}** generates the lowest.")

    if product_col and sales_col:
        top_product = df.groupby(product_col)[sales_col].sum().idxmax()
        insights.append(f"The top-selling product by total Sales is **{top_product}**.")

    if city_col and sales_col:
        top_city = df.groupby(city_col)[sales_col].sum().idxmax()
        insights.append(f"**{top_city}** is the top-performing location by total Sales.")

    if payment_col:
        common_pay = df[payment_col].mode()
        if not common_pay.empty:
            insights.append(f"The most common payment method is **{common_pay.iloc[0]}**.")

    if rating_col:
        best_rating_idx = df[rating_col].idxmax() if not df[rating_col].isna().all() else None
        if best_rating_idx is not None and product_col:
            insights.append(f"Highest rated entry belongs to **{df.loc[best_rating_idx, product_col]}** "
                             f"with a rating of **{df.loc[best_rating_idx, rating_col]}**.")

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)
        if corr.values.max() > 0:
            idx = np.unravel_index(np.argmax(corr.values), corr.shape)
            col1, col2 = numeric_cols[idx[0]], numeric_cols[idx[1]]
            actual_r = df[col1].corr(df[col2])
            insights.append(f"The strongest numerical relationship is between **{col1}** and **{col2}** "
                             f"(r = {actual_r:.2f}, {corr_strength(actual_r).lower()}).")

    missing_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    insights.append(f"Overall missing data in this dataset is **{missing_pct:.2f}%**.")

    if not insights:
        st.warning("Not enough recognizable columns to generate automatic insights for this dataset.")
        return

    for point in insights:
        st.markdown(f"- {point}")


# ============================================================
# 21. SIDEBAR FILTERS
# ============================================================
def apply_filters(df, categorical_cols, date_cols):
    st.sidebar.header("🔍 Filters")
    filtered = df.copy()

    filterable = [c for c in ["Category", "City", "Region", "Customer_Type", "Payment_Method"] if c in df.columns]
    filterable += [c for c in categorical_cols if c not in filterable][:3]
    filterable = list(dict.fromkeys(filterable))[:6]

    for col in filterable:
        options = sorted(df[col].dropna().unique().tolist())
        selected = st.sidebar.multiselect(f"{col}", options, default=[])
        if selected:
            filtered = filtered[filtered[col].isin(selected)]

    if date_cols:
        date_col = date_cols[0]
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        valid_dates = parsed.dropna()
        if not valid_dates.empty:
            min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
            date_range = st.sidebar.date_input(f"{date_col} range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
            if isinstance(date_range, tuple) and len(date_range) == 2:
                parsed_full = pd.to_datetime(filtered[date_col], errors="coerce")
                mask = (parsed_full.dt.date >= date_range[0]) & (parsed_full.dt.date <= date_range[1])
                filtered = filtered[mask.fillna(False)]

    st.sidebar.markdown("---")
    st.sidebar.info(f"Filtered Records: **{len(filtered)}** / {len(df)}")
    return filtered


# ============================================================
# MAIN APP
# ============================================================
def main():
    st.title("🛒 E-Commerce Sales Data Analyzer")
    st.markdown("##### Interactive Data Analysis, Comparison and Business Insights Dashboard")

    with st.expander("ℹ️ About this Project", expanded=False):
        st.write(
            "This application lets you upload any compatible CSV dataset (or use the built-in "
            "E-Commerce sample dataset) to clean data, explore statistics, compare categories, "
            "study relationships between variables, and generate automatic business insights. "
            "All calculations are performed dynamically from the loaded dataset — nothing is hard-coded."
        )

    # ---------------- Data Source ----------------
    st.sidebar.header("📂 Data Source")
    source = st.sidebar.radio("Choose data source", ["Use E-Commerce Sample Dataset", "Upload my own CSV"])

    if source == "Upload my own CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            df = load_dataset(uploaded_file)
            if df is None:
                st.stop()
            st.sidebar.success(f"Loaded: {uploaded_file.name}")
        else:
            st.info("👈 Upload a CSV file from the sidebar, or switch to the sample dataset, to get started.")
            st.stop()
    else:
        df = generate_sample_dataset()
        st.sidebar.success("Using E-Commerce Sample Dataset")

    st.sidebar.markdown(f"**Rows:** {df.shape[0]}  \n**Columns:** {df.shape[1]}")

    numeric_cols, categorical_cols, date_cols = detect_column_types(df)

    filtered_df = apply_filters(df, categorical_cols, date_cols)
    if filtered_df.empty:
        st.warning("No records match the selected filters. Showing unfiltered data instead.")
        filtered_df = df

    tab_names = [
        "🏠 Dashboard", "📋 Overview", "🧹 Data Cleaning", "📊 Statistics",
        "⚖️ Comparative Analysis", "🛍️ Product Analysis", "🌍 Regional Analysis",
        "💰 Sales & Profit", "🏷️ Discount", "🔗 Correlation", "🏆 Ranking",
        "📅 Time Analysis", "📈 Visualization", "🤖 Automatic Insights",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        show_kpis(filtered_df, numeric_cols)
        st.markdown("---")
        st.markdown("### 🧮 Category + Metric Comparison")
        flexible_comparison(filtered_df, categorical_cols, numeric_cols)
        st.markdown("---")
        multi_metric_comparison(filtered_df, categorical_cols, numeric_cols)

    with tabs[1]:
        show_overview(filtered_df)

    with tabs[2]:
        data_cleaning_tab(filtered_df)

    with tabs[3]:
        show_statistics(filtered_df, numeric_cols)

    with tabs[4]:
        comparative_analysis(filtered_df, categorical_cols, numeric_cols)

    with tabs[5]:
        product_analysis(filtered_df, categorical_cols, numeric_cols)

    with tabs[6]:
        regional_analysis(filtered_df, categorical_cols, numeric_cols)

    with tabs[7]:
        sales_profit_analysis(filtered_df, categorical_cols)

    with tabs[8]:
        discount_analysis(filtered_df, numeric_cols)

    with tabs[9]:
        correlation_analysis(filtered_df, numeric_cols)

    with tabs[10]:
        ranking_analysis(filtered_df, numeric_cols, categorical_cols)

    with tabs[11]:
        time_analysis(filtered_df, date_cols, numeric_cols)

    with tabs[12]:
        create_visualization(filtered_df, numeric_cols, categorical_cols, date_cols)

    with tabs[13]:
        generate_insights(filtered_df, numeric_cols, categorical_cols, date_cols)


if __name__ == "__main__":
    main()
