# ecommerce_model_and_dashboard.py
# AI-Integrated Strategic Management Decision Support Tool
# Primary Focus: Amazon India | Comparison: Flipkart & Meesho

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import plotly.express as px



# -------------------------------
# 1) APP CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI-Integrated E-Commerce Strategy Dashboard",
    layout="wide",
)

st.title("AI-Integrated E-Commerce Strategy Dashboard")
st.caption("Primary Focus: Amazon India | Benchmark Comparison: Flipkart & Meesho")


# -------------------------------
# 2) LOAD DATA
# -------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Order_Date" in df.columns:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    return df

df = load_data("ecommerce_data.csv")


# -------------------------------
# 3) VALIDATION (simple + faculty-friendly)
# -------------------------------
required_cols = [
    "Customer_ID", "Platform",
    "Recency", "Frequency", "Monetary",
    "Discount_Use", "Category_Diversity", "Return_Rate",
    "Location_Tier", "Average_Order_Value", "Margin_Percent",
    "Expected_Active_Months", "Churn_Flag"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Dataset is missing required columns: {missing}")
    st.stop()


# -------------------------------
# 4) FEATURE ENGINEERING (Financial Analysis + Strategy Inputs)
# -------------------------------
# Financial analysis fields
df["Revenue"] = df["Frequency"] * df["Average_Order_Value"]
df["Contribution"] = df["Revenue"] * df["Margin_Percent"]
df["CLV"] = df["Contribution"] * df["Expected_Active_Months"]

# Strategy-related numeric metric
df["Discount_Dependency"] = df["Discount_Use"] * df["Frequency"]


# -------------------------------
# 5) CHURN MODELLING (Data Analytics)
# -------------------------------
features_for_churn = ["Recency", "Frequency", "Monetary", "Discount_Use", "Return_Rate"]
df_churn = df.dropna(subset=features_for_churn + ["Churn_Flag"]).copy()

if df_churn["Churn_Flag"].nunique() > 1 and len(df_churn) >= 10:
    X = df_churn[features_for_churn]
    y = df_churn["Churn_Flag"]
    churn_model = LogisticRegression(max_iter=2000)
    churn_model.fit(X, y)
    df_churn["Churn_Prob"] = churn_model.predict_proba(X)[:, 1]
else:
    # fallback for small demo datasets
    df_churn["Churn_Prob"] = np.clip(
        0.15 + 0.01 * (df_churn["Recency"] / (df_churn["Recency"].max() + 1)),
        0.05, 0.60
    )

df = df.merge(df_churn[["Customer_ID", "Churn_Prob"]], on="Customer_ID", how="left")


# -------------------------------
# 6) SEGMENTATION (Data Analytics)
# -------------------------------
cluster_features = ["CLV", "Discount_Dependency", "Category_Diversity", "Churn_Prob"]
df_seg = df.dropna(subset=cluster_features).copy()

if len(df_seg) >= 8:
    scaler = StandardScaler()
    Xc = scaler.fit_transform(df_seg[cluster_features])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_seg["Cluster"] = kmeans.fit_predict(Xc)

    # Create simple, business-friendly names using centroid ranking logic
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=cluster_features
    )
    # Rank clusters by CLV high->low and churn low->high to label
    centroids["CLV_rank"] = centroids["CLV"].rank(ascending=False, method="dense")
    centroids["Churn_rank"] = centroids["Churn_Prob"].rank(ascending=True, method="dense")

    # Heuristic naming
    names = {}
    for idx, row in centroids.iterrows():
        if row["CLV_rank"] == 1 and row["Churn_rank"] <= 2:
            names[idx] = "High-Value Loyal"
        elif row["Discount_Dependency"] >= centroids["Discount_Dependency"].median() and row["CLV_rank"] >= 3:
            names[idx] = "Deal-Driven / Price Sensitive"
        elif row["Churn_rank"] >= 3 and row["CLV_rank"] >= 3:
            names[idx] = "At-Risk / Low Value"
        else:
            names[idx] = "Occasional / Mid Value"

    df_seg["Cluster_Name"] = df_seg["Cluster"].map(names)
    df = df.merge(df_seg[["Customer_ID", "Cluster", "Cluster_Name"]], on="Customer_ID", how="left")
else:
    df["Cluster_Name"] = np.nan


# -------------------------------
# 7) SIDEBAR FILTERS (Dynamic Dashboard)
# -------------------------------
st.sidebar.header("Filters")
platforms = ["All"] + sorted(df["Platform"].unique().tolist())
sel_platform = st.sidebar.selectbox("Platform", platforms)

tiers = ["All"] + sorted(df["Location_Tier"].astype(str).unique().tolist())
sel_tier = st.sidebar.selectbox("Location Tier", tiers)

df_view = df.copy()
if sel_platform != "All":
    df_view = df_view[df_view["Platform"] == sel_platform]
if sel_tier != "All":
    df_view = df_view[df_view["Location_Tier"].astype(str) == sel_tier]


# -------------------------------
# 8) TOP KPIs (Static + Screenshot-friendly)
# -------------------------------
st.markdown("## Executive KPIs (Platform View)")

def kpis_for(platform: str) -> dict:
    d = df_view[df_view["Platform"] == platform]
    if len(d) == 0:
        return {"clv": np.nan, "churn": np.nan, "disc": np.nan}
    return {
        "clv": d["CLV"].mean(),
        "churn": d["Churn_Prob"].mean(),
        "disc": d["Discount_Dependency"].mean(),
    }

kpi_cols = st.columns(3)
for idx, p in enumerate(["Amazon", "Flipkart", "Meesho"]):
    vals = kpis_for(p)
    with kpi_cols[idx]:
        st.markdown(f"### {p}")
        st.metric("Avg CLV", "N/A" if pd.isna(vals["clv"]) else f"{vals['clv']:,.0f}")
        st.metric("Avg Churn Probability", "N/A" if pd.isna(vals["churn"]) else f"{vals['churn']:.2f}")
        st.metric("Avg Discount Dependency", "N/A" if pd.isna(vals["disc"]) else f"{vals['disc']:.2f}")


# -------------------------------
# 9) DASHBOARD SECTIONS (Tabs)
# -------------------------------
tab_strategy, tab_analytics, tab_groups, tab_forecast, tab_scenarios, tab_data = st.tabs(
    ["Strategy Frameworks", "Analytics (Python)", "Strategic Groups", "Trend Forecasting", "Scenario Simulation", "Data View"]
)

# ======================================================
# TAB 1: STRATEGY FRAMEWORKS (Explicit framework placement)
# ======================================================
with tab_strategy:
    st.markdown("## Strategic Frameworks (Applied Explicitly)")
    st.info(
        "This section explicitly applies the required frameworks: "
        "**PESTEL, Porter’s Five Forces, VRIO, Value Chain, Strategic Groups**."
    )

    st.markdown("### PESTEL (Macro-environment for Indian E-Commerce)")
    st.write(
        """
- **Political:** Digital India, UPI, ONDC ecosystem; compliance requirements for marketplaces.  
- **Economic:** Price sensitivity; festival-sale cycles; growth of Tier-2/3 demand.  
- **Social:** Mobile-first behaviour; convenience-led adoption; trust & service expectations.  
- **Technological:** AI recommendations/search; logistics automation; payments and fraud prevention.  
- **Environmental:** Packaging waste concerns; sustainability pressure on deliveries.  
- **Legal:** Consumer protection, refunds/returns norms, data/privacy compliance.
"""
    )

    st.markdown("### Porter’s Five Forces (Industry Competitiveness)")
    st.write(
        """
- **Rivalry:** High—Amazon vs Flipkart vs Meesho (plus JioMart & vertical players).  
- **Buyer Power:** High—low switching cost and offer-driven behaviour.  
- **Supplier Power:** Moderate—sellers multi-home across platforms; logistics partners have choices.  
- **Threat of Substitutes:** Moderate—offline retail, brand websites (D2C).  
- **Threat of New Entrants:** Moderate—tech is easy but logistics scale is difficult.
"""
    )

    st.markdown("### VRIO (Amazon India — internal advantage check)")
    st.write(
        """
- **Logistics & fulfilment network:** Valuable / Rare / Difficult to imitate / Organised → sustained edge.  
- **Prime ecosystem:** Valuable / Rare / Hard to replicate / Organised → loyalty & retention lever.  
- **Data + analytics capability:** Valuable / partially rare / improving defensibility → supports personalization.  
- **Brand trust + service:** Valuable / difficult to match consistently → supports premium positioning.
"""
    )

    st.markdown("### Value Chain (Amazon India — simplified)")
    st.write(
        """
- **Inbound logistics:** Seller onboarding, fulfilment centre network, inventory placement.  
- **Operations:** Warehouse automation, order processing, forecasting and routing.  
- **Outbound logistics:** Strong last-mile delivery and reliability.  
- **Marketing & sales:** Personalised recommendations, Prime bundling, targeted campaigns.  
- **Service:** Returns/refunds, customer support, trust-building.
"""
    )

    st.markdown("### Strategic Groups (Where each player sits)")
    st.write(
        """
- Strategic Groups are reflected quantitatively in the **Strategic Group Map** tab using **CLV vs Discount Dependency**.  
- Interpretation:
  - **Amazon:** higher value & lower discount dependency (premium + loyalty-led).  
  - **Flipkart:** mid-zone (category-led + promotional cycles).  
  - **Meesho:** low value & high discount dependency (price-led, volume-driven).
"""
    )


# ======================================================
# TAB 2: ANALYTICS (Python) – Financial analysis, churn, segmentation
# ======================================================
with tab_analytics:
    st.markdown("## Data Analytics (Python) — Financial Analysis, Churn, Segmentation")

    st.markdown("### Financial Analysis (Revenue, Contribution, CLV)")
    fin_summary = df_view.groupby("Platform").agg(
        Avg_Revenue=("Revenue", "mean"),
        Avg_Contribution=("Contribution", "mean"),
        Avg_CLV=("CLV", "mean")
    ).reset_index()
    st.dataframe(fin_summary, use_container_width=True)

    fig_clv = px.bar(fin_summary, x="Platform", y="Avg_CLV", title="Average CLV by Platform", text_auto=".0f")
    st.plotly_chart(fig_clv, use_container_width=True)

    st.markdown("### Churn Modelling (Logistic Regression → Churn Probability)")
    churn_summary = df_view.groupby("Platform")["Churn_Prob"].mean().reset_index()
    fig_churn = px.bar(churn_summary, x="Platform", y="Churn_Prob", title="Average Churn Probability by Platform", text_auto=".2f")
    st.plotly_chart(fig_churn, use_container_width=True)

    st.markdown("### Customer Segmentation (K-Means)")
    if df_view["Cluster_Name"].notna().any():
        seg_summary = df_view.groupby(["Platform", "Cluster_Name"])["Customer_ID"].count().reset_index(name="Count")
        fig_seg = px.bar(
            seg_summary, x="Platform", y="Count", color="Cluster_Name",
            barmode="stack", title="Segment Mix by Platform"
        )
        st.plotly_chart(fig_seg, use_container_width=True)
    else:
        st.warning("Segmentation could not be generated (dataset too small or missing values).")


# ======================================================
# TAB 3: STRATEGIC GROUPS – Quantitative mapping
# ======================================================
with tab_groups:
    st.markdown("## Strategic Group Map (Quantitative Strategic Groups)")

    st.markdown("### CLV vs Discount Dependency (Strategic Positioning)")
    fig_sg = px.scatter(
        df_view,
        x="Discount_Dependency",
        y="CLV",
        color="Platform",
        size="Frequency",
        hover_data=["Customer_ID", "Location_Tier"],
        title="Strategic Group Map: Customer Value vs Discount Sensitivity"
    )
    st.plotly_chart(fig_sg, use_container_width=True)

    st.markdown("### Platform-Level Positioning Summary (Mean Values)")
    sg_summary = df_view.groupby("Platform").agg(
        Avg_CLV=("CLV", "mean"),
        Avg_Discount_Dependency=("Discount_Dependency", "mean"),
        Avg_Churn=("Churn_Prob", "mean")
    ).reset_index()
    st.dataframe(sg_summary, use_container_width=True)


# ======================================================
# TAB 4: TREND FORECASTING – Coded forecast
# ======================================================
with tab_forecast:
    st.markdown("## Trend Forecasting (Python)")

    if "Order_Date" not in df_view.columns or df_view["Order_Date"].isna().all():
        st.warning("Order_Date is missing or invalid. Trend forecasting requires valid dates.")
    else:
        temp = df_view.dropna(subset=["Order_Date"]).copy()
        temp["Month"] = temp["Order_Date"].dt.to_period("M").astype(str)

        monthly = temp.groupby(["Platform", "Month"])["Customer_ID"].count().reset_index(name="Orders")
        st.dataframe(monthly, use_container_width=True)

        # Forecast for selected platform (or Amazon by default)
        forecast_platform = st.selectbox("Forecast Platform", ["Amazon", "Flipkart", "Meesho"], index=0)
        series = monthly[monthly["Platform"] == forecast_platform].copy()

        if len(series) < 3:
            st.info("Not enough monthly points to forecast. Add more dates/rows for stronger forecasting.")
        else:
            X = np.arange(len(series)).reshape(-1, 1)
            y = series["Orders"].values

            lr = LinearRegression()
            lr.fit(X, y)
            series["Forecast"] = lr.predict(X)

            fig_trend = px.line(
                series,
                x="Month",
                y=["Orders", "Forecast"],
                title=f"{forecast_platform} – Orders Trend & Simple Forecast"
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            st.caption("Forecast uses a simple linear model for academic illustration. More advanced models can be used if required.")


# ======================================================
# TAB 5: SCENARIO SIMULATION – Coded scenario generation
# ======================================================
with tab_scenarios:
    st.markdown("## Scenario Generation (Python) — Decision Support")

    st.markdown("### Scenario 1: Retention Improvement (Amazon → Total CLV Uplift)")
    retention_improve = st.slider(
        "Assumed % improvement in retention for Amazon high-value customers",
        min_value=0, max_value=20, step=1, value=5
    )

    amazon_df = df_view[df_view["Platform"] == "Amazon"].copy()
    if len(amazon_df) == 0:
        st.warning("No Amazon data available under current filters.")
    else:
        base_total_clv = amazon_df["CLV"].sum()
        uplift_factor = 1 + (retention_improve / 100.0) * 0.5  # simple academic assumption
        new_total_clv = base_total_clv * uplift_factor

        c1, c2 = st.columns(2)
        c1.metric("Current Total CLV (Amazon)", f"{base_total_clv:,.0f}")
        c2.metric("Simulated Total CLV (Retention Scenario)", f"{new_total_clv:,.0f}")

    st.markdown("---")

    st.markdown("### Scenario 2: Discount Reduction (Amazon → Margin vs Volume Trade-off)")
    discount_reduction = st.slider(
        "Assumed % reduction in discount usage for Amazon",
        min_value=0, max_value=30, step=5, value=10
    )

    if len(amazon_df) > 0:
        base_total_clv2 = amazon_df["CLV"].sum()

        # Assumptions (faculty-friendly, not over-technical):
        margin_uplift = 1 + (discount_reduction / 100.0) * 0.4
        frequency_drop = 1 - (discount_reduction / 100.0) * 0.1

        scenario_clv = base_total_clv2 * margin_uplift * frequency_drop

        c3, c4 = st.columns(2)
        c3.metric("Current Total CLV (Amazon)", f"{base_total_clv2:,.0f}")
        c4.metric("Simulated Total CLV (Discount Scenario)", f"{scenario_clv:,.0f}")

        st.caption("This scenario illustrates that reducing discount dependency can improve profitability even if orders soften slightly.")

    st.markdown("---")

    st.markdown("### Scenario 3: Strategic Group Shift (Amazon Bharat Model – Illustrative)")
    shift_discount = st.slider(
        "Assumed increase in discount dependency (Bharat model)",
        min_value=0.0, max_value=1.0, step=0.05, value=0.20
    )
    shift_clv = st.slider(
        "Assumed CLV change (₹) due to low-cost model",
        min_value=-3000, max_value=0, step=500, value=-1500
    )

    if len(amazon_df) > 0:
        scen = amazon_df.copy()
        scen["Scenario_Discount_Dependency"] = scen["Discount_Dependency"] + shift_discount
        scen["Scenario_CLV"] = scen["CLV"] + shift_clv

        fig_shift = px.scatter(
            scen,
            x="Scenario_Discount_Dependency",
            y="Scenario_CLV",
            title="Amazon Strategic Group Shift – Bharat Model Scenario",
            hover_data=["Customer_ID", "Location_Tier"]
        )
        st.plotly_chart(fig_shift, use_container_width=True)

        st.caption("This scenario helps evaluate how Amazon’s positioning may shift on the Strategic Group Map under a low-cost Bharat-focused model.")


# ======================================================
# TAB 6: DATA VIEW – For annexure screenshots
# ======================================================
with tab_data:
    st.markdown("## Data View (for validation & annexure snapshots)")
    st.dataframe(df_view, use_container_width=True)

    st.download_button(
        label="Download filtered dataset (CSV)",
        data=df_view.to_csv(index=False).encode("utf-8"),
        file_name="filtered_ecommerce_data.csv",
        mime="text/csv"
    )


st.markdown("---")
st.caption(
    "Guideline alignment: Strategic Frameworks (PESTEL, Five Forces, VRIO, Value Chain, Strategic Groups) "
    "+ Data Analytics (Financial analysis, Strategic group mapping, Trend forecasting) "
    "+ Decision Support Tool (interactive dashboard with scenarios)."
)
