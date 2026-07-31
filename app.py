import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from huggingface_hub import InferenceClient

# ------------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Handeerhalk - Inventory & Delivery Engine",
    page_icon="📦",
    layout="wide"
)

st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: #1E293B !important;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #334155;
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #38BDF8 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. Setup HuggingFace Inference Client
# ------------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN", "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")
try:
    client = InferenceClient(api_key=HF_TOKEN) if HF_TOKEN else None
except Exception:
    client = None

# ------------------------------------------------------------------------------
# 3. Safe Load Artifacts
# ------------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists("model.pkl"):
        return None, "File 'model.pkl' not found in current directory."
    try:
        with open("model.pkl", "rb") as f:
            artifacts = pickle.load(f)
        return artifacts, None
    except Exception as e:
        return None, str(e)

artifacts, err = load_artifacts()

if err:
    st.error(f"⚠️ Error loading model.pkl: {err}")
    st.info("Make sure 'model.pkl' exists in the same directory as 'app.py'")
    st.stop()

# Extract from pickle
category_season_shares = artifacts.get("category_season_shares", pd.DataFrame())
summary_stats = artifacts.get("summary_stats", {})
state_delay_rates = artifacts.get("state_delay_rates", pd.DataFrame())
global_delay_mean = artifacts.get("global_delay_mean", 0.1)

# ------------------------------------------------------------------------------
# 4. Sidebar Navigation
# ------------------------------------------------------------------------------
st.sidebar.title("Handeerhalk Engine")
st.sidebar.caption("Smart E-Commerce & Inventory Optimization Engine")
menu_option = st.sidebar.radio("MENU", ["Main Dashboard", "Prediction & Stock Allocation", "Raw Data Details"])

# ------------------------------------------------------------------------------
# Tab 1: Main Dashboard
# ------------------------------------------------------------------------------
if menu_option == "Main Dashboard":
    st.title("Handeerhalk - E-Commerce Analytics")
    st.caption("Supply Chain & Sales Forecasting Engine")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Delivered Orders", f"{summary_stats.get('total_orders', 0):,}")
    col2.metric("Late Delivery Rate", f"{summary_stats.get('late_rate', 0):.2%}")
    col3.metric("Avg Freight Value", f"R$ {summary_stats.get('avg_freight', 0):.2f}")
    col4.metric("Avg Shipping Distance", f"{summary_stats.get('avg_distance', 0):.1f} km")

    st.markdown("---")
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Regional Delay Rates by Customer State")
        if not state_delay_rates.empty:
            state_delay_sorted = state_delay_rates.sort_values(by="customer_state_delay_rate", ascending=False)
            fig = px.bar(
                state_delay_sorted, 
                x="customer_state", 
                y="customer_state_delay_rate", 
                labels={"customer_state": "State", "customer_state_delay_rate": "Delay Rate"},
                color="customer_state_delay_rate", 
                color_continuous_scale="Greens"
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("🤖 Quick AI Logistics Assistant")
        user_query = st.text_input("Ask AI Assistant:", "How can we reduce delay rates in distant states?")
        if st.button("Ask AI"):
            if client:
                with st.spinner("LLM Generating Answer..."):
                    try:
                        response = client.chat_completion(
                            model="Qwen/Qwen2.5-72B-Instruct",
                            messages=[{"role": "user", "content": user_query}],
                            max_tokens=200
                        )
                        st.success(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"HF API Error: {e}")
            else:
                st.warning("HuggingFace Client is not initialized.")

# ------------------------------------------------------------------------------
# Tab 2: Prediction & Regional Stock Allocation
# ------------------------------------------------------------------------------
elif menu_option == "Prediction & Stock Allocation":
    st.title("Handeerhalk - Inventory & Sales Forecasting Engine")
    st.subheader("Select Parameters to Forecast Sales & Recommended Warehouse Stock Mix")

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        available_states = summary_stats.get("available_states", ["SP", "RJ", "MG"])
        selected_state = st.selectbox("Select State / Region:", available_states)
    with col_sel2:
        selected_season_label = st.selectbox(
            "Select Season / Period:", 
            ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec / Black Friday)"]
        )
        selected_q = selected_season_label.split()[0]

    st.markdown("---")

    if st.button("Run Prediction & Stock Optimization", type="primary"):
        col_res1, col_res2 = st.columns([1.2, 1])

        # Filter Stock Allocation Matrix from artifacts
        allocation_data = pd.DataFrame()
        if not category_season_shares.empty:
            allocation_data = category_season_shares[
                (category_season_shares['customer_state'] == selected_state) & 
                (category_season_shares['season'] == selected_q)
            ].head(7)

        with col_res1:
            st.subheader(f"Forecast for `{selected_state}` in `{selected_season_label}`")
            
            matched_state = state_delay_rates[state_delay_rates["customer_state"] == selected_state] if not state_delay_rates.empty else pd.DataFrame()
            avg_delay_risk = (matched_state["customer_state_delay_rate"].values[0] if not matched_state.empty else global_delay_mean) * 100
            
            if not category_season_shares.empty:
                state_season_units = category_season_shares[
                    (category_season_shares['customer_state'] == selected_state) & 
                    (category_season_shares['season'] == selected_q)
                ]['category_units'].sum()
                predicted_volume = int(state_season_units) if state_season_units > 0 else 1200
            else:
                predicted_volume = 1200

            m1, m2 = st.columns(2)
            m1.metric("Predicted Order Volume", f"{predicted_volume:,} Units")
            m2.metric("Estimated Delay Risk Rate", f"{avg_delay_risk:.2f}%")

            st.markdown("### Recommended Warehouse Stock Mix (%)")
            if not allocation_data.empty:
                fig_pie = px.pie(
                    allocation_data, 
                    values='recommended_stock_pct', 
                    names='product_category_name',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                st.dataframe(
                    allocation_data[['product_category_name', 'recommended_stock_pct', 'category_units']]
                    .rename(columns={
                        'product_category_name': 'Product Category',
                        'recommended_stock_pct': 'Stock Share (%)',
                        'category_units': 'Historical Units Sold'
                    }).reset_index(drop=True),
                    use_container_width=True
                )
            else:
                st.info("No specific category stock data available for this combination.")

        with col_res2:
            st.subheader("LLM Strategic Action Plan")
            top_cats = allocation_data['product_category_name'].head(3).tolist() if not allocation_data.empty else ["General Merchandise"]
            st.info(f"""
            **Automated Warehouse Plan for {selected_state} ({selected_season_label}):**
            
            1. **Stock Priority:** Focus 60% of regional buffer capacity on top category: **{top_cats[0]}**.
            2. **Lead Time Handling:** Expected delay risk is **{avg_delay_risk:.1f}%**. Pre-position fulfillment stock in nearest hub.
            3. **Volume Scaling:** Estimated demand is **{predicted_volume:,} units**. Increase cross-docking slots accordingly.
            """)

# ------------------------------------------------------------------------------
# Tab 3: Raw Data Details
# ------------------------------------------------------------------------------
elif menu_option == "Raw Data Details":
    st.title("Full Stock Allocation Dataset")
    if not category_season_shares.empty:
        st.dataframe(category_season_shares, use_container_width=True)
    else:
        st.warning("No data found in pickle file.")
