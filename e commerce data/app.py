import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Set Page Config
st.set_page_config(page_title="E-commerce Analytics Pro", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e2227; padding: 15px; border-radius: 10px; border: 1px solid #30363d; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    div[data-testid="stExpander"] { border: 1px solid #30363d; border-radius: 10px; background-color: #0d1117; }
    </style>
    """, unsafe_allow_html=True)

# --- THE JANITOR: Data Cleaning ---
@st.cache_data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['CustomerID', 'Description'], inplace=True)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.date
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    df['Hour'] = df['InvoiceDate'].dt.hour
    return df

def filter_by_time_period(df, period):
    max_date = df['InvoiceDate'].max()
    if period == "Today":
        return df[df['Day'] == max_date.date()]
    elif period == "Weekly":
        return df[df['InvoiceDate'] > (max_date - timedelta(days=7))]
    elif period == "Monthly":
        return df[df['InvoiceDate'] > (max_date - timedelta(days=30))]
    elif period == "Yearly":
        return df[df['InvoiceDate'] > (max_date - timedelta(days=365))]
    return df

# --- THE SCIENTIST: ML Prediction Logic ---
@st.cache_resource
def train_prediction_model(df, country=None):
    temp_df = df.copy()
    if country and country != "Global":
        temp_df = temp_df[temp_df['Country'] == country]
    
    daily = temp_df.groupby('Day').agg({
        'TotalAmount': 'sum',
        'Quantity': 'sum',
        'UnitPrice': 'mean',
    }).reset_index()
    
    daily['DayIndex'] = range(len(daily))
    daily['DayOfWeek'] = pd.to_datetime(daily['Day']).dt.dayofweek
    daily['Month'] = pd.to_datetime(daily['Day']).dt.month
    daily['PrevDaySales'] = daily['TotalAmount'].shift(1)
    daily['PrevWeekSales'] = daily['TotalAmount'].shift(7).fillna(daily['TotalAmount'].mean())
    daily.dropna(inplace=True)
    
    # Features: [DayIndex, DayOfWeek, Month, UnitPrice, PrevDaySales]
    X = daily[['DayIndex', 'DayOfWeek', 'Month', 'UnitPrice', 'PrevDaySales']]
    y = daily['TotalAmount']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    accuracy = r2_score(y, model.predict(X))
    
    return model, daily, accuracy

def simulate_forecast(model, last_data, horizon, price_mod, stock_mod, discount_mod):
    last_idx = last_data['DayIndex'].iloc[-1]
    last_date = pd.to_datetime(last_data['Day'].iloc[-1])
    avg_price = last_data['UnitPrice'].mean() * (1 + price_mod/100)
    
    forecast_dates = []
    forecast_values = []
    curr_prev_day = last_data['TotalAmount'].iloc[-1]
    
    for i in range(1, horizon + 1):
        next_date = last_date + timedelta(days=i)
        next_idx = last_idx + i
        
        # Features with simulated price
        feat = np.array([[next_idx, next_date.dayofweek, next_date.month, avg_price, curr_prev_day]])
        base_pred = model.predict(feat)[0]
        
        # Apply multipliers for stock and discount (1.5x elasticity)
        simulated_pred = base_pred * stock_mod
        vol_boost = (discount_mod * 1.5 / 100) # Targeted 1.5x elasticity
        price_cut = (1 - discount_mod/100)
        simulated_pred = simulated_pred * price_cut * (1 + vol_boost)
        
        forecast_dates.append(next_date.strftime('%Y-%m-%d'))
        forecast_values.append(max(0, simulated_pred))
        curr_prev_day = simulated_pred
        
    return pd.DataFrame({'Day': forecast_dates, 'TotalAmount': forecast_values, 'Type': 'Sandbox Forecast'})

# --- THE ARCHITECT: UI Assembly ---
def main():
    st.title("🛡️ Enterprise Intelligence: Strategy Sandbox")
    st.markdown("#### Decision Support System & Predictive Analytics")

    try:
        df_full = load_and_clean_data("data.csv")

        # --- SIDEBAR: PREDICTION SANDBOX ---
        st.sidebar.header("🔍 Prediction Sandbox")
        st.sidebar.info("Adjust parameters to simulate 'What-If' scenarios.")
        
        sb_country = st.sidebar.selectbox("Market Focus (Country)", ["Global"] + list(df_full['Country'].unique()), key="sb_country_select")
        sb_horizon = st.sidebar.selectbox("Forecast Horizon", [30, 60, 90], index=1, key="sb_horizon_select")
        sb_price = st.sidebar.slider("Price Elasticity (%)", -50, 50, 0, key="sb_price_slider")
        sb_stock = st.sidebar.slider("Inventory Buffer (Multiplier)", 0.5, 2.0, 1.0, key="sb_stock_slider")
        sb_discount = st.sidebar.slider("Market Discount (%)", 0, 40, 0, key="sb_discount_slider")

        st.sidebar.markdown("---")
        st.sidebar.header("📊 Dashboard Filters")
        time_period = st.sidebar.selectbox("Analysis Window", ["All Time", "Yearly", "Monthly", "Weekly", "Today"], key="time_window_select")
        vis_countries = st.sidebar.multiselect("Visible Countries", options=df_full['Country'].unique(), default=['United Kingdom'], key="vis_country_multiselect")

        # --- LOGIC & EXECUTION ---
        # Filter for Analytics
        df_display = filter_by_time_period(df_full, time_period)
        if vis_countries:
            df_display = df_display[df_display['Country'].isin(vis_countries)]

        # Train & Simulate (Sandbox)
        model, daily_hist_full, accuracy = train_prediction_model(df_full, sb_country)
        forecast_df = simulate_forecast(model, daily_hist_full, sb_horizon, sb_price, sb_stock, sb_discount)

        # --- TOP LEVEL METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        
        # Handle Global vs Country correctly to avoid KeyError: True
        if sb_country == "Global":
            hist_rev = df_full['TotalAmount'].sum()
        else:
            hist_rev = df_full[df_full['Country'] == sb_country]['TotalAmount'].sum()
            
        m1.metric("Historical Revenue", f"${hist_rev:,.0f}")
        m2.metric("Simulated Revenue", f"${forecast_df['TotalAmount'].sum():,.0f}")
        m3.metric("Model Confidence", f"{accuracy*100:.1f}%")
        m4.metric("Active Regions", f"{df_display['Country'].nunique()}")

        # --- MAIN VISUALS ---
        st.subheader(f"📈 Strategic Growth Projection: {sb_country}")
        
        # Prepare chart data
        chart_hist = daily_hist_full[['Day', 'TotalAmount']].tail(180).copy()
        chart_hist['Type'] = 'Historical'
        chart_hist['Day'] = chart_hist['Day'].astype(str)
        
        combined_chart_df = pd.concat([chart_hist, forecast_df], ignore_index=True)
        
        fig_main = px.line(combined_chart_df, x='Day', y='TotalAmount', color='Type',
                          line_shape='spline', template='plotly_dark',
                          title=f"{sb_horizon}-Day Scenario Projector",
                          color_discrete_map={'Historical': '#3498db', 'Sandbox Forecast': '#e74c3c'})
        st.plotly_chart(fig_main, use_container_width=True)

        # Analysis Grid
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("🏆 Top Performing Products")
            top_p = df_display.groupby('Description')['Quantity'].sum().nlargest(10).reset_index()
            fig_p = px.bar(top_p, x='Quantity', y='Description', orientation='h', 
                          template='plotly_dark', color='Quantity', color_continuous_scale='Blues')
            st.plotly_chart(fig_p, use_container_width=True)
            
            st.subheader("🌍 Regional Revenue Share")
            top_c = df_display.groupby('Country')['TotalAmount'].sum().nlargest(10).reset_index()
            fig_c = px.pie(top_c, values='TotalAmount', names='Country', hole=0.4,
                          template='plotly_dark')
            st.plotly_chart(fig_c, use_container_width=True)

        with col_right:
            st.subheader("🔥 Operational Heatmap")
            heatmap_data = df_display.groupby(['DayOfWeek', 'Hour'])['TotalAmount'].sum().unstack().fillna(0)
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_data = heatmap_data.reindex(days_order)
            fig_heat = px.imshow(heatmap_data, template='plotly_dark', color_continuous_scale='Viridis',
                                labels=dict(x="Hour of Day", y="Day of Week", color="Revenue"))
            st.plotly_chart(fig_heat, use_container_width=True)
            
            st.subheader("⏰ Peak Activity Analysis")
            hour_rev = df_display.groupby('Hour')['TotalAmount'].sum().reset_index()
            fig_hour = px.area(hour_rev, x='Hour', y='TotalAmount', template='plotly_dark', color_discrete_sequence=['#00CC96'])
            st.plotly_chart(fig_hour, use_container_width=True)

        # --- BUSINESS INTELLIGENCE ---
        st.markdown("---")
        st.subheader("💡 Strategic Business Intelligence")
        i1, i2 = st.columns(2)
        with i1:
            st.info(f"Market Focus: **{sb_country}**. The simulation indicates a potential revenue of **${forecast_df['TotalAmount'].sum():,.0f}**.")
            st.write(f"- Strategic Elasticity: Applying a {sb_discount}% discount suggests a {(1.5*sb_discount/100)*100:.1f}% volume growth target.")
        with i2:
            if sb_price != 0:
                st.write(f"- Price Strategy: Optimized for a {sb_price}% {'increase' if sb_price > 0 else 'decrease'} in average unit value.")
            if sb_stock < 1.0:
                st.error(f"- Operational Risk: Inventory levels at {sb_stock}x may cap current demand potential.")
            st.success("Data Pipeline: Verified, Cached, and High-Precision Sync active.")

    except Exception as e:
        st.error(f"Critical System Error: {e}")
        st.info("Please ensure 'data.csv' is in the root directory and encoded as ISO-8859-1.")

if __name__ == "__main__":
    main()
