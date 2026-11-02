import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="TransitIQ | Smart Mobility",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. THEME
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080F1D;
    --sidebar: #101C30;
    --card: #14243A;
    --card2: #172A43;
    --border: #29435F;
    --text: #E2E8F0;
    --muted: #94A3B8;
    --cyan: #22D3EE;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] {
    background: rgba(8, 15, 29, 0.96);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111D32 0%, #0B1526 100%);
    border-right: 1px solid #263750;
}
[data-testid="stSidebar"] h1 {
    color: #F8FAFC !important;
    font-size: 27px !important;
    font-weight: 800 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #263750;
}
h1, h2, h3 {
    color: #F8FAFC !important;
    font-weight: 700 !important;
}
p, .stCaption {
    color: var(--muted);
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #172A43, #112239);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 5px 18px rgba(0,0,0,.15);
    transition: .2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: var(--cyan);
    transform: translateY(-2px);
}
[data-testid="stMetricLabel"] {
    color: #A8B6CA !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
    font-size: 29px !important;
    font-weight: 800 !important;
}
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(90deg, #0891B2, #06B6D4);
    color: white;
    border: 0;
    border-radius: 9px;
    min-height: 42px;
    font-weight: 700;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: #0E7490;
    color: white;
    border: 0;
}
.stSelectbox > div > div, .stNumberInput input, .stTextInput input {
    background: #14243A !important;
    color: #F1F5F9 !important;
    border-color: #304761 !important;
    border-radius: 8px !important;
}
.stProgress > div > div > div {
    background: #06B6D4;
}
hr {
    border-color: #24364F !important;
    margin-top: 24px;
    margin-bottom: 24px;
}
[data-testid="stDataFrame"] {
    border: 1px solid #29435F;
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stAlert"] {
    border-radius: 12px;
}
footer {
    visibility: hidden;
}

/* ================= ANIMATION ENHANCEMENTS ================= */

/* Smooth entrance whenever a page renders */
@keyframes pageFadeUp {
    from {
        opacity: 0;
        transform: translateY(16px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes softGlow {
    0%, 100% {
        text-shadow: 0 0 8px rgba(34, 211, 238, 0.10);
    }
    50% {
        text-shadow: 0 0 22px rgba(34, 211, 238, 0.38);
    }
}

@keyframes borderShimmer {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

@keyframes buttonShine {
    0% { transform: translateX(-150%) skewX(-20deg); }
    100% { transform: translateX(250%) skewX(-20deg); }
}

@keyframes sidebarPulse {
    0%, 100% { opacity: 0.82; }
    50% { opacity: 1; }
}

/* Main content entrance */
[data-testid="stMainBlockContainer"] {
    animation: pageFadeUp 0.65s ease-out both;
}

/* Animated dashboard title */
.transitiq-title {
    animation: softGlow 3.5s ease-in-out infinite;
}

/* KPI cards: lift and glow on hover */
[data-testid="stMetric"] {
    position: relative;
    overflow: hidden;
    transition:
        transform 0.28s ease,
        border-color 0.28s ease,
        box-shadow 0.28s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-5px) scale(1.012);
    border-color: #22D3EE !important;
    box-shadow:
        0 10px 28px rgba(0, 0, 0, 0.24),
        0 0 18px rgba(34, 211, 238, 0.13);
}

/* Slight stagger for KPI cards */
[data-testid="stHorizontalBlock"] [data-testid="stMetric"] {
    animation: pageFadeUp 0.7s ease-out both;
}
[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(1) [data-testid="stMetric"] {
    animation-delay: 0.06s;
}
[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(2) [data-testid="stMetric"] {
    animation-delay: 0.14s;
}
[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(3) [data-testid="stMetric"] {
    animation-delay: 0.22s;
}
[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(4) [data-testid="stMetric"] {
    animation-delay: 0.30s;
}

/* Buttons get a moving highlight on hover */
.stButton > button,
.stFormSubmitButton > button {
    position: relative;
    overflow: hidden;
    transition: transform 0.22s ease, box-shadow 0.22s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 7px 20px rgba(6, 182, 212, 0.24);
}

/* Sidebar navigation hover */
[data-testid="stSidebar"] [role="radiogroup"] label {
    transition: transform 0.2s ease, color 0.2s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    transform: translateX(4px);
}

/* Chart and table containers fade in gently */
[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"] {
    animation: pageFadeUp 0.7s ease-out both;
}

/* Inputs respond smoothly */
.stSelectbox, .stNumberInput, .stTextInput, .stSlider {
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

/* Subtle animated progress fill */
.stProgress > div > div > div {
    background: linear-gradient(
        90deg,
        #0891B2,
        #22D3EE,
        #67E8F9,
        #0891B2
    );
    background-size: 200% 100%;
    animation: borderShimmer 3s linear infinite;
}

/* Sidebar brand accent */
[data-testid="stSidebar"] h1 {
    animation: sidebarPulse 4s ease-in-out infinite;
}

/* Respect users who prefer reduced motion */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. CONSTANTS AND FEATURE LISTS
# =========================================================
BG = "#080F1D"
TEXT = "#E2E8F0"
MUTED = "#94A3B8"
CROWD_ORDER = ["Low", "Medium", "High", "Critical"]
CROWD_COLORS = {
    "Low": "#10B981",
    "Medium": "#FBBF24",
    "High": "#F97316",
    "Critical": "#EF4444"
}

NUMERICAL_FEATURES = [
    "hour", "is_weekend", "is_holiday", "scheduled_buses",
    "actual_buses", "temperature_c", "rainfall_mm", "bus_capacity"
]
CATEGORICAL_FEATURES = [
    "day_of_week", "route_id", "route_name", "origin_stop",
    "destination_stop", "weather", "special_event", "hour_category"
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


# =========================================================
# 4. LOAD DATA AND TRAINED MODELS
# =========================================================
@st.cache_data
def load_data():
    data = pd.read_csv("chandigarh_cleaned.csv")
    return data


@st.cache_resource
def load_models():
    rf = joblib.load("models/demand_rf.pkl")
    xgb = joblib.load("models/demand_xgb.pkl")
    classifier = joblib.load("models/crowd_classifier.pkl")
    kmeans = joblib.load("models/kmeans_model.pkl")
    return rf, xgb, classifier, kmeans


try:
    df = load_data()
    rf_model, xgb_model, crowd_model, kmeans_model = load_models()
except Exception as exc:
    st.error(
        "TransitIQ could not load its data or model files. "
        "Check that app.py, chandigarh_cleaned.csv and the models folder "
        "are in the same project directory."
    )
    st.exception(exc)
    st.stop()

required_columns = set(
    MODEL_FEATURES + [
        "record_id", "passenger_count", "crowd_level",
        "occupancy_rate", "demand_cluster"
    ]
)
missing_columns = required_columns.difference(df.columns)
# demand_cluster is created in the clustered CSV, not necessarily cleaned CSV.
missing_for_app = missing_columns.difference({"demand_cluster"})
if missing_for_app:
    st.error(f"Dataset is missing required columns: {sorted(missing_for_app)}")
    st.stop()


# =========================================================
# 5. REUSABLE HELPERS
# =========================================================
def hour_category(hour):
    if hour < 7:
        return "Early Morning"
    if hour < 11:
        return "Morning Peak"
    if hour < 16:
        return "Afternoon"
    if hour < 20:
        return "Evening Peak"
    return "Night"


def crowd_from_occupancy(rate):
    if rate < 0.50:
        return "Low"
    if rate < 0.80:
        return "Medium"
    if rate < 1.00:
        return "High"
    return "Critical"


def make_input(
    route_id, origin, destination, day, hour, weather,
    temperature, rainfall, special_event, scheduled_buses,
    actual_buses, capacity, holiday
):
    route_names = df.loc[df["route_id"] == route_id, "route_name"].dropna()
    route_name = route_names.mode().iloc[0] if not route_names.empty else "Unknown"

    row = {
        "day_of_week": day,
        "route_id": route_id,
        "route_name": route_name,
        "origin_stop": origin,
        "destination_stop": destination,
        "weather": weather,
        "special_event": special_event,
        "hour_category": hour_category(hour),
        "hour": int(hour),
        "is_weekend": int(day in ["Saturday", "Sunday"]),
        "is_holiday": int(holiday),
        "scheduled_buses": int(scheduled_buses),
        "actual_buses": int(actual_buses),
        "temperature_c": float(temperature),
        "rainfall_mm": float(rainfall),
        "bus_capacity": int(capacity)
    }
    return pd.DataFrame([row], columns=MODEL_FEATURES)


def style_chart(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial", color=TEXT, size=12),
        height=height,
        margin=dict(l=18, r=18, t=42, b=28),
        hoverlabel=dict(bgcolor="#172A43", font_color="white"),
        legend_title_text=""
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#263750")
    fig.update_yaxes(gridcolor="#263750", zeroline=False)
    return fig


def style_table(dataframe):
    """Apply dark styling to Streamlit's Pandas Styler table."""
    return (
        dataframe.style
        .format(precision=2)
        .set_properties(**{
            "background-color": "#14243A",
            "color": "#E2E8F0",
            "border-color": "#29435F",
            "padding": "10px"
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#0E1B2E"),
                    ("color", "#22D3EE"),
                    ("font-weight", "700"),
                    ("border-color", "#29435F"),
                    ("padding", "11px")
                ]
            },
            {
                "selector": "td",
                "props": [("border-color", "#29435F")]
            },
            {
                "selector": "table",
                "props": [
                    ("background-color", "#14243A"),
                    ("border-collapse", "collapse"),
                    ("width", "100%")
                ]
            }
        ])
    )


def page_header(title, subtitle, section):
    # Flush-left HTML prevents Markdown from treating the markup as code.
    html = f"""
<div class="transitiq-page-header" style="padding: 8px 0 22px 0;">
  <div style="color:#22D3EE;font-size:11px;font-weight:800;
              letter-spacing:3px;margin-bottom:12px;">
    {section}
  </div>
  <div class="transitiq-title"
       style="color:#F8FAFC;font-size:35px;font-weight:800;
              letter-spacing:-1px;line-height:1.2;">
    {title}
  </div>
  <div style="color:#94A3B8;font-size:14px;margin-top:10px;">
    {subtitle}
  </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def section_heading(title, subtitle=None):
    st.markdown(
        f"<h3 style='margin-bottom:3px'>{title}</h3>",
        unsafe_allow_html=True
    )
    if subtitle:
        st.caption(subtitle)


def show_table(data, hide_index=True):
    st.dataframe(
        style_table(data),
        use_container_width=True,
        hide_index=hide_index
    )


# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================
st.sidebar.markdown("# 🚌 TransitIQ")
st.sidebar.markdown(
    "<p style='color:#7F93AC;font-size:11px;letter-spacing:2px;"
    "font-weight:700;'>SMART MOBILITY ANALYTICS</p>",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='color:#22D3EE;font-size:11px;font-weight:800;"
    "letter-spacing:2px;'>NAVIGATION</p>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Demand Prediction",
        "Crowd Analysis",
        "Route Analytics",
        "Route Explorer",
        "Bus Allocation Simulator",
        "Demand Heatmap",
        "Smart Insights",
        "ML Performance"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Overview")
st.sidebar.markdown("""
<div style="background:#172A43;border:1px solid #29435F;
            padding:13px;border-radius:10px;line-height:2.1;
            color:#CBD5E1;font-size:13px;">
📍 Chandigarh, India<br>
🤖 Machine Learning<br>
📊 4 ML Algorithms<br>
🚌 Smart Mobility Analytics
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.caption("TransitIQ | MCA Machine Learning Project")


# =========================================================
# 7. OVERVIEW
# =========================================================
if page == "Overview":
    page_header(
        "Transport Intelligence",
        "Monitor passenger demand, route performance and crowd patterns across Chandigarh.",
        "CHANDIGARH  /  SMART MOBILITY"
    )
    st.markdown("---")

    total_records = len(df)
    avg_passengers = float(df["passenger_count"].mean())
    busiest_route = (
        df.groupby("route_id")["passenger_count"].mean().idxmax()
    )
    critical_count = int((df["crowd_level"] == "Critical").sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Passenger Observations", f"{total_records:,}")
    k2.metric("Average Passengers", f"{avg_passengers:.1f}")
    k3.metric("Busiest Route", str(busiest_route))
    k4.metric("Critical Crowd Cases", f"{critical_count:,}")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        section_heading(
            "Route-wise Passenger Demand",
            "Average passenger count across bus routes"
        )
        route_data = (
            df.groupby("route_id", as_index=False)["passenger_count"]
            .mean()
            .sort_values("passenger_count", ascending=False)
        )
        fig = px.bar(
            route_data,
            x="route_id",
            y="passenger_count",
            color="passenger_count",
            color_continuous_scale=["#164E63", "#0891B2", "#22D3EE", "#A5F3FC"],
            labels={"route_id": "Route", "passenger_count": "Average Passengers"}
        )
        fig.update_traces(
            marker_line_width=0,
            hovertemplate="<b>Route %{x}</b><br>Average passengers: %{y:.1f}<extra></extra>"
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with right:
        section_heading(
            "Crowd Distribution",
            "Share of observations by crowd category"
        )
        crowd_data = (
            df["crowd_level"].value_counts()
            .reindex(CROWD_ORDER, fill_value=0)
            .rename_axis("Crowd Level")
            .reset_index(name="Count")
        )
        fig = px.pie(
            crowd_data,
            names="Crowd Level",
            values="Count",
            hole=0.66,
            color="Crowd Level",
            color_discrete_map=CROWD_COLORS
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            textfont=dict(size=13, color="white"),
            marker=dict(line=dict(color=BG, width=3)),
            hovertemplate="<b>%{label}</b><br>Records: %{value:,}<br>Share: %{percent}<extra></extra>"
        )
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.20,
                        xanchor="center", x=0.5)
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.markdown("---")
    section_heading(
        "Passenger Demand Throughout the Day",
        "Average passenger demand by journey hour"
    )
    hourly = (
        df.groupby("hour", as_index=False)["passenger_count"].mean()
        .sort_values("hour")
    )
    fig = px.area(hourly, x="hour", y="passenger_count")
    fig.update_traces(
        line=dict(color="#22D3EE", width=3),
        fillcolor="rgba(34,211,238,0.13)",
        marker=dict(size=6, color="#22D3EE")
    )
    fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Average Passengers")
    st.plotly_chart(style_chart(fig, 340), use_container_width=True)

    st.markdown("---")
    section_heading("Top Performing Routes", "Routes ranked by average passenger demand")
    top_routes = route_data.head(5).rename(columns={
        "route_id": "Route",
        "passenger_count": "Average Passenger Demand"
    })
    show_table(top_routes)


# =========================================================
# 8. DEMAND PREDICTION
# =========================================================
elif page == "Demand Prediction":
    page_header(
        "Passenger Demand Prediction",
        "Estimate passenger demand using the trained regression models.",
        "PREDICTIVE ANALYTICS"
    )
    st.markdown("---")

    with st.form("demand_prediction_form"):
        st.subheader("Journey Information")
        c1, c2 = st.columns(2)

        with c1:
            route_id = st.selectbox(
                "Select Bus Route",
                sorted(df["route_id"].dropna().unique())
            )
            route_rows = df[df["route_id"] == route_id]
            origins = sorted(route_rows["origin_stop"].dropna().unique())
            destinations = sorted(route_rows["destination_stop"].dropna().unique())
            origin = st.selectbox("Origin Stop", origins, key="demand_origin")
            destination = st.selectbox(
                "Destination Stop", destinations, key="demand_destination"
            )
            day = st.selectbox(
                "Day of Week",
                ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
            )
            hour = st.slider("Journey Hour", 0, 23, 8)
            weather = st.selectbox(
                "Weather", sorted(df["weather"].dropna().unique())
            )

        with c2:
            special_event = st.selectbox(
                "Special Event", sorted(df["special_event"].dropna().unique())
            )
            temperature = st.number_input(
                "Temperature (°C)", -10.0, 55.0, 25.0
            )
            rainfall = st.number_input(
                "Rainfall (mm)", 0.0, 100.0, 0.0
            )
            scheduled_buses = st.number_input(
                "Scheduled Buses", 1, 20, 4
            )
            actual_buses = st.number_input(
                "Actual Buses", 1, 20, 4
            )
            capacity = st.selectbox(
                "Capacity Per Bus", [40, 45, 50, 55, 60], index=2
            )
            holiday = st.selectbox(
                "Is Holiday?", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

        submitted = st.form_submit_button(
            "Generate Demand Prediction", use_container_width=True
        )

    if submitted:
        if origin == destination:
            st.error("Origin and destination cannot be the same.")
        else:
            model_input = make_input(
                route_id, origin, destination, day, hour, weather,
                temperature, rainfall, special_event, scheduled_buses,
                actual_buses, capacity, holiday
            )
            rf_prediction = max(0.0, float(rf_model.predict(model_input)[0]))
            xgb_prediction = max(0.0, float(xgb_model.predict(model_input)[0]))
            predicted_occupancy = xgb_prediction / (actual_buses * capacity)
            predicted_crowd = crowd_from_occupancy(predicted_occupancy)

            st.markdown("---")
            st.subheader("Prediction Results")
            a, b, c = st.columns(3)
            a.metric("Expected Passengers", f"{xgb_prediction:.0f}")
            b.metric("Available Capacity", f"{actual_buses * capacity}")
            c.metric("Predicted Occupancy", f"{predicted_occupancy * 100:.1f}%")

            st.markdown(
                f"""
<div style="background:#14243A;border:1px solid #29435F;
            border-left:5px solid {CROWD_COLORS[predicted_crowd]};
            border-radius:12px;padding:20px;margin:12px 0;">
  <div style="color:#94A3B8;font-size:12px;font-weight:700;letter-spacing:2px;">
    PREDICTED CROWD STATUS
  </div>
  <div style="font-size:30px;font-weight:800;color:{CROWD_COLORS[predicted_crowd]};
              margin-top:8px;">
    {predicted_crowd}
  </div>
  <div style="color:#CBD5E1;font-size:13px;">
    Estimated occupancy based on predicted passengers and entered bus capacity.
  </div>
</div>
""",
                unsafe_allow_html=True
            )

            st.markdown("#### Capacity Utilization")
            st.progress(float(np.clip(predicted_occupancy, 0.0, 1.0)))
            st.caption(f"{predicted_occupancy * 100:.1f}% of available passenger capacity")

            if predicted_crowd == "Critical":
                st.error("Potential overcrowding detected. Consider increasing bus availability.")
            elif predicted_crowd == "High":
                st.warning("High passenger demand expected. Monitor bus capacity.")
            else:
                st.success("Expected demand is within the estimated available capacity.")

            st.markdown("---")
            section_heading("Regression Model Comparison", "Predicted passengers for the same journey")
            comparison = pd.DataFrame({
                "Model": ["Random Forest", "XGBoost"],
                "Predicted Passengers": [rf_prediction, xgb_prediction]
            })
            fig = px.bar(
                comparison, x="Model", y="Predicted Passengers",
                color="Model", color_discrete_sequence=["#0891B2", "#22D3EE"],
                text="Predicted Passengers"
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            st.plotly_chart(style_chart(fig), use_container_width=True)
            st.caption(
                "The crowd status on this page is calculated from XGBoost's "
                "passenger estimate and the entered capacity."
            )


# =========================================================
# 9. CROWD ANALYSIS
# =========================================================
elif page == "Crowd Analysis":
    page_header(
        "Crowd Monitoring & Classification",
        "Classify crowd levels using the trained Decision Tree classifier.",
        "CROWD INTELLIGENCE"
    )
    st.markdown("---")

    with st.form("crowd_prediction_form"):
        st.subheader("Transportation Conditions")
        c1, c2 = st.columns(2)

        with c1:
            route_id = st.selectbox(
                "Select Route",
                sorted(df["route_id"].dropna().unique()),
                key="classifier_route"
            )
            route_rows = df[df["route_id"] == route_id]
            origins = sorted(route_rows["origin_stop"].dropna().unique())
            destinations = sorted(route_rows["destination_stop"].dropna().unique())
            origin = st.selectbox("Origin Stop", origins, key="classifier_origin")
            destination = st.selectbox(
                "Destination Stop", destinations, key="classifier_destination"
            )
            day = st.selectbox(
                "Day",
                ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"],
                key="classifier_day"
            )
            hour = st.slider("Journey Hour", 0, 23, 8, key="classifier_hour")
            weather = st.selectbox(
                "Weather", sorted(df["weather"].dropna().unique()),
                key="classifier_weather"
            )

        with c2:
            special_event = st.selectbox(
                "Special Event", sorted(df["special_event"].dropna().unique()),
                key="classifier_event"
            )
            temperature = st.number_input(
                "Temperature (°C)", -10.0, 55.0, 25.0,
                key="classifier_temperature"
            )
            rainfall = st.number_input(
                "Rainfall (mm)", 0.0, 100.0, 0.0,
                key="classifier_rainfall"
            )
            scheduled_buses = st.number_input(
                "Scheduled Buses", 1, 20, 4, key="classifier_scheduled"
            )
            actual_buses = st.number_input(
                "Actual Buses", 1, 20, 4, key="classifier_actual"
            )
            capacity = st.selectbox(
                "Capacity Per Bus", [40, 45, 50, 55, 60],
                index=2, key="classifier_capacity"
            )
            holiday = st.selectbox(
                "Is Holiday?", [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                key="classifier_holiday"
            )

        classify_clicked = st.form_submit_button(
            "Classify Crowd Level", use_container_width=True
        )

    if classify_clicked:
        if origin == destination:
            st.error("Origin and destination cannot be the same.")
        else:
            model_input = make_input(
                route_id, origin, destination, day, hour, weather,
                temperature, rainfall, special_event, scheduled_buses,
                actual_buses, capacity, holiday
            )
            prediction = str(crowd_model.predict(model_input)[0])
            probabilities = crowd_model.predict_proba(model_input)[0]
            classes = crowd_model.named_steps["model"].classes_

            st.markdown("---")
            st.subheader("Classification Result")
            st.markdown(
                f"""
<div style="background:#14243A;border:1px solid #29435F;
            border-left:5px solid {CROWD_COLORS.get(prediction, '#22D3EE')};
            border-radius:12px;padding:22px;">
  <div style="font-size:12px;color:#94A3B8;letter-spacing:2px;font-weight:700;">
    PREDICTED CROWD LEVEL
  </div>
  <div style="font-size:32px;font-weight:800;
              color:{CROWD_COLORS.get(prediction, '#22D3EE')};
              margin-top:10px;">
    {prediction}
  </div>
</div>
""",
                unsafe_allow_html=True
            )

            st.markdown("### Class Prediction Probabilities")
            probability_data = pd.DataFrame({
                "Crowd Level": classes,
                "Probability": probabilities * 100
            })
            fig = px.bar(
                probability_data,
                x="Crowd Level",
                y="Probability",
                color="Crowd Level",
                color_discrete_map=CROWD_COLORS,
                text="Probability"
            )
            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title=None,
                yaxis_title="Estimated Probability (%)",
                yaxis_range=[0, max(100, float(probability_data["Probability"].max()) + 10)]
            )
            st.plotly_chart(style_chart(fig), use_container_width=True)
            st.caption(
                "These are the Decision Tree's estimated class probabilities, "
                "not guaranteed real-world confidence."
            )


# =========================================================
# 10. ROUTE ANALYTICS
# =========================================================
elif page == "Route Analytics":
    page_header(
        "Route Performance Analytics",
        "Explore passenger demand, route utilization and peak travel periods.",
        "ROUTE INTELLIGENCE"
    )
    st.markdown("---")

    route_data = (
        df.groupby("route_id")
        .agg(
            Average_Passengers=("passenger_count", "mean"),
            Total_Passengers=("passenger_count", "sum"),
            Average_Occupancy=("occupancy_rate", "mean"),
            Observations=("record_id", "count")
        )
        .reset_index()
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Routes", f"{df['route_id'].nunique()}")
    c2.metric(
        "Highest Demand Route",
        str(route_data.loc[route_data["Average_Passengers"].idxmax(), "route_id"])
    )
    c3.metric(
        "Average Occupancy",
        f"{df['occupancy_rate'].mean() * 100:.1f}%"
    )

    st.markdown("---")
    section_heading("Route Performance Table", "Demand and occupancy summary by route")

    display_data = route_data.rename(columns={
        "route_id": "Route",
        "Average_Passengers": "Average Passengers",
        "Total_Passengers": "Total Passengers",
        "Average_Occupancy": "Average Occupancy",
        "Observations": "Observations"
    })
    show_table(display_data)

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        section_heading("Peak Hour Analysis")
        hourly_data = (
            df.groupby("hour", as_index=False)["passenger_count"].mean()
        )
        fig = px.bar(
            hourly_data,
            x="hour",
            y="passenger_count",
            color="passenger_count",
            color_continuous_scale=["#164E63", "#0891B2", "#22D3EE"]
        )
        fig.update_layout(
            coloraxis_showscale=False,
            xaxis_title="Hour",
            yaxis_title="Average Passengers"
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with right:
        section_heading("Route Demand Distribution")
        fig = px.box(
            df, x="route_id", y="passenger_count",
            color="route_id"
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="Route",
            yaxis_title="Passenger Count"
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.markdown("---")
    section_heading("Top Five Routes by Demand")
    top_five = route_data.nlargest(5, "Average_Passengers")
    fig = px.bar(
        top_five,
        x="route_id",
        y="Average_Passengers",
        color="Average_Passengers",
        color_continuous_scale=["#164E63", "#0891B2", "#22D3EE"],
        text="Average_Passengers"
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Route",
        yaxis_title="Average Passengers"
    )
    st.plotly_chart(style_chart(fig), use_container_width=True)



# =========================================================
# 11. INTERACTIVE ROUTE EXPLORER
# =========================================================
elif page == "Route Explorer":
    page_header(
        "Interactive Route Explorer",
        "Choose a Chandigarh bus route and explore its demand, crowd profile and busiest travel periods.",
        "EXPLORE THE NETWORK"
    )
    route_options = sorted(df["route_id"].dropna().unique())
    selected_route = st.selectbox("Choose a bus route", route_options, key="explorer_route")
    route_df = df[df["route_id"] == selected_route].copy()

    if route_df.empty:
        st.info("No observations are available for this route.")
    else:
        avg_demand = float(route_df["passenger_count"].mean())
        peak_hour = int(route_df.groupby("hour")["passenger_count"].mean().idxmax())
        avg_occupancy = float(route_df["occupancy_rate"].mean())
        critical_share = float((route_df["crowd_level"] == "Critical").mean() * 100)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Average Passengers", f"{avg_demand:.1f}")
        m2.metric("Average Occupancy", f"{avg_occupancy:.0%}")
        m3.metric("Busiest Hour", f"{peak_hour:02d}:00")
        m4.metric("Critical Observations", f"{critical_share:.1f}%")

        st.markdown("---")
        lcol, rcol = st.columns(2)
        with lcol:
            section_heading("Demand by Hour", "Average passengers recorded for this route")
            by_hour = route_df.groupby("hour", as_index=False)["passenger_count"].mean()
            fig = px.area(by_hour, x="hour", y="passenger_count",
                          markers=True, labels={"hour": "Hour", "passenger_count": "Average passengers"})
            fig.update_traces(line=dict(color="#22D3EE", width=3),
                              fillcolor="rgba(34,211,238,0.15)")
            st.plotly_chart(style_chart(fig), use_container_width=True)
        with rcol:
            section_heading("Crowd Profile", "How often each crowd level appears")
            crowd_counts = (route_df["crowd_level"].value_counts()
                            .reindex(CROWD_ORDER, fill_value=0)
                            .rename_axis("Crowd Level").reset_index(name="Records"))
            fig = px.pie(crowd_counts, names="Crowd Level", values="Records",
                         hole=0.62, color="Crowd Level", color_discrete_map=CROWD_COLORS)
            fig.update_traces(textinfo="percent", hovertemplate="%{label}: %{value} records<extra></extra>")
            st.plotly_chart(style_chart(fig), use_container_width=True)

        section_heading("Route observation explorer", "Filter the route's records and download the selected subset")
        day_options = ["All"] + sorted(route_df["day_of_week"].dropna().unique().tolist())
        chosen_day = st.selectbox("Filter by day", day_options, key="explorer_day")
        filtered_route = route_df if chosen_day == "All" else route_df[route_df["day_of_week"] == chosen_day]
        st.caption(f"Showing {len(filtered_route):,} of {len(route_df):,} observations.")
        show_cols = [c for c in ["record_id", "day_of_week", "hour", "origin_stop",
                                  "destination_stop", "passenger_count", "occupancy_rate",
                                  "crowd_level"] if c in filtered_route.columns]
        show_table(filtered_route[show_cols].head(100))
        st.download_button(
            "Download route observations",
            data=filtered_route.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_route.lower()}_route_observations.csv",
            mime="text/csv"
        )


# =========================================================
# 12. BUS ALLOCATION SIMULATOR
# =========================================================
elif page == "Bus Allocation Simulator":
    page_header(
        "Smart Bus Allocation Simulator",
        "Experiment with fleet size and passenger capacity to understand how additional buses may reduce crowding.",
        "WHAT-IF OPERATIONS LAB"
    )
    st.info("Simulation uses the trained XGBoost demand estimate and your selected capacity assumptions. It is a planning aid, not a live dispatch system.")

    with st.container(border=True):
        st.subheader("Configure a transport scenario")
        a, b = st.columns(2)
        with a:
            sim_route = st.selectbox("Route", sorted(df["route_id"].dropna().unique()), key="sim_route")
            sim_day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], key="sim_day")
            sim_hour = st.slider("Travel hour", 0, 23, 8, key="sim_hour")
            sim_weather = st.selectbox("Weather", sorted(df["weather"].dropna().unique()), key="sim_weather")
            sim_event = st.selectbox("Special event", sorted(df["special_event"].dropna().unique()), key="sim_event")
        with b:
            sim_buses = st.slider("Buses available", 1, 20, 4, key="sim_buses")
            sim_capacity = st.select_slider("Seats / passenger capacity per bus", options=[30, 35, 40, 45, 50, 55, 60, 65, 70], value=50, key="sim_capacity")
            sim_temperature = st.number_input("Temperature (°C)", -10.0, 55.0, 25.0, key="sim_temp")
            sim_rain = st.number_input("Rainfall (mm)", 0.0, 100.0, 0.0, key="sim_rain")
            sim_holiday = st.selectbox("Holiday?", [0, 1], format_func=lambda x: "Yes" if x else "No", key="sim_holiday")

        sim_route_rows = df[df["route_id"] == sim_route]
        sim_origin = sim_route_rows["origin_stop"].dropna().mode()
        sim_destination = sim_route_rows["destination_stop"].dropna().mode()
        if sim_origin.empty or sim_destination.empty:
            st.warning("This route does not have enough stop information to run the simulation.")
        else:
            sim_input = make_input(
                sim_route, str(sim_origin.iloc[0]), str(sim_destination.iloc[0]),
                sim_day, sim_hour, sim_weather, sim_temperature, sim_rain,
                sim_event, sim_buses, sim_buses, sim_capacity, sim_holiday
            )
            estimated_passengers = max(0.0, float(xgb_model.predict(sim_input)[0]))
            total_capacity = int(sim_buses * sim_capacity)
            occupancy = estimated_passengers / total_capacity if total_capacity else 0.0
            required_buses_80 = max(1, int(np.ceil(estimated_passengers / (sim_capacity * 0.80))))
            additional = max(0, required_buses_80 - sim_buses)
            status = crowd_from_occupancy(occupancy)

            st.markdown("---")
            st.subheader("Simulation outcome")
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Estimated Passengers", f"{estimated_passengers:.0f}")
            x2.metric("Total Capacity", f"{total_capacity:,}")
            x3.metric("Projected Occupancy", f"{occupancy:.1%}")
            x4.metric("Buses for ≤80% occupancy", f"{required_buses_80}")

            st.progress(float(np.clip(occupancy, 0.0, 1.0)))
            st.markdown(
                f"""<div style="background:#14243A;border:1px solid #29435F;
                border-left:5px solid {CROWD_COLORS[status]};border-radius:12px;
                padding:17px;margin-top:12px;">
                <div style="color:#94A3B8;font-size:11px;font-weight:800;letter-spacing:2px;">SIMULATED CROWD STATUS</div>
                <div style="font-size:27px;font-weight:800;color:{CROWD_COLORS[status]};margin-top:6px;">{status}</div>
                <div style="color:#CBD5E1;font-size:13px;">{('Consider adding ' + str(additional) + ' bus(es) to bring estimated occupancy to 80% or below.') if additional else 'The selected fleet meets the 80% planning occupancy target.'}</div>
                </div>""",
                unsafe_allow_html=True
            )

            fleet_values = list(range(1, 21))
            fleet_data = pd.DataFrame({
                "Buses": fleet_values,
                "Capacity": [n * sim_capacity for n in fleet_values],
                "Occupancy (%)": [estimated_passengers / (n * sim_capacity) * 100 for n in fleet_values]
            })
            fig = px.line(fleet_data, x="Buses", y="Occupancy (%)", markers=True,
                          labels={"Occupancy (%)": "Estimated occupancy (%)"},
                          title="How fleet size changes estimated occupancy")
            fig.add_hline(y=80, line_dash="dash", line_color="#FBBF24",
                          annotation_text="80% planning target")
            fig.add_vline(x=sim_buses, line_dash="dot", line_color="#22D3EE",
                          annotation_text="Selected fleet")
            fig.update_traces(line=dict(color="#22D3EE", width=3))
            st.plotly_chart(style_chart(fig, 390), use_container_width=True)


# =========================================================
# 13. DEMAND HEATMAP
# =========================================================
elif page == "Demand Heatmap":
    page_header(
        "Passenger Demand Heatmap",
        "Discover when passenger demand is highest across days and hours.",
        "TIME-BASED DEMAND EXPLORER"
    )
    route_filter = st.selectbox(
        "Route filter",
        ["All Routes"] + sorted(df["route_id"].dropna().unique().tolist()),
        key="heatmap_route"
    )
    heat_df = df if route_filter == "All Routes" else df[df["route_id"] == route_filter]
    heat_pivot = heat_df.pivot_table(
        index="day_of_week", columns="hour", values="passenger_count", aggfunc="mean"
    )
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat_pivot = heat_pivot.reindex([d for d in day_order if d in heat_pivot.index])
    heat_pivot = heat_pivot.reindex(columns=range(24))
    fig = go.Figure(data=go.Heatmap(
        z=heat_pivot.values,
        x=[f"{h:02d}:00" for h in heat_pivot.columns],
        y=heat_pivot.index.tolist(),
        colorscale=[[0, "#10243A"], [0.35, "#0E7490"], [0.7, "#06B6D4"], [1, "#F59E0B"]],
        colorbar=dict(title="Avg passengers"),
        hovertemplate="Day: %{y}<br>Time: %{x}<br>Avg passengers: %{z:.1f}<extra></extra>"
    ))
    fig.update_layout(xaxis_title="Hour of day", yaxis_title="Day of week")
    st.plotly_chart(style_chart(fig, 430), use_container_width=True)

    if not heat_pivot.empty and heat_pivot.notna().any().any():
        peak_day, peak_hour = np.unravel_index(np.nanargmax(heat_pivot.values), heat_pivot.shape)
        st.success(
            f"Highest average demand in this selection: {heat_pivot.index[peak_day]} "
            f"at {int(heat_pivot.columns[peak_hour]):02d}:00 "
            f"({heat_pivot.values[peak_day, peak_hour]:.1f} passengers on average)."
        )
    else:
        st.warning("There is not enough data to build this heatmap.")


# =========================================================
# 14. SMART INSIGHTS
# =========================================================
elif page == "Smart Insights":
    page_header(
        "Smart Mobility Insights",
        "Automatically surface useful operational observations from the available transport dataset.",
        "DATA-DRIVEN RECOMMENDATIONS"
    )
    st.caption("Insights are calculated from the project's synthetic dataset; they are not live Chandigarh transport alerts.")

    route_summary = df.groupby("route_id").agg(
        avg_passengers=("passenger_count", "mean"),
        avg_occupancy=("occupancy_rate", "mean"),
        critical_rate=("crowd_level", lambda x: (x == "Critical").mean()),
        observations=("record_id", "count")
    ).reset_index().sort_values("avg_passengers", ascending=False)

    busiest = route_summary.iloc[0]
    most_critical = route_summary.sort_values("critical_rate", ascending=False).iloc[0]
    peak = df.groupby("hour")["passenger_count"].mean().idxmax()
    low = route_summary.sort_values("avg_passengers").iloc[0]

    cards = st.columns(3)
    cards[0].metric("Highest Average Demand", str(busiest["route_id"]), f"{busiest['avg_passengers']:.1f} passengers")
    cards[1].metric("Highest Critical-Crowd Share", str(most_critical["route_id"]), f"{most_critical['critical_rate']:.1%} of records")
    cards[2].metric("Network Peak Hour", f"{int(peak):02d}:00", "Highest average demand")

    st.markdown("---")
    section_heading("Operational recommendations", "Rule-based suggestions derived from route-level patterns")
    recommendations = []
    recommendations.append((
        "Prioritise high-demand route",
        f"Route {busiest['route_id']} has the highest average passenger count ({busiest['avg_passengers']:.1f}). Review whether its scheduled fleet is sufficient during peak periods."
    ))
    recommendations.append((
        "Review crowding risk",
        f"Route {most_critical['route_id']} has the largest share of Critical observations ({most_critical['critical_rate']:.1%}). Validate its capacity and service frequency assumptions."
    ))
    recommendations.append((
        "Plan around the network peak",
        f"The dataset's highest average passenger demand occurs at {int(peak):02d}:00. Consider checking route-level fleet availability around this hour."
    ))
    recommendations.append((
        "Compare lower-demand services",
        f"Route {low['route_id']} has the lowest average demand ({low['avg_passengers']:.1f}). Compare its service pattern before reallocating buses; low demand alone does not prove excess supply."
    ))

    for idx, (heading, detail) in enumerate(recommendations, start=1):
        st.markdown(
            f"""<div style="background:linear-gradient(135deg,#14243A,#102036);
            border:1px solid #29435F;border-radius:13px;padding:17px 19px;
            margin:10px 0;">
            <div style="color:#22D3EE;font-size:11px;font-weight:800;letter-spacing:2px;">INSIGHT {idx:02d}</div>
            <div style="color:#F8FAFC;font-size:17px;font-weight:750;margin:6px 0;">{heading}</div>
            <div style="color:#B7C5D8;font-size:13px;line-height:1.6;">{detail}</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("---")
    section_heading("Route priority matrix", "Compare demand and critical-crowd share together")
    priority = route_summary.copy()
    priority["Critical Crowd Share (%)"] = priority["critical_rate"] * 100
    fig = px.scatter(
        priority, x="avg_passengers", y="Critical Crowd Share (%)",
        size="observations", color="route_id", hover_name="route_id",
        labels={"avg_passengers": "Average passengers", "route_id": "Route"}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_chart(fig, 410), use_container_width=True)
    show_table(priority[["route_id", "avg_passengers", "avg_occupancy",
                         "Critical Crowd Share (%)", "observations"]].rename(columns={
        "route_id": "Route", "avg_passengers": "Average Passengers",
        "avg_occupancy": "Average Occupancy", "observations": "Observations"
    }))


# =========================================================
# 15. ML PERFORMANCE
# =========================================================
elif page == "ML Performance":
    page_header(
        "Machine Learning Performance",
        "Evaluate and compare the trained predictive algorithms.",
        "MODEL INTELLIGENCE"
    )
    st.markdown("---")

    if os.path.exists("model_comparison.csv"):
        results = pd.read_csv("model_comparison.csv")
        best_index = results["R2 Score"].idxmax()
        best_model = results.loc[best_index, "Model"]
        best_score = float(results.loc[best_index, "R2 Score"])

        c1, c2 = st.columns(2)
        c1.metric("Best Regression Model", str(best_model))
        c2.metric("Best R² Score", f"{best_score:.4f}")

        st.markdown("---")
        section_heading(
            "Regression Model Comparison",
            "Passenger demand prediction performance"
        )
        show_table(results)

        left, right = st.columns(2)
        with left:
            fig = px.bar(
                results, x="Model", y="R2 Score",
                color="Model",
                color_discrete_sequence=["#0891B2", "#22D3EE"]
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title=None,
                yaxis_title="R² Score",
                yaxis_range=[0, 1]
            )
            st.plotly_chart(style_chart(fig), use_container_width=True)

        with right:
            error_data = results.melt(
                id_vars="Model",
                value_vars=["MAE", "RMSE"],
                var_name="Metric",
                value_name="Error"
            )
            fig = px.bar(
                error_data,
                x="Model",
                y="Error",
                color="Metric",
                barmode="group",
                color_discrete_sequence=["#22D3EE", "#F59E0B"]
            )
            fig.update_layout(xaxis_title=None, yaxis_title="Prediction Error")
            st.plotly_chart(style_chart(fig), use_container_width=True)
    else:
        st.warning("model_comparison.csv was not found.")

    st.markdown("---")
    section_heading(
        "Decision Tree Classification",
        "Class-wise precision, recall and F1-score"
    )

    if os.path.exists("classification_report.csv"):
        report = pd.read_csv("classification_report.csv", index_col=0)
        show_table(report.reset_index().rename(columns={"index": "Class"}))

        class_names = [name for name in CROWD_ORDER if name in report.index]
        if class_names:
            class_metrics = report.loc[
                class_names, ["precision", "recall", "f1-score"]
            ].reset_index()
            class_metrics = class_metrics.rename(columns={"index": "Crowd Level"})
            class_metrics = class_metrics.melt(
                id_vars="Crowd Level",
                var_name="Metric",
                value_name="Score"
            )
            fig = px.bar(
                class_metrics,
                x="Crowd Level",
                y="Score",
                color="Metric",
                barmode="group",
                color_discrete_sequence=["#22D3EE", "#10B981", "#F59E0B"]
            )
            fig.update_layout(
                xaxis_title="Crowd Category",
                yaxis_title="Score",
                yaxis_range=[0, 1]
            )
            st.plotly_chart(style_chart(fig), use_container_width=True)
    else:
        st.warning("classification_report.csv was not found.")

    st.markdown("---")
    section_heading(
        "K-Means Demand Clustering",
        "Explore passenger demand patterns discovered by unsupervised learning"
    )

    if os.path.exists("chandigarh_clustered.csv"):
        cluster_df = pd.read_csv("chandigarh_clustered.csv")
        cluster_summary = (
            cluster_df.groupby("demand_cluster")
            .agg(
                Average_Passengers=("passenger_count", "mean"),
                Average_Hour=("hour", "mean"),
                Average_Occupancy=("occupancy_rate", "mean"),
                Records=("record_id", "count")
            )
            .reset_index()
            .rename(columns={
                "demand_cluster": "Cluster",
                "Average_Passengers": "Average Passengers",
                "Average_Hour": "Average Hour",
                "Average_Occupancy": "Average Occupancy",
                "Records": "Records"
            })
        )
        show_table(cluster_summary)

        sample = cluster_df.sample(
            min(1200, len(cluster_df)),
            random_state=42
        )
        fig = px.scatter(
            sample,
            x="hour",
            y="passenger_count",
            color="demand_cluster",
            hover_data=["route_id", "crowd_level"],
            labels={
                "hour": "Hour of Day",
                "passenger_count": "Passenger Count",
                "demand_cluster": "Cluster"
            },
            color_continuous_scale="Teal"
        )
        st.plotly_chart(style_chart(fig, 420), use_container_width=True)
    else:
        st.warning(
            "chandigarh_clustered.csv was not found. Run train_models.py first."
        )


# =========================================================
# 16. FOOTER
# =========================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:15px;color:#64748B;font-size:12px;">
  <span style="color:#22D3EE;font-weight:800;">TransitIQ</span>
  &nbsp; | &nbsp; Smart Public Transport Demand &amp; Crowd Prediction
  <br><br>
  MCA Machine Learning Project &nbsp; • &nbsp; Chandigarh
</div>
""", unsafe_allow_html=True)
