import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
import plotly.graph_objects as go
from datetime import datetime
import requests
from google.transit import gtfs_realtime_pb2

# Page config
st.set_page_config(
    page_title="TransLink Delay Predictor",
    page_icon="🚌",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = keras.models.load_model("models/delay_prediction_final.h5")
        scaler = pickle.load(open("models/scaler_final.pkl", 'rb'))
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load reference data
@st.cache_data
def load_reference_data():
    try:
        df = pd.read_csv("data/delays_with_features.csv")
        route_stats = df.groupby('route_id')['arrival_delay'].mean().to_dict()
        stop_stats = df.groupby('stop_id')['arrival_delay'].mean().to_dict()
        return route_stats, stop_stats
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {}, {}

# Get real-time trip updates from TransLink
@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_realtime_data():
    """Fetch real-time transit data from TransLink"""
    try:
        # Correct TransLink GTFS-RT API endpoint (v3)
        url = "https://gtfsapi.translink.ca/v3/gtfsrealtime"
        
        params = {
            'apikey': st.secrets["TRANSLINK_API_KEY"]
        }
        
        headers = {
            'accept': 'application/x-google-protobuf'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        
        # Debug information
        st.sidebar.caption(f"API Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                feed = gtfs_realtime_pb2.FeedMessage()
                feed.ParseFromString(response.content)
                
                realtime_delays = []
                for entity in feed.entity:
                    if entity.HasField('trip_update'):
                        trip = entity.trip_update.trip
                        for stop_update in entity.trip_update.stop_time_update:
                            delay = None
                            if stop_update.HasField('arrival') and stop_update.arrival.HasField('delay'):
                                delay = stop_update.arrival.delay
                            elif stop_update.HasField('departure') and stop_update.departure.HasField('delay'):
                                delay = stop_update.departure.delay
                            
                            if delay is not None:
                                delay_data = {
                                    'trip_id': trip.trip_id if trip.HasField('trip_id') else 'unknown',
                                    'route_id': trip.route_id if trip.HasField('route_id') else 'unknown',
                                    'stop_id': stop_update.stop_id if stop_update.HasField('stop_id') else 'unknown',
                                    'stop_sequence': stop_update.stop_sequence if stop_update.HasField('stop_sequence') else 0,
                                    'delay': delay
                                }
                                realtime_delays.append(delay_data)
                
                if realtime_delays:
                    return pd.DataFrame(realtime_delays)
                else:
                    st.sidebar.warning("No delay data in feed")
                    return None
                    
            except Exception as parse_error:
                st.sidebar.error(f"Parse error: {str(parse_error)}")
                return None
        
        elif response.status_code == 401:
            st.sidebar.error("❌ Invalid API Key")
            return None
        elif response.status_code == 403:
            st.sidebar.error("❌ Access Forbidden")
            return None
        else:
            st.sidebar.error(f"❌ API Error: {response.status_code}")
            return None
            
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        return None

# Get vehicle positions
@st.cache_data(ttl=30)
def get_vehicle_positions():
    """Fetch real-time vehicle positions from TransLink"""
    try:
        url = "https://gtfsapi.translink.ca/v3/gtfsposition"
        params = {'apikey': st.secrets["TRANSLINK_API_KEY"]}
        headers = {'accept': 'application/x-google-protobuf'}
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        
        if response.status_code == 200:
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(response.content)
            
            positions = []
            for entity in feed.entity:
                if entity.HasField('vehicle'):
                    vehicle = entity.vehicle
                    if vehicle.HasField('position'):
                        pos_data = {
                            'vehicle_id': vehicle.vehicle.id if vehicle.HasField('vehicle') else 'unknown',
                            'trip_id': vehicle.trip.trip_id if vehicle.HasField('trip') else 'unknown',
                            'route_id': vehicle.trip.route_id if vehicle.HasField('trip') else 'unknown',
                            'latitude': vehicle.position.latitude,
                            'longitude': vehicle.position.longitude,
                            'bearing': vehicle.position.bearing if vehicle.position.HasField('bearing') else None,
                            'speed': vehicle.position.speed if vehicle.position.HasField('speed') else None
                        }
                        positions.append(pos_data)
            
            return pd.DataFrame(positions) if positions else None
        else:
            return None
            
    except Exception as e:
        return None

# Get service alerts
@st.cache_data(ttl=60)
def get_service_alerts():
    """Fetch service alerts from TransLink"""
    try:
        url = "https://gtfsapi.translink.ca/v3/gtfsalerts"
        params = {'apikey': st.secrets["TRANSLINK_API_KEY"]}
        headers = {'accept': 'application/x-google-protobuf'}
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        
        if response.status_code == 200:
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(response.content)
            
            alerts = []
            for entity in feed.entity:
                if entity.HasField('alert'):
                    alert = entity.alert
                    alert_data = {
                        'header': alert.header_text.translation[0].text if alert.header_text.translation else 'No header',
                        'description': alert.description_text.translation[0].text if alert.description_text.translation else 'No description'
                    }
                    alerts.append(alert_data)
            
            return alerts if alerts else None
        else:
            return None
            
    except Exception as e:
        return None

model, scaler = load_model_and_scaler()
route_stats, stop_stats = load_reference_data()

if model is None:
    st.stop()

# Header
st.title("🚌 TransLink Transit Delay Predictor")
st.markdown("---")

# Sidebar
st.sidebar.header("📝 Input Parameters")

# Real-time mode toggle
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔴 Real-Time Mode")

live_mode = st.sidebar.checkbox(
    "Enable Live Data", 
    help="Fetch real-time delays from TransLink"
)

realtime_df = None
if live_mode:
    # Test API button
    if st.sidebar.button("🧪 Test API Connection"):
        with st.spinner("Testing TransLink API..."):
            try:
                # Test Trip Updates
                test_url = "https://gtfsapi.translink.ca/v3/gtfsrealtime"
                test_response = requests.get(
                    test_url, 
                    params={'apikey': st.secrets["TRANSLINK_API_KEY"]},
                    timeout=10
                )
                
                if test_response.status_code == 200:
                    st.sidebar.success(f"✓ Trip Updates API Working!")
                    st.sidebar.caption(f"Data size: {len(test_response.content)} bytes")
                else:
                    st.sidebar.error(f"Trip Updates failed: {test_response.status_code}")
                
                # Test Positions
                pos_url = "https://gtfsapi.translink.ca/v3/gtfsposition"
                pos_response = requests.get(
                    pos_url,
                    params={'apikey': st.secrets["TRANSLINK_API_KEY"]},
                    timeout=10
                )
                
                if pos_response.status_code == 200:
                    st.sidebar.success(f"✓ Vehicle Positions API Working!")
                else:
                    st.sidebar.warning(f"Positions: {pos_response.status_code}")
                    
            except Exception as e:
                st.sidebar.error(f"Test failed: {str(e)}")
    
    st.sidebar.markdown("---")
    
    with st.spinner("Fetching real-time data..."):
        realtime_df = get_realtime_data()
    
    if realtime_df is not None and len(realtime_df) > 0:
        st.sidebar.success(f"✓ Connected - {len(realtime_df)} active updates")
        
        # Show last update time
        current_time = datetime.now().strftime("%H:%M:%S")
        st.sidebar.caption(f"Last updated: {current_time}")
        
        # Auto-refresh option
        if st.sidebar.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Display some real-time stats
        st.sidebar.markdown("#### Live Statistics")
        avg_delay = realtime_df['delay'].mean() / 60
        max_delay = realtime_df['delay'].max() / 60
        min_delay = realtime_df['delay'].min() / 60
        st.sidebar.metric("Avg System Delay", f"{avg_delay:.1f} min")
        st.sidebar.metric("Max Delay", f"{max_delay:.1f} min")
        st.sidebar.metric("Min Delay", f"{min_delay:.1f} min")
    else:
        st.sidebar.warning("No real-time data available")
else:
    st.sidebar.info("Live mode disabled - using manual inputs")

st.sidebar.markdown("---")

# Input method selection
input_method = st.sidebar.radio(
    "Input Method",
    ["Quick Mode (Common Values)", "Advanced Mode (Exact IDs)", "Live Trip Selection"],
    help="Choose how to input route and stop information"
)

# Initialize variables
route_id = None
stop_id = None
stop_sequence = None
relative_position = None
upstream_delay = 0
transit_mode = None
route_number = None
stop_location = None

if input_method == "Live Trip Selection" and live_mode and realtime_df is not None:
    st.sidebar.markdown("### 🚏 Select Active Trip")
    
    active_routes = sorted(realtime_df['route_id'].unique())
    route_id = st.sidebar.selectbox(
        "Active Route",
        active_routes,
        help="Routes currently in service"
    )
    
    # Filter stops for selected route
    route_data = realtime_df[realtime_df['route_id'] == route_id]
    if len(route_data) > 0:
        stop_id = st.sidebar.selectbox(
            "Active Stop",
            sorted(route_data['stop_id'].unique())
        )
        
        # Get actual delay for this stop
        stop_data = route_data[route_data['stop_id'] == stop_id]
        if len(stop_data) > 0:
            actual_delay = stop_data.iloc[0]['delay']
            stop_sequence = int(stop_data.iloc[0]['stop_sequence'])
            upstream_delay = actual_delay
            
            st.sidebar.info(f"Current delay: {actual_delay/60:.1f} min")
            st.sidebar.info(f"Stop sequence: {stop_sequence}")
            
            # Estimate relative position
            max_seq = route_data['stop_sequence'].max()
            relative_position = stop_sequence / max_seq if max_seq > 0 else 0.5

elif input_method == "Quick Mode (Common Values)":
    st.sidebar.markdown("### 🚏 Transit Information")
    
    # Transit mode selection
    transit_mode = st.sidebar.selectbox(
        "Transit Mode",
        ["Bus", "SkyTrain", "SeaBus", "West Coast Express"],
        help="Select type of transit"
    )
    
    # Route options based on mode
    if transit_mode == "Bus":
        route_options = ["99 B-Line", "R4 41st Ave", "N15 Cambie", "3 Main St", "20 Victoria", "Other Bus"]
        route_mapping = {
            "99 B-Line": 6600,
            "R4 41st Ave": 6700,
            "N15 Cambie": 6800,
            "3 Main St": 6900,
            "20 Victoria": 7000,
            "Other Bus": 7100
        }
    elif transit_mode == "SkyTrain":
        route_options = ["Expo Line", "Millennium Line", "Canada Line"]
        route_mapping = {
            "Expo Line": 9800,
            "Millennium Line": 9900,
            "Canada Line": 9950
        }
    elif transit_mode == "SeaBus":
        route_options = ["SeaBus - Waterfront to Lonsdale"]
        route_mapping = {
            "SeaBus - Waterfront to Lonsdale": 9990
        }
    else:  # West Coast Express
        route_options = ["WCE - Waterfront to Mission"]
        route_mapping = {
            "WCE - Waterfront to Mission": 9995
        }
    
    route_number = st.sidebar.selectbox(
        f"{transit_mode} Route",
        route_options,
        help=f"Select a {transit_mode} route"
    )
    
    route_id = route_mapping.get(route_number, 7000)
    
    # Stop description (adjust based on mode)
    if transit_mode == "SkyTrain":
        stop_location = st.sidebar.selectbox(
            "Station Location",
            ["Starting terminal", "Early stations", "Mid-line stations", "Later stations", "End terminal"],
            help="Approximate station location along the line"
        )
        position_mapping = {
            "Starting terminal": (1, 0.0),
            "Early stations": (5, 0.25),
            "Mid-line stations": (10, 0.5),
            "Later stations": (15, 0.75),
            "End terminal": (20, 1.0)
        }
    elif transit_mode == "SeaBus":
        stop_location = st.sidebar.selectbox(
            "Terminal",
            ["Waterfront", "Lonsdale Quay"],
            help="Select terminal"
        )
        position_mapping = {
            "Waterfront": (1, 0.0),
            "Lonsdale Quay": (2, 1.0)
        }
    else:
        stop_location = st.sidebar.selectbox(
            "Stop Location",
            ["Beginning of route", "Quarter way", "Halfway", "Three-quarters", "End of route"],
            help="Approximate location along the route"
        )
        position_mapping = {
            "Beginning of route": (5, 0.1),
            "Quarter way": (15, 0.25),
            "Halfway": (25, 0.5),
            "Three-quarters": (35, 0.75),
            "End of route": (45, 0.9)
        }
    
    stop_sequence, relative_position = position_mapping[stop_location]
    stop_id = 1000 + stop_sequence
    
    # Upstream delay input
    st.sidebar.markdown("### 📊 Current Conditions")
    delay_source = st.sidebar.radio(
        "Upstream Delay",
        ["No delay", "Minor delay (1-2 min)", "Moderate delay (3-5 min)", "Major delay (5+ min)"]
    )
    
    delay_mapping = {
        "No delay": 0,
        "Minor delay (1-2 min)": 90,
        "Moderate delay (3-5 min)": 240,
        "Major delay (5+ min)": 360
    }
    upstream_delay = delay_mapping[delay_source]
    
else:  # Advanced Mode
    st.sidebar.markdown("### 🚏 Route & Stop IDs")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        route_id = st.number_input("Route ID", 6000, 10000, 6613, help="Enter exact route ID")
    with col2:
        stop_id = st.number_input("Stop ID", 1, 15000, 155, help="Enter exact stop ID")
    
    stop_sequence = st.sidebar.number_input("Stop Sequence", 1, 100, 10, help="Position in route")
    relative_position = st.sidebar.slider("Trip Progress", 0.0, 1.0, 0.5, help="0=start, 1=end")
    
    # Upstream delay
    st.sidebar.markdown("### 📊 Current Conditions")
    upstream_delay = st.sidebar.slider("Upstream Delay (seconds)", -300, 600, 0)

# Time inputs
st.sidebar.markdown("### ⏰ Time Settings")
current_time = datetime.now()
col3, col4 = st.sidebar.columns(2)
with col3:
    hour = st.slider("Hour", 0, 23, current_time.hour)
with col4:
    day_of_week = st.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        index=current_time.weekday()
    )
    day_of_week_num = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)

# Weather
st.sidebar.markdown("### 🌤️ Weather")
weather_condition = st.sidebar.selectbox(
    "Weather Condition",
    ["Clear", "Light Rain", "Heavy Rain", "Snow", "Extreme Cold"]
)

weather_mapping = {
    "Clear": (15, 0),
    "Light Rain": (12, 1.5),
    "Heavy Rain": (10, 5.0),
    "Snow": (2, 3.0),
    "Extreme Cold": (-5, 0)
}
temperature, precipitation = weather_mapping[weather_condition]

# Calculate derived features
route_avg = route_stats.get(route_id, 0) if route_id else 0
stop_avg = stop_stats.get(stop_id, 0) if stop_id else 0

is_weekend = 1 if day_of_week_num >= 5 else 0
is_rush_hour = 1 if hour in [7, 8, 16, 17, 18] else 0

# Create feature vector
features = {
    'upstream_delay': upstream_delay,
    'route_avg_delay': route_avg,
    'delay_change': upstream_delay * 0.1,
    'upstream_delay_variance': (upstream_delay * 0.1) ** 2,
    'delay_acceleration': upstream_delay * 0.01,
    'stop_current_congestion': stop_avg,
    'route_delay_percentile': 0.5,
    'delay_momentum': upstream_delay * stop_sequence if stop_sequence else 0,
    'route_hour_avg_delay': route_avg,
    'direction_avg_delay': route_avg,
    'peak_hour_intensity': route_avg if is_rush_hour else route_avg * 0.5,
    'relative_position': relative_position if relative_position is not None else 0.5,
    'cumulative_delay': upstream_delay * (relative_position if relative_position is not None else 0.5),
    'temperature': temperature,
    'precipitation': precipitation,
    'stop_sequence': stop_sequence if stop_sequence else 10
}

# Service alerts
if live_mode:
    alerts = get_service_alerts()
    if alerts:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚠️ Service Alerts")
        for i, alert in enumerate(alerts[:3]):  # Show top 3 alerts
            with st.sidebar.expander(f"Alert {i+1}: {alert['header'][:30]}..."):
                st.write(alert['description'])

# Real-time system dashboard
if live_mode and realtime_df is not None and len(realtime_df) > 0:
    st.markdown("## 🔴 Real-Time System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trips = len(realtime_df['trip_id'].unique())
        st.metric("Active Trips", total_trips)
    
    with col2:
        delayed = (realtime_df['delay'] > 300).sum()  # More than 5 min
        st.metric("Delayed Services", delayed)
    
    with col3:
        early = (realtime_df['delay'] < -60).sum()  # More than 1 min early
        st.metric("Early Services", early)
    
    with col4:
        on_time = ((realtime_df['delay'] >= -60) & (realtime_df['delay'] <= 300)).sum()
        st.metric("On-Time Services", on_time)
    
    # Show most delayed routes
    st.markdown("### 🚨 Most Delayed Routes")
    route_delays = realtime_df.groupby('route_id')['delay'].mean().sort_values(ascending=False).head(5)
    delay_df = pd.DataFrame({
        'Route': route_delays.index,
        'Average Delay (min)': (route_delays.values / 60).round(2)
    })
    st.dataframe(delay_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")

# Main prediction area
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("## 🎯 Prediction Results")
    
    if st.button("🚀 Predict Delay", type="primary", use_container_width=True):
        with st.spinner("Analyzing transit patterns..."):
            # Prepare and predict
            feature_df = pd.DataFrame([features])
            feature_scaled = scaler.transform(feature_df)
            prediction_seconds = model.predict(feature_scaled, verbose=0)[0][0]
            prediction_minutes = prediction_seconds / 60
            
            # Display result
            if prediction_minutes < -1:
                status = "🟢 Early"
            elif prediction_minutes > 2:
                status = "🔴 Delayed"
            else:
                status = "🟡 On Time"
            
            st.markdown(f"### {status}")
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Predicted Delay", f"{prediction_minutes:.2f} min")
            with col_b:
                st.metric("In Seconds", f"{prediction_seconds:.0f} sec")
            with col_c:
                confidence = max(70, 100 - abs(prediction_minutes) * 10)
                st.metric("Confidence", f"{confidence:.0f}%")
            
            st.progress(min(confidence / 100, 1.0))
            
            # Key factors
            st.markdown("#### 📊 Contributing Factors")
            
            if input_method == "Live Trip Selection":
                factor_values = [
                    f"{upstream_delay/60:.1f} min",
                    weather_condition,
                    f"Stop #{stop_sequence}",
                    f"{hour}:00 - {'Rush Hour' if is_rush_hour else 'Off-Peak'}"
                ]
            elif input_method == "Quick Mode (Common Values)":
                factor_values = [
                    delay_source if 'delay_source' in locals() else "N/A",
                    weather_condition,
                    stop_location if stop_location else "N/A",
                    f"{hour}:00 - {'Rush Hour' if is_rush_hour else 'Off-Peak'}"
                ]
            else:
                factor_values = [
                    f"{upstream_delay/60:.1f} min",
                    weather_condition,
                    f"Sequence {stop_sequence}",
                    f"{hour}:00 - {'Rush Hour' if is_rush_hour else 'Off-Peak'}"
                ]
            
            factors = pd.DataFrame({
                'Factor': ['Upstream Delay', 'Weather Conditions', 'Stop Location', 'Time of Day'],
                'Value': factor_values,
                'Impact': ['High', 'Medium', 'Medium', 'High' if is_rush_hour else 'Low']
            })
            st.dataframe(factors, use_container_width=True, hide_index=True)

with col2:
    st.markdown("## 📍 Trip Info")
    if input_method == "Quick Mode (Common Values)":
        st.metric("Mode", transit_mode if transit_mode else "N/A")
        st.metric("Route", route_number if route_number else "N/A")
        if transit_mode == "SkyTrain":
            st.metric("Station", stop_location if stop_location else "N/A")
        elif transit_mode == "SeaBus":
            st.metric("Terminal", stop_location if stop_location else "N/A")
        else:
            st.metric("Location", stop_location if stop_location else "N/A")
    elif input_method == "Live Trip Selection":
        st.metric("Route ID", route_id if route_id else "N/A")
        st.metric("Stop ID", stop_id if stop_id else "N/A")
        st.metric("Live Data", "Active")
    else:
        st.metric("Route ID", route_id if route_id else "N/A")
        st.metric("Stop ID", stop_id if stop_id else "N/A")
    
    st.metric("Stop #", stop_sequence if stop_sequence else "N/A")
    st.metric("Time", f"{hour}:00")
    st.metric("Day", day_of_week)

with col3:
    st.markdown("## 🌤️ Conditions")
    st.metric("Weather", weather_condition)
    st.metric("Temp", f"{temperature}°C")
    st.metric("Rain", f"{precipitation} mm")
    
    weather_impact = "High" if precipitation > 2 or temperature < 0 else "Moderate" if precipitation > 0 else "Low"
    st.metric("Weather Impact", weather_impact)
    
    rush_status = "Yes" if is_rush_hour else "No"
    st.metric("Rush Hour", rush_status)

# Model performance
st.markdown("---")
st.markdown("## 📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("MAE", "0.144 min", "37% better", delta_color="inverse")
with col2:
    st.metric("RMSE", "0.712 min", "75% improvement", delta_color="inverse")
with col3:
    st.metric("R² Score", "0.978", "97.8% explained")
with col4:
    st.metric("Accuracy", "98.1%", "within 1 min")

# Feature importance visualization
st.markdown("### 🔍 Model Feature Importance")

feature_importance = {
    'upstream_delay': 0.536,
    'cumulative_delay': 0.480,
    'delay_change': 0.480,
    'delay_momentum': 0.386,
    'route_hour_avg_delay': 0.367,
    'stop_current_congestion': 0.346,
    'delay_acceleration': 0.282,
    'direction_avg_delay': 0.273,
    'route_avg_delay': 0.252,
    'peak_hour_intensity': 0.131
}

fig = go.Figure(go.Bar(
    x=list(feature_importance.values()),
    y=list(feature_importance.keys()),
    orientation='h',
    marker=dict(
        color=list(feature_importance.values()), 
        colorscale='Blues',
        colorbar=dict(title="Correlation")
    )
))
fig.update_layout(
    title="Top 10 Features by Correlation with Delay",
    xaxis_title="Correlation Coefficient",
    yaxis_title="Feature",
    height=400,
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# About section
st.markdown("---")
with st.expander("ℹ️ About This Model"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Dataset**
        - 4.78M records from TransLink Vancouver
        - Data period: Nov 20-26, 2025
        - 47,757 unique trips
        - 222 routes, 8,341 stops
        - Includes: Buses, SkyTrain, SeaBus, WCE
        
        **Model Architecture**
        - Deep Neural Network (DNN)
        - Layers: 128 → 64 → 32 units
        - Activation: ReLU
        - Regularization: Batch Normalization + 10% Dropout
        """)
    
    with col2:
        st.markdown("""
        **Features (16 total)**
        - Delay propagation metrics
        - Route/stop patterns
        - Temporal features
        - Weather conditions
        
        **Performance**
        - Target MAE: 0.230 minutes
        - Achieved: 0.144 minutes (37% better)
        - 98.1% predictions within 1 minute
        - R² score: 0.978
        
        **Real-Time Integration**
        - Live GTFS-RT feed from TransLink
        - 30-second data refresh rate
        - Active trip monitoring
        - Vehicle positions & service alerts
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Streamlit | Powered by TensorFlow & Deep Learning</p>
    <p>Northeastern University | Data Analytics Engineering</p>
</div>
""", unsafe_allow_html=True)
