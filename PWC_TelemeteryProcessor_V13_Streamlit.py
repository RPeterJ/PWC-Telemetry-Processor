import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import json
import matplotlib.pyplot as plt
import pydeck as pdk  # For advanced maps

# --- Optional Imports ---
try:
    import fitparse
except ImportError:
    fitparse = None
try:
    import gpxpy
    import gpxpy.gpx
except ImportError:
    gpxpy = None


# --- CORE LOGIC (PARSERS & CALCULATIONS - UNCHANGED) ---
# (Parsers and calculate_telemetry function are the same as before, included here for completeness)
@st.cache_data
def parse_fit_file(uploaded_file):
    if not fitparse: raise ImportError("'fitparse' library not found.")
    fitfile = fitparse.FitFile(uploaded_file)
    records = list(fitfile.get_messages('record'))
    if not records: raise ValueError("No 'record' messages found in .fit file.")
    data, semicircles_to_degrees = [], 180 / 2 ** 31
    for r in records:
        vals = r.get_values()
        if all(k in vals for k in ['timestamp', 'speed', 'position_lat', 'position_long']):
            data.append({'timestamp': vals['timestamp'], 'speed_ms': vals['speed'],
                         'latitude': vals['position_lat'] * semicircles_to_degrees,
                         'longitude': vals['position_long'] * semicircles_to_degrees})
    if not data: raise ValueError("Could not extract valid data from .fit file.")
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['speed_kmh'] = df['speed_ms'] * 3.6
    return df


@st.cache_data
def parse_gpx_file(uploaded_file):
    if not gpxpy: raise ImportError("'gpxpy' library not found.")
    gpx_file_content = uploaded_file.getvalue().decode('utf-8')
    gpx = gpxpy.parse(gpx_file_content)
    data = []
    for track in gpx.tracks:
        for segment in track.segments:
            if not segment.points: continue
            segment.points[0].speed = 0.0
            for i in range(1, len(segment.points)):
                segment.points[i].speed = segment.points[i].speed_between(segment.points[i - 1])
            for point in segment.points:
                if point.time and point.speed is not None:
                    ts = point.time.replace(tzinfo=None)  # Make timezone naive for consistency
                    data.append({'timestamp': ts, 'speed_ms': point.speed, 'latitude': point.latitude,
                                 'longitude': point.longitude})
    if not data: raise ValueError("Could not extract valid data from .gpx file.")
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['speed_kmh'] = df['speed_ms'] * 3.6
    return df


def calculate_telemetry(data_df, ride_conditions, profile_data):
    rider_weight_kg, fuel_load, drive_mode, water_condition = ride_conditions
    rpm_model, fuel_model = profile_data['rpm_model'], profile_data['fuel_model']
    speed_data_kmh, rpm_data = np.array(rpm_model['speed_kmh']), np.array(rpm_model['rpm'], dtype=float)
    fuel_rpm_data, fuel_lph_data = np.array(fuel_model['rpm']), np.array(fuel_model['lph'])
    weight_factor = 1.0 + (rider_weight_kg - 80) * 0.001
    fuel_factors = {'Full': 1.02, 'Half': 1.01, 'Low': 1.0}
    water_factor = 1.03 if water_condition.lower() == 'rough' else 1.0
    drive_factors = {'Normal': 1.0, 'L-Mode': 0.85, 'No-Wake': 0.3}
    total_factor = weight_factor * fuel_factors[fuel_load] * water_factor * drive_factors[drive_mode]
    adjusted_rpm_data = rpm_data * total_factor
    rpm_interp_func = interp1d(speed_data_kmh, adjusted_rpm_data, kind='linear', fill_value="extrapolate")
    fuel_interp_func = interp1d(fuel_rpm_data, fuel_lph_data, kind='linear', fill_value="extrapolate")
    telemetry_df = data_df.copy()
    telemetry_df['rpm'] = rpm_interp_func(telemetry_df['speed_kmh']).round().astype(int)
    telemetry_df['rpm'] = telemetry_df['rpm'].clip(lower=rpm_data.min(), upper=rpm_data.max() * total_factor)
    telemetry_df['fuel_consumption_lph'] = fuel_interp_func(telemetry_df['rpm'])
    telemetry_df['fuel_consumption_lph'] = telemetry_df['fuel_consumption_lph'].clip(lower=0)
    telemetry_df['time_delta_s'] = telemetry_df['timestamp'].diff().dt.total_seconds().fillna(0)
    fuel_used_in_interval = telemetry_df['fuel_consumption_lph'] * (telemetry_df['time_delta_s'] / 3600)
    telemetry_df['cumulative_fuel_used_l'] = fuel_used_in_interval.cumsum()
    telemetry_df['distance_interval_km'] = telemetry_df['speed_kmh'] * (telemetry_df['time_delta_s'] / 3600)
    total_distance_km = telemetry_df['distance_interval_km'].sum()
    total_fuel_l = telemetry_df['cumulative_fuel_used_l'].iloc[-1] if not telemetry_df.empty else 0
    max_speed_kmh = telemetry_df['speed_kmh'].max()
    moving_df = telemetry_df[telemetry_df['speed_kmh'] > 5]
    avg_moving_speed_kmh = moving_df['speed_kmh'].mean() if not moving_df.empty else 0
    best_cruise_speed_kmh, best_cruise_economy_km_per_l = 0, 0
    try:
        eco_df = telemetry_df.copy()
        eco_df['fuel_used_interval_l'] = fuel_used_in_interval
        cruise_df = eco_df[(eco_df['speed_kmh'] >= 30) & (eco_df['fuel_used_interval_l'] > 0)].copy()
        if not cruise_df.empty:
            bins = np.arange(30, cruise_df['speed_kmh'].max() + 2, 2)
            cruise_df['speed_bin'] = pd.cut(cruise_df['speed_kmh'], bins=bins, right=False)
            economy_by_speed = cruise_df.groupby('speed_bin', observed=False).agg(
                total_dist_km=('distance_interval_km', 'sum'),
                total_fuel_l=('fuel_used_interval_l', 'sum')).reset_index()
            economy_by_speed['economy_km_per_l'] = economy_by_speed.apply(
                lambda row: row['total_dist_km'] / row['total_fuel_l'] if row['total_fuel_l'] > 0 else 0, axis=1)
            planing_economy_df = economy_by_speed[economy_by_speed['economy_km_per_l'] > 0].copy()
            if not planing_economy_df.empty:
                best_bin = planing_economy_df.loc[planing_economy_df['economy_km_per_l'].idxmax()]
                best_cruise_economy_km_per_l = best_bin['economy_km_per_l']
                best_cruise_speed_kmh = best_bin['speed_bin'].mid
    except Exception:
        pass
    summary_data = {"total_distance_km": total_distance_km, "total_fuel_l": total_fuel_l,
                    "max_speed_kmh": max_speed_kmh, "avg_moving_speed_kmh": avg_moving_speed_kmh,
                    "l_per_100km": (total_fuel_l / total_distance_km) * 100 if total_distance_km > 0 else 0,
                    "km_per_l": total_distance_km / total_fuel_l if total_fuel_l > 0 else 0,
                    "l_per_hour": total_fuel_l / (telemetry_df['time_delta_s'].sum() / 3600) if telemetry_df[
                                                                                                    'time_delta_s'].sum() > 0 else 0,
                    "best_cruise_speed_kmh": best_cruise_speed_kmh,
                    "best_cruise_economy_km_per_l": best_cruise_economy_km_per_l}
    telemetry_df.rename(
        columns={'speed_kmh': 'Speed (km/h)', 'rpm': 'Engine RPM (rpm)', 'fuel_consumption_lph': 'Fuel Rate (L/h)',
                 'cumulative_fuel_used_l': 'Fuel Used (L)'}, inplace=True)
    return telemetry_df, summary_data


# --- STREAMLIT UI APPLICATION ---

# VISUAL IMPROVEMENT: Set page config for a better look and feel
st.set_page_config(
    page_title="PWC Telemetry Processor",
    page_icon="🚤",  # Add an emoji
    layout="wide",
    initial_sidebar_state="expanded"
)

# VISUAL IMPROVEMENT: Use a consistent color palette
# Using Tableau's color palette which works well for distinct categories
COLOR_PALETTE = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
                 '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']


@st.cache_data
def load_profiles():
    try:
        with open("profiles.json", "r") as f:
            return json.load(f)
    except:
        st.error("Could not load 'profiles.json'. Make sure it's in the same directory.")
        return {}


profiles = load_profiles()

if not profiles: st.stop()

# --- SIDEBAR FOR INPUTS ---
# VISUAL IMPROVEMENT: Move all controls to a sidebar for a cleaner main page
with st.sidebar:
    st.header("⚙️ Configuration")

    uploaded_file = st.file_uploader("Upload a .gpx file", type=['gpx'])

    profile_list = sorted(profiles.keys())
    selected_profile_name = st.selectbox("Select PWC Profile", profile_list)

    st.subheader("Ride Conditions")
    default_weight = profiles[selected_profile_name].get('default_rider_weight', 80.0)
    rider_weight = st.number_input("Rider Weight (kg)", value=default_weight, min_value=30.0, max_value=200.0, step=1.0)
    fuel_load = st.selectbox("Fuel Load", ["Full", "Half", "Low"])
    drive_mode = st.selectbox("Drive Mode", ["Normal", "L-Mode", "No-Wake"])
    water_condition = st.selectbox("Water Condition", ["Calm", "Rough"])

# --- MAIN PAGE ---
st.title("🚤 PWC Telemetry Processor")
st.markdown("Analyze your PWC ride data. Upload a file and configure the ride in the sidebar to begin.")

if uploaded_file is not None:
    try:
        # Parsing (cached for speed)
        file_ext = "." + uploaded_file.name.split('.')[-1].lower()
        if file_ext == '.fit':
            raw_df = parse_fit_file(uploaded_file)
        elif file_ext == '.gpx':
            raw_df = parse_gpx_file(uploaded_file)
        else:
            raise ValueError("Unsupported file type.")

        # Calculation
        ride_conditions = (rider_weight, fuel_load, drive_mode, water_condition)
        selected_profile_data = profiles[selected_profile_name]
        calculated_df, summary_data = calculate_telemetry(raw_df[['timestamp', 'speed_kmh']], ride_conditions,
                                                          selected_profile_data)

        st.success(f"**{uploaded_file.name}** processed successfully!")

        # --- DISPLAY RIDE SUMMARY ---
        # VISUAL IMPROVEMENT: Use columns and st.metric for a dashboard look
        st.subheader("Ride Summary")
        best_cruise_text = "N/A"
        if summary_data['best_cruise_speed_kmh'] > 0:
            best_cruise_text = f"{summary_data['best_cruise_economy_km_per_l']:.2f} km/L @ {summary_data['best_cruise_speed_kmh']:.0f} km/h"

        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        sum_col1.metric("Total Distance", f"{summary_data['total_distance_km']:.2f} km")
        sum_col2.metric("Total Fuel Used", f"{summary_data['total_fuel_l']:.2f} L")
        sum_col3.metric("Max Speed", f"{summary_data['max_speed_kmh']:.1f} km/h")
        sum_col4.metric("Avg. Moving Speed", f"{summary_data['avg_moving_speed_kmh']:.1f} km/h")

        sum_col5, sum_col6, sum_col7 = st.columns(3)
        sum_col5.metric("Overall Economy", f"{summary_data['km_per_l']:.2f} km/L")
        sum_col6.metric("Overall Consumption", f"{summary_data['l_per_100km']:.2f} L/100km")
        sum_col7.metric("Best Planing Economy", best_cruise_text)

        # --- DISPLAY VISUALIZATIONS ---
        st.subheader("Visualizations")
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Speed & RPM", "📊 Engine Analysis", "⚖️ PWC Comparison", "🗺️ Map"])

        with tab1:
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(calculated_df['timestamp'], calculated_df['Speed (km/h)'], color=COLOR_PALETTE[0], label='Speed')
            ax1.set_xlabel('Time');
            ax1.set_ylabel('Speed (km/h)', color=COLOR_PALETTE[0]);
            ax1.tick_params(axis='y', labelcolor=COLOR_PALETTE[0])
            ax2 = ax1.twinx()
            ax2.plot(calculated_df['timestamp'], calculated_df['Engine RPM (rpm)'], color=COLOR_PALETTE[1],
                     linestyle='--', label='RPM')
            ax2.set_ylabel('Engine RPM (rpm)', color=COLOR_PALETTE[1]);
            ax2.tick_params(axis='y', labelcolor=COLOR_PALETTE[1])
            st.pyplot(fig)

        with tab2:
            fig, ax = plt.subplots(figsize=(10, 5))
            max_rpm = calculated_df['Engine RPM (rpm)'].max()
            bins = np.arange(0, max_rpm + 1000, 1000)
            labels = [f'{int(bins[i])}-{int(bins[i + 1])}' for i in range(len(bins) - 1)]
            calculated_df['rpm_bin'] = pd.cut(calculated_df['Engine RPM (rpm)'], bins=bins, labels=labels, right=False)
            time_in_rpm = calculated_df.groupby('rpm_bin', observed=False)['time_delta_s'].sum() / 60
            bars = ax.barh(time_in_rpm.index, time_in_rpm.values, color=COLOR_PALETTE[3])
            ax.set_xlabel('Time (Minutes)');
            ax.set_ylabel('RPM Range');
            ax.set_title('Time Spent in RPM Ranges')
            st.pyplot(fig)

        with tab3:
            # FIX: Use Session State to remember selections
            if 'visible_profiles' not in st.session_state:
                st.session_state.visible_profiles = sorted(profiles.keys())

            st.multiselect(
                'Select profiles to display:',
                options=sorted(profiles.keys()),
                key='visible_profiles'  # Link to the session state key
            )

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            speed_range_kmh = np.linspace(20, 120, 100)

            # Use a color cycler for consistent colors
            color_cycler = plt.cycler(color=COLOR_PALETTE)
            ax1.set_prop_cycle(color_cycler)
            ax2.set_prop_cycle(color_cycler)

            for profile_name in st.session_state.visible_profiles:
                profile_data = profiles[profile_name]
                rpm_model, fuel_model = profile_data['rpm_model'], profile_data['fuel_model']
                rpm_interp = interp1d(rpm_model['speed_kmh'], rpm_model['rpm'], kind='linear', fill_value="extrapolate")
                fuel_interp = interp1d(fuel_model['rpm'], fuel_model['lph'], kind='linear', fill_value="extrapolate")
                rpm_values = rpm_interp(speed_range_kmh)
                fuel_lph_values = np.clip(fuel_interp(rpm_values), a_min=0.1, a_max=None)
                economy_kml_values = speed_range_kmh / fuel_lph_values
                is_selected = (profile_name == selected_profile_name)
                line_width = 3.5 if is_selected else 1.5
                alpha = 1.0 if is_selected else 0.7
                zorder = 10 if is_selected else 5
                ax1.plot(speed_range_kmh, fuel_lph_values, label=profile_name, linewidth=line_width, alpha=alpha,
                         zorder=zorder)
                ax2.plot(speed_range_kmh, economy_kml_values, label=profile_name, linewidth=line_width, alpha=alpha,
                         zorder=zorder)

            ax1.set_ylabel('Fuel Usage (L/h)');
            ax1.set_title('Fuel Consumption vs. Speed');
            ax1.grid(True, linestyle='--');
            ax1.legend()
            ax2.set_xlabel('Speed (km/h)');
            ax2.set_ylabel('Economy (km/L)');
            ax2.set_title('Fuel Economy vs. Speed');
            ax2.grid(True, linestyle='--');
            ax2.legend()
            st.pyplot(fig)

        with tab4:
            # ENHANCEMENT: Use PyDeck for a better map with a path
            map_data = raw_df[['latitude', 'longitude']].dropna()

            initial_view_state = pdk.ViewState(
                latitude=map_data['latitude'].mean(),
                longitude=map_data['longitude'].mean(),
                zoom=11,
                pitch=50,
            )

            path_layer = pdk.Layer(
                'PathLayer',
                data=map_data,
                get_path='[longitude, latitude]',
                get_color='[242, 43, 43]',  # A bright red color
                width_min_pixels=2,
            )

            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/satellite-streets-v11',
                initial_view_state=initial_view_state,
                layers=[path_layer],
            ))

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

else:
    st.info("Waiting for a file to be uploaded...")