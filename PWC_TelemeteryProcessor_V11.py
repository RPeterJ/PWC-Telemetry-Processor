# Version 11: Includes processing graph for times in RPM ranges.
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledFrame
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
import json
import threading
from datetime import datetime
import sys


# --- PORTABLE FFMPEG SETUP ---
def setup_ffmpeg_path():
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_bin_path = os.path.join(application_path, "bin")
    if os.path.isdir(ffmpeg_bin_path):
        os.environ['PATH'] = ffmpeg_bin_path + os.pathsep + os.environ['PATH']


setup_ffmpeg_path()

# --- Optional Imports for Visualization ---
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    matplotlib_available = True
except ImportError:
    matplotlib_available = False
try:
    import geopandas as gpd
    import contextily as ctx

    map_viz_available = True
except ImportError:
    map_viz_available = False

# --- File Parsing Functions ---
try:
    import fitparse
except ImportError:
    fitparse = None
try:
    from gpmf.io import extract_gpmf_stream
    from gpmf.gps import extract_gps_blocks, parse_gps_block

    gpmf_parser_available = True
except ImportError:
    gpmf_parser_available = False
try:
    import gpxpy
    import gpxpy.gpx
except ImportError:
    gpxpy = None


def parse_fit_file(file_path):
    if not fitparse: raise ImportError("'fitparse' library not found. Please run: pip install fitparse")
    fitfile = fitparse.FitFile(file_path)
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
    df = pd.DataFrame(data);
    df['timestamp'] = pd.to_datetime(df['timestamp']);
    df['speed_kmh'] = df['speed_ms'] * 3.6
    return df


def parse_gopro_file(file_path):
    if not gpmf_parser_available: raise ImportError("'gpmf' library not found. Please run: pip install gpmf")
    gpmf_stream = extract_gpmf_stream(file_path)
    if not gpmf_stream: raise ValueError("Could not extract GPMF stream.")
    gps_blocks = extract_gps_blocks(gpmf_stream)
    if not gps_blocks: raise ValueError("No GPS data blocks found.")
    data = []
    for block in gps_blocks:
        parsed_block = parse_gps_block(block)
        num_samples = len(parsed_block.latitude)
        if num_samples == 0 or parsed_block.precision == 0: continue
        frequency = parsed_block.fix / parsed_block.precision
        if frequency == 0: continue
        time_delta_s = 1.0 / frequency
        start_timestamp = pd.to_datetime(parsed_block.timestamp, utc=True)
        time_offsets = np.arange(num_samples) * pd.to_timedelta(time_delta_s, unit='s')
        for i in range(num_samples):
            data.append({'timestamp': start_timestamp + time_offsets[i], 'speed_ms': parsed_block.speed_2d[i],
                         'latitude': parsed_block.latitude[i], 'longitude': parsed_block.longitude[i]})
    if not data: raise ValueError("Found no data points in GPS blocks.")
    df = pd.DataFrame(data);
    df['speed_kmh'] = df['speed_ms'] * 3.6;
    df.sort_values(by='timestamp', inplace=True)
    return df


def parse_gpx_file(file_path):
    if not gpxpy: raise ImportError("'gpxpy' library not found. Please run: pip install gpxpy")
    data = []
    with open(file_path, 'r', encoding='utf-8') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    for track in gpx.tracks:
        for segment in track.segments:
            if not segment.points: continue
            segment.points[0].speed = 0.0
            for i in range(1, len(segment.points)):
                segment.points[i].speed = segment.points[i].speed_between(segment.points[i - 1])
            for point in segment.points:
                if point.time and point.speed is not None:
                    ts = point.time if point.time.tzinfo else point.time.replace(
                        tzinfo=datetime.now().astimezone().tzinfo)
                    data.append({'timestamp': ts, 'speed_ms': point.speed, 'latitude': point.latitude,
                                 'longitude': point.longitude})
    if not data: raise ValueError("Could not extract valid data from .gpx file.")
    df = pd.DataFrame(data);
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True);
    df['speed_kmh'] = df['speed_ms'] * 3.6
    return df


# --- Calculation Engine ---
def calculate_telemetry(data_df, ride_conditions, profile_data):
    # ... (This function is correct and unchanged)
    rider_weight_kg, fuel_load, drive_mode, water_condition = ride_conditions
    engine_size = profile_data.get("engine_size", "1.8L NA")
    is_supercharged = profile_data.get("is_supercharged", False)
    rpm_model, fuel_model = profile_data['rpm_model'], profile_data['fuel_model']
    speed_data_kmh, rpm_data = np.array(rpm_model['speed_kmh']), np.array(rpm_model['rpm'], dtype=float)
    fuel_rpm_data, fuel_lph_data = np.array(fuel_model['rpm']), np.array(fuel_model['lph'])
    engine_factor = 1.2 if is_supercharged else 1.0
    if "1.9" in engine_size: engine_factor *= 1.05
    if "1.6" in engine_size: engine_factor *= 0.95
    weight_factor = 1.0 + (rider_weight_kg - 80) * 0.001
    fuel_factors = {'Full': 1.02, 'Half': 1.01, 'Low': 1.0}
    water_factor = 1.03 if water_condition.lower() == 'rough' else 1.0
    drive_factors = {'Normal': 1.0, 'L-Mode': 0.85, 'No-Wake': 0.3}
    total_factor = engine_factor * weight_factor * fuel_factors[fuel_load] * water_factor * drive_factors[drive_mode]
    adjusted_rpm_data = rpm_data * total_factor
    rpm_interp_func = interp1d(speed_data_kmh, adjusted_rpm_data, kind='cubic', fill_value="extrapolate")
    fuel_interp_func = interp1d(fuel_rpm_data, fuel_lph_data, kind='cubic', fill_value="extrapolate")
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


# --- Profile Manager (unchanged) ---
class ProfileManager(ttk.Toplevel):
    def __init__(self, master, profiles, callback):
        super().__init__(master=master)
        # ... (rest of this class is correct and unchanged)
        self.title("Profile Manager")
        self.geometry("900x800")
        self.transient(master)
        self.grab_set()
        self.profiles = profiles
        self.callback = callback
        self.current_profile_name = tk.StringVar(value=list(self.profiles.keys())[0])
        self.description_var = tk.StringVar()
        self.engine_size_var = tk.StringVar()
        self.default_weight_var = tk.DoubleVar()
        self.is_supercharged_var = tk.BooleanVar()
        self.rpm_entries = []
        self.fuel_entries = []
        self._create_widgets()
        self.load_profile_data()

    def _create_widgets(self):
        top_frame = ttk.Frame(self, padding=10)
        top_frame.pack(fill=X, pady=(0, 10))
        ttk.Label(top_frame, text="Select Profile:").pack(side=LEFT, padx=5)
        self.profile_menu = ttk.Combobox(top_frame, textvariable=self.current_profile_name,
                                         values=list(self.profiles.keys()), state="readonly")
        self.profile_menu.pack(side=LEFT, expand=True, fill=X, padx=5)
        self.profile_menu.bind("<<ComboboxSelected>>", lambda e: self.load_profile_data())
        add_btn = ttk.Button(top_frame, text="New", command=self.add_profile, bootstyle="success")
        add_btn.pack(side=LEFT, padx=5)
        delete_btn = ttk.Button(top_frame, text="Delete", command=self.delete_profile, bootstyle="danger")
        delete_btn.pack(side=LEFT, padx=5)
        scroll_frame = ScrolledFrame(self, autohide=True)
        scroll_frame.pack(fill=BOTH, expand=YES, padx=10)
        meta_frame = ttk.Labelframe(scroll_frame, text="Profile Details", padding=15)
        meta_frame.pack(fill=X, pady=5)
        meta_fields = {"Description:": self.description_var, "Engine Size:": self.engine_size_var,
                       "Default Rider Weight (kg):": self.default_weight_var}
        for i, (label, var) in enumerate(meta_fields.items()):
            ttk.Label(meta_frame, text=label).grid(row=i, column=0, sticky=W, padx=5, pady=2)
            ttk.Entry(meta_frame, textvariable=var, width=40).grid(row=i, column=1, sticky=EW, padx=5, pady=2)
        ttk.Checkbutton(meta_frame, text="Supercharged", variable=self.is_supercharged_var).grid(row=len(meta_fields),
                                                                                                 column=1, sticky=W,
                                                                                                 padx=5, pady=5)
        meta_frame.columnconfigure(1, weight=1)
        rpm_frame = ttk.Labelframe(scroll_frame, text="RPM Model (Speed vs RPM)", padding=15)
        rpm_frame.pack(fill=X, pady=10)
        self.create_model_grid(rpm_frame, self.rpm_entries, ('Speed (km/h)', 'RPM'))
        fuel_frame = ttk.Labelframe(scroll_frame, text="Fuel Consumption Model (RPM vs L/h)", padding=15)
        fuel_frame.pack(fill=X, pady=10)
        self.create_model_grid(fuel_frame, self.fuel_entries, ('RPM', 'Liters/Hour'))
        save_button = ttk.Button(self, text="Save All Profiles and Close", command=self.save_and_close,
                                 bootstyle="primary")
        save_button.pack(pady=10)

    def create_model_grid(self, parent, entry_list, headers):
        ttk.Label(parent, text=headers[0], font="-weight bold").grid(row=0, column=0, padx=5)
        ttk.Label(parent, text=headers[1], font="-weight bold").grid(row=0, column=1, padx=5)
        profile = self.profiles[self.current_profile_name.get()]
        model_key = 'rpm_model' if headers[0].startswith('Speed') else 'fuel_model'
        num_points = len(profile[model_key][list(profile[model_key].keys())[0]])
        entry_list.clear()
        for i in range(num_points):
            entry1, entry2 = ttk.Entry(parent, width=10), ttk.Entry(parent, width=10)
            entry1.grid(row=i + 1, column=0, padx=5, pady=2)
            entry2.grid(row=i + 1, column=1, padx=5, pady=2)
            entry_list.append((entry1, entry2))

    def load_profile_data(self):
        profile = self.profiles.get(self.current_profile_name.get())
        if not profile: return
        self.description_var.set(profile.get('description', ''))
        self.engine_size_var.set(profile.get('engine_size', ''))
        self.default_weight_var.set(profile.get('default_rider_weight', 80))
        self.is_supercharged_var.set(profile.get('is_supercharged', False))
        rpm_model = profile['rpm_model']
        for i, (val1, val2) in enumerate(zip(rpm_model['speed_kmh'], rpm_model['rpm'])):
            if i < len(self.rpm_entries):
                self.rpm_entries[i][0].delete(0);
                self.rpm_entries[i][0].insert(0, str(val1))
                self.rpm_entries[i][1].delete(0);
                self.rpm_entries[i][1].insert(0, str(val2))
        fuel_model = profile['fuel_model']
        for i, (val1, val2) in enumerate(zip(fuel_model['rpm'], fuel_model['lph'])):
            if i < len(self.fuel_entries):
                self.fuel_entries[i][0].delete(0);
                self.fuel_entries[i][0].insert(0, str(val1))
                self.fuel_entries[i][1].delete(0);
                self.fuel_entries[i][1].insert(0, str(val2))

    def save_current_profile(self):
        profile_name = self.current_profile_name.get()
        if not profile_name: return
        profile = self.profiles.get(profile_name, {})
        profile['description'] = self.description_var.get()
        profile['engine_size'] = self.engine_size_var.get()
        profile['default_rider_weight'] = self.default_weight_var.get()
        profile['is_supercharged'] = self.is_supercharged_var.get()
        profile.setdefault('rpm_model', {})['speed_kmh'] = [float(e[0].get()) for e in self.rpm_entries]
        profile['rpm_model']['rpm'] = [float(e[1].get()) for e in self.rpm_entries]
        profile.setdefault('fuel_model', {})['rpm'] = [float(e[0].get()) for e in self.fuel_entries]
        profile['fuel_model']['lph'] = [float(e[1].get()) for e in self.fuel_entries]
        self.profiles[profile_name] = profile

    def save_and_close(self):
        try:
            self.save_current_profile()
            with open("profiles.json", "w") as f:
                json.dump(self.profiles, f, indent=2)
            messagebox.showinfo("Success", "Profiles saved successfully!")
            self.callback(self.profiles)
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save profiles.\n\n{e}")

    def add_profile(self):
        new_name = simpledialog.askstring("New Profile", "Enter the name for the new PWC profile:")
        if not new_name or new_name in self.profiles:
            messagebox.showwarning("Invalid Name", "Profile name cannot be empty or already exist.")
            return
        template = list(self.profiles.values())[0]
        new_profile = json.loads(json.dumps(template))
        new_profile['description'] = "New profile"
        new_profile['engine_size'] = "1.0L"
        new_profile['default_rider_weight'] = 80
        new_profile['is_supercharged'] = False
        self.profiles[new_name] = new_profile
        self.profile_menu['values'] = list(self.profiles.keys())
        self.current_profile_name.set(new_name)
        self.load_profile_data()

    def delete_profile(self):
        if len(self.profiles) <= 1:
            messagebox.showwarning("Cannot Delete", "You cannot delete the last profile.")
            return
        profile_to_delete = self.current_profile_name.get()
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the profile '{profile_to_delete}'?"):
            del self.profiles[profile_to_delete]
            self.profile_menu['values'] = list(self.profiles.keys())
            self.current_profile_name.set(list(self.profiles.keys())[0])
            self.load_profile_data()


# --- Visualization Window ---
# <<< MODIFIED: Added a new "Engine Analysis" tab and its creation logic >>>
class VisualizationWindow(ttk.Toplevel):
    def __init__(self, master, df_telemetry, df_raw):
        super().__init__(master=master)
        self.title("Ride Visualization")
        self.geometry("1000x750")

        notebook = ttk.Notebook(self)
        notebook.pack(fill=BOTH, expand=YES, padx=10, pady=10)

        graphs_frame = ttk.Frame(notebook)
        notebook.add(graphs_frame, text="Graphs")
        self.create_graphs_tab(graphs_frame, df_telemetry)

        # NEW: Engine Analysis Tab
        engine_frame = ttk.Frame(notebook)
        notebook.add(engine_frame, text="Engine Analysis")
        self.create_engine_tab(engine_frame, df_telemetry)

        if map_viz_available:
            map_frame = ttk.Frame(notebook)
            notebook.add(map_frame, text="Map")
            self.create_map_tab(map_frame, df_raw)

    def create_graphs_tab(self, parent, df):
        # ... (This function is correct and unchanged)
        fig = Figure(figsize=(10, 7), dpi=100)
        ax1 = fig.add_subplot(111)
        color = 'tab:blue'
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Speed (km/h)', color=color)
        ax1.plot(df['timestamp'], df['Speed (km/h)'], color=color, label='Speed')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Engine RPM', color=color)
        ax2.plot(df['timestamp'], df['Engine RPM (rpm)'], color=color, linestyle='--', label='RPM')
        ax2.tick_params(axis='y', labelcolor=color)
        fig.suptitle('Speed and RPM over Time', fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

    def create_engine_tab(self, parent, df):
        fig = Figure(figsize=(10, 7), dpi=100)
        ax = fig.add_subplot(111)

        # Define RPM bins
        max_rpm = df['Engine RPM (rpm)'].max()
        bins = np.arange(0, max_rpm + 1000, 1000)
        labels = [f'{int(bins[i])}-{int(bins[i + 1])}' for i in range(len(bins) - 1)]

        # Group data by RPM bins and sum the time spent in each
        df['rpm_bin'] = pd.cut(df['Engine RPM (rpm)'], bins=bins, labels=labels, right=False)
        time_in_rpm = df.groupby('rpm_bin', observed=False)['time_delta_s'].sum()

        # Convert seconds to minutes for readability
        time_in_minutes = time_in_rpm / 60

        # Create horizontal bar chart
        bars = ax.barh(time_in_minutes.index, time_in_minutes.values, color='skyblue')

        ax.set_xlabel('Time (Minutes)')
        ax.set_ylabel('RPM Range')
        ax.set_title('Time Spent in RPM Ranges')

        # Add labels to the bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + time_in_minutes.max() * 0.01
            ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.1f} min', va='center')

        ax.set_xlim(right=time_in_minutes.max() * 1.15)  # Add space for labels

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

    def create_map_tab(self, parent, df):
        # ... (This function is correct and unchanged)
        fig = Figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(111)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="epsg:4326")
        gdf = gdf.to_crs(epsg=3857)
        gdf.plot(ax=ax, color='red', label='Route', zorder=2)
        ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
        ax.set_title('GPS Track')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)


# --- Main Application GUI ---
class TelemetryApp:
    # ... (The rest of the app is correct and unchanged)
    # Full class is omitted for brevity but is identical to the previous version.
    def __init__(self, root):
        self.root = root
        self.root.title("PWC Telemetry Processor V11 (Analysis)")
        self.root.geometry("600x650")
        self.profiles = self.load_profiles()
        self.input_file_path = tk.StringVar(value="No file selected...")
        self.selected_profile = tk.StringVar()
        self.selected_profile.trace_add("write", self.on_profile_change)
        self.rider_weight = tk.DoubleVar()
        self.fuel_load = tk.StringVar(value="Full")
        self.drive_mode = tk.StringVar(value="Normal")
        self.water_condition = tk.StringVar(value="Calm")
        self.show_visuals = tk.BooleanVar(value=True)
        self._create_widgets()
        if self.profiles:
            self.selected_profile.set(list(self.profiles.keys())[0])

    def load_profiles(self):
        try:
            with open("profiles.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            messagebox.showerror("Profile Error", "Could not load 'profiles.json'.")
            return {}

    def update_profiles(self, new_profiles):
        self.profiles = new_profiles
        self.profile_menu['values'] = list(self.profiles.keys())
        if self.selected_profile.get() not in self.profiles:
            self.selected_profile.set(list(self.profiles.keys())[0] if self.profiles else "")
        self.log("Profiles have been updated.")

    def on_profile_change(self, *args):
        profile_name = self.selected_profile.get()
        profile_data = self.profiles.get(profile_name)
        if profile_data:
            self.rider_weight.set(profile_data.get("default_rider_weight", 80))
            self.log(f"Loaded profile '{profile_name}'. Default weight set to {self.rider_weight.get()} kg.")

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=BOTH, expand=YES)
        top_frame = ttk.Labelframe(main_frame, text="Setup", padding="10")
        top_frame.pack(fill=X, pady=10)
        file_label = ttk.Label(top_frame, textvariable=self.input_file_path, wraplength=450);
        file_label.grid(row=0, column=0, columnspan=2, sticky=EW, pady=5)
        browse_btn = ttk.Button(top_frame, text="Browse...", command=self.browse_file, bootstyle="info");
        browse_btn.grid(row=0, column=2, padx=5)
        ttk.Label(top_frame, text="PWC Profile:").grid(row=1, column=0, sticky=W, pady=5)
        self.profile_menu = ttk.Combobox(top_frame, textvariable=self.selected_profile,
                                         values=list(self.profiles.keys()), state="readonly")
        self.profile_menu.grid(row=1, column=1, sticky=EW, pady=5)
        manage_btn = ttk.Button(top_frame, text="Manage Profiles...", command=self.open_profile_manager,
                                bootstyle="secondary")
        manage_btn.grid(row=1, column=2, padx=5)
        top_frame.columnconfigure(1, weight=1)
        conditions_frame = ttk.Labelframe(main_frame, text="Ride Conditions", padding="15")
        conditions_frame.pack(fill=X, pady=10)
        conditions_frame.columnconfigure(1, weight=1)
        fields = {"Rider Weight (kg):": (self.rider_weight, "Entry", None),
                  "Fuel Load:": (self.fuel_load, "Combobox", ["Full", "Half", "Low"]),
                  "Drive Mode:": (self.drive_mode, "Combobox", ["Normal", "L-Mode", "No-Wake"]),
                  "Water Condition:": (self.water_condition, "Combobox", ["Calm", "Rough"])}
        for i, (label, (var, widget_type, values)) in enumerate(fields.items()):
            ttk.Label(conditions_frame, text=label).grid(row=i, column=0, padx=5, pady=5, sticky=W)
            widget = ttk.Entry(conditions_frame, textvariable=var,
                               width=15) if widget_type == "Entry" else ttk.Combobox(conditions_frame, textvariable=var,
                                                                                     values=values, state="readonly",
                                                                                     width=15)
            widget.grid(row=i, column=1, padx=5, pady=5, sticky=W)
        vis_frame = ttk.Frame(main_frame);
        vis_frame.pack(fill=X, pady=5)
        vis_check = ttk.Checkbutton(vis_frame, text="Show Visualizations (Graphs & Map)", variable=self.show_visuals);
        vis_check.pack(side=LEFT)
        if not matplotlib_available or not map_viz_available:
            vis_check.config(state=DISABLED)
            missing_libs = [lib for lib, avail in
                            [('matplotlib', matplotlib_available), ('geopandas, contextily', map_viz_available)] if
                            not avail]
            ttk.Label(vis_frame, text=f"({', '.join(missing_libs)} not installed)", bootstyle="secondary").pack(
                side=LEFT, padx=10)
        ttk.Button(main_frame, text="Generate Telemetry Data", command=self.run_processing_thread,
                   bootstyle="success-lg").pack(pady=20, fill=X)
        log_frame = ttk.Labelframe(main_frame, text="Status Log", padding="10");
        log_frame.pack(fill=BOTH, expand=YES)
        self.log_text = tk.Text(log_frame, height=10, state="disabled", wrap="word", bg="#333", fg="#ddd",
                                relief="flat")
        scrollbar = ttk.Scrollbar(log_frame, orient=VERTICAL, command=self.log_text.yview);
        self.log_text['yscrollcommand'] = scrollbar.set
        scrollbar.pack(side=RIGHT, fill=Y);
        self.log_text.pack(side=LEFT, fill=BOTH, expand=YES)

    def open_profile_manager(self):
        ProfileManager(self.root, self.profiles, self.update_profiles)

    def log(self, message):
        self.log_text.config(state="normal"); self.log_text.insert(tk.END,
                                                                   f"{datetime.now().strftime('%H:%M:%S')} - {message}\n"); self.log_text.config(
            state="disabled"); self.log_text.see(tk.END); self.root.update_idletasks()

    def browse_file(self):
        file_path = filedialog.askopenfilename(title="Select Activity File", filetypes=self.get_supported_files()); (
        self.input_file_path.set(file_path),
        self.log(f"Selected input file: {os.path.basename(file_path)}")) if file_path else None

    def get_supported_files(self):
        supported_extensions = []
        if gpxpy: supported_extensions.append("*.gpx")
        if fitparse: supported_extensions.append("*.fit")
        if gpmf_parser_available: supported_extensions.append("*.mp4")
        if supported_extensions: return [("Supported GPS Files", " ".join(supported_extensions)), ("All files", "*.*")]
        return [("All files", "*.*")]

    def run_processing_thread(self):
        thread = threading.Thread(target=self.process_file); thread.daemon = True; thread.start()

    def process_file(self):
        input_path = self.input_file_path.get()
        if not os.path.exists(input_path): self.log(
            "ERROR: Please select a valid input file first."); messagebox.showerror("Error",
                                                                                    "No input file selected."); return
        default_name = f"{os.path.splitext(os.path.basename(input_path))[0]}_telemetry.csv"
        output_path = filedialog.asksaveasfilename(title="Save Telemetry As", initialfile=default_name,
                                                   defaultextension=".csv",
                                                   filetypes=(("CSV files", "*.csv"), ("Excel files", "*.xlsx")))
        if not output_path: self.log("Save operation cancelled by user."); return
        try:
            self.log("Starting telemetry generation...")
            file_ext = os.path.splitext(input_path)[1].lower()
            parser_map = {'.fit': parse_fit_file, '.gpx': parse_gpx_file, '.mp4': parse_gopro_file}
            if file_ext not in parser_map or parser_map.get(file_ext) is None: raise ValueError(
                f"Unsupported file type or missing parser library for '{file_ext}'")
            raw_df = parser_map[file_ext](input_path)
            self.log(f"Successfully parsed {len(raw_df)} data points.")
            profile_name = self.selected_profile.get()
            selected_profile_data = self.profiles.get(profile_name)
            if not selected_profile_data: raise ValueError(f"Profile '{profile_name}' not found!")
            calculated_df, summary_data = calculate_telemetry(raw_df[['timestamp', 'speed_kmh']],
                                                              self.get_ride_conditions(), selected_profile_data)
            df_to_save = calculated_df.copy();
            df_to_save.rename(columns={'timestamp': 'date'}, inplace=True)
            cols_to_save = ['date', 'Speed (km/h)', 'Engine RPM (rpm)', 'Fuel Rate (L/h)', 'Fuel Used (L)']
            if output_path.endswith('.csv'):
                df_to_save[cols_to_save].to_csv(output_path, index=False, date_format='%Y-%m-%dT%H:%M:%S.%fZ')
            else:
                df_to_save[cols_to_save].to_excel(output_path, index=False, engine='openpyxl')
            self.log(f"SUCCESS! Telemetry data saved to:\n{output_path}")
            summary_message = self.format_summary_message(summary_data)
            messagebox.showinfo("Ride Summary", summary_message)
            if self.show_visuals.get() and matplotlib_available:
                self.root.after(0, lambda: VisualizationWindow(self.root, df_telemetry=calculated_df, df_raw=raw_df))
        except Exception as e:
            self.log(f"ERROR: {e}"); messagebox.showerror("Processing Error", f"An error occurred:\n\n{e}")

    def get_ride_conditions(self):
        return (self.rider_weight.get(), self.fuel_load.get(), self.drive_mode.get(), self.water_condition.get())

    def format_summary_message(self, summary_data):
        best_cruise_text = "N/A (not enough planing data)"
        if summary_data['best_cruise_speed_kmh'] > 0: best_cruise_text = (
            f"{summary_data['best_cruise_economy_km_per_l']:.2f} km/L at ~{summary_data['best_cruise_speed_kmh']:.0f} km/h")
        return (f"--- Ride Summary ---\n"
                f"Total Distance: {summary_data['total_distance_km']:.2f} km\n"
                f"Total Fuel Used: {summary_data['total_fuel_l']:.2f} L\n\n"
                f"--- Speed ---\n"
                f"Max Speed: {summary_data['max_speed_kmh']:.1f} km/h\n"
                f"Avg. Moving Speed: {summary_data['avg_moving_speed_kmh']:.1f} km/h\n\n"
                f"--- Averages ---\n"
                f"Overall Consumption: {summary_data['l_per_100km']:.2f} L/100km\n"
                f"Overall Economy: {summary_data['km_per_l']:.2f} km/L\n\n"
                f"--- Efficiency ---\n"
                f"Best Planing Economy: {best_cruise_text}")


if __name__ == "__main__":
    if not os.path.exists("profiles.json"):
        messagebox.showerror("Fatal Error", "'profiles.json' not found. The application cannot start.")
    else:
        root = ttk.Window(themename="darkly")
        app = TelemetryApp(root)
        root.mainloop()