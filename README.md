PWC Telemetry Processor - V11
A desktop application for parsing, analyzing, and exporting telemetry data from personal watercraft (PWC) activities. The tool can process GPS data from `.gpx`, and GoPro `.mp4` files, and uses customizable PWC profiles to calculate detailed engine and fuel consumption metrics.

It features a user-friendly GUI built with Tkinter and TTKBootstrap, providing data visualization, ride summaries, and a powerful profile manager to fine-tune calculations for different watercraft models and conditions.

## Key Features

-   **Multi-Format Support:** Ingests telemetry data standard GPS (`.gpx`), and GoPro (`.mp4`) files.
-   **Advanced Calculation Engine:**
    -   Calculates engine RPM based on speed using a cubic spline interpolation model.
    -   Estimates fuel consumption (L/h) and cumulative fuel usage.
    -   Provides a detailed ride summary including max speed, average speed, total distance, fuel economy (L/100km and km/L), and best cruise efficiency.
-   **Customizable PWC Profiles:**
    -   Manage multiple PWC profiles via a user-friendly interface.
    -   Define engine size, supercharger status, default rider weight, and detailed performance curves (Speed-vs-RPM, RPM-vs-Fuel).
    -   Profiles are stored in a simple `profiles.json` file.
-   **Data Visualization:**
    -   **Graphs:** Plots speed and RPM over time.
    -   **Engine Analysis:** A bar chart showing the total time spent in different RPM ranges.
    -   **GPS Map:** Displays the recorded GPS track on an OpenStreetMap tile.
-   **Data Export:** Saves the processed telemetry data to `.csv` format.
-   **User-Friendly GUI:** A modern, dark-themed interface built with `ttkbootstrap`.

## Setup and Installation

To run this application, you will need Python 3 and a few external dependencies.

### 1. Prerequisites

-   **Python 3.9+**
-   **FFmpeg** (Required for GoPro `.mp4` file processing)

#### FFmpeg Installation
This application requires `ffmpeg.exe` to extract the GPMF telemetry stream from GoPro video files.

-   Download a release build of FFmpeg from the official site: [ffmpeg.org/download.html](https://ffmpeg.org/download.html)
-   From the downloaded `.zip` file, find `ffmpeg.exe` inside the `bin` folder.
-   **Place `ffmpeg.exe` inside a `bin` folder in the root of this project directory.**

Your project folder structure should look like this:
```
PWC_Telemetry_Processor/
│
├── bin/
│   └── ffmpeg.exe      <-- PLACE FFMPEG HERE
│
├── PWC_Telemetry_Processor_V11.py
├── profiles.json
└── requirements.txt
```

### 2. Install Python Libraries

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install all required packages from the requirements file
pip install -r requirements.txt
```

The `requirements.txt` file should contain:
```
ttkbootstrap
pandas
numpy
scipy
matplotlib
geopandas
contextily
fitparse
gpmf
gpxpy
openpyxl
```

## How to Use

1.  **Launch the Application:**
    Run the main Python script from your terminal (make sure your virtual environment is active).
    ```bash
    python PWC_Telemetry_Processor_V11.py
    ```

2.  **Manage Profiles (First Time):**
    -   Click the **"Manage Profiles..."** button.
    -   Review and edit the default profile to match your PWC's specifications. You can adjust the Speed-to-RPM curve and the RPM-to-Fuel-Consumption curve.
    -   You can add new profiles for different watercraft.
    -   Click **"Save All Profiles and Close"**.

3.  **Process a File:**
    -   Click **"Browse..."** to select your `.gpx`, or `.mp4` activity file.
    -   Select the appropriate **PWC Profile** from the dropdown menu.
    -   Adjust the **Ride Conditions** (Rider Weight, Fuel Load, etc.) for accuracy.
    -   Click the **"Generate Telemetry Data"** button.
    -   You will be prompted to choose a location and name for the output `.csv` or `.xlsx` file.

4.  **Review Results:**
    -   A **Ride Summary** pop-up will appear with key statistics.
    -   If "Show Visualizations" is checked, a new window will open with tabs for **Graphs**, **Engine Analysis**, and a **Map** of your ride.
    -   The main application window will show a status log of the entire process.

## Technology Stack

-   **GUI Framework:** [Tkinter](https://docs.python.org/3/library/tkinter.html) with [ttkbootstrap](https://ttkbootstrap.readthedocs.io/en/latest/) for modern styling.
-   **Data Manipulation:** [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for efficient data handling and calculations.
-   **Scientific Computing:** [SciPy](https://scipy.org/) for cubic spline interpolation (`interp1d`).
-   **Data Visualization:** [Matplotlib](https://matplotlib.org/) for plotting graphs.
-   **Geospatial Analysis:** [GeoPandas](https://geopandas.org/) and [Contextily](https://contextily.readthedocs.io/en/latest/) for rendering the GPS track on a map.
-   **File Parsers:**
    -   `gpmf` for GoPro telemetry.
    -   `gpxpy` for `.gpx` files.
