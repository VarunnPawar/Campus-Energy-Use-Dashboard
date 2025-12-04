# campus_energy_dashboard.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import glob

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# --- Sample data generator (for testing) ---
def generate_sample_building_csvs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    buildings = ["Library", "Admin", "Hostel"]
    rng = pd.date_range(end=pd.Timestamp.today(), periods=24*30, freq='H')  # last 30 days hourly
    for b in buildings:
        df = pd.DataFrame({
            "timestamp": rng,
            "kwh": np.abs( (np.sin(np.linspace(0,6.28,len(rng))) + np.random.normal(0,0.5,len(rng))) * (10 + np.random.rand()*5) + 5 )
        })
        fname = DATA_DIR / f"{b.replace(' ','_')}_sample.csv"
        df.to_csv(fname, index=False)
        logging.info("Wrote sample data %s", fname)

# --- OOP Models ---
class MeterReading:
    def __init__(self, timestamp: pd.Timestamp, kwh: float):
        self.timestamp = pd.to_datetime(timestamp)
        self.kwh = float(kwh)

class Building:
    def __init__(self, name: str):
        self.name = name
        self.readings = []  # list of MeterReading

    def add_reading(self, reading: MeterReading):
        self.readings.append(reading)

    def to_dataframe(self):
        df = pd.DataFrame([{"timestamp": r.timestamp, "kwh": r.kwh} for r in self.readings])
        if not df.empty:
            df = df.set_index(pd.to_datetime(df['timestamp'])).drop(columns=['timestamp'])
        return df

    def total_consumption(self):
        df = self.to_dataframe()
        return float(df['kwh'].sum()) if not df.empty else 0.0

    def summary_stats(self):
        df = self.to_dataframe()
        if df.empty:
            return {}
        return {
            "mean_kwh": df['kwh'].mean(),
            "min_kwh": df['kwh'].min(),
            "max_kwh": df['kwh'].max(),
            "total_kwh": df['kwh'].sum()
        }

class BuildingManager:
    def __init__(self):
        self.buildings = {}

    def ingest_csv(self, filepath: Path):
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            logging.error("File not found: %s", filepath)
            return
        except Exception as e:
            logging.error("Error reading %s : %s", filepath, e)
            return
        # Guess building name from filename
        name = filepath.stem
        # Standardize columns
        if 'timestamp' not in df.columns and 'time' in df.columns:
            df.rename(columns={'time':'timestamp'}, inplace=True)
        if 'kwh' not in df.columns and 'energy' in df.columns:
            df.rename(columns={'energy':'kwh'}, inplace=True)
        # Add optional metadata column if missing
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception:
            logging.warning("Could not parse timestamps in %s; skipping.", filepath)
            return
        df = df.dropna(subset=['timestamp','kwh'])
        # create Building
        b = Building(name)
        for _, row in df.iterrows():
            b.add_reading(MeterReading(row['timestamp'], row['kwh']))
        self.buildings[name] = b
        logging.info("Ingested %d readings for building %s", len(b.readings), name)

    def to_combined_dataframe(self):
        frames = []
        for name, b in self.buildings.items():
            df = b.to_dataframe().copy()
            if df.empty: 
                continue
            df['building'] = name
            frames.append(df.reset_index().rename(columns={'index':'timestamp'}))
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined['timestamp'] = pd.to_datetime(combined['timestamp'])
            combined = combined.sort_values('timestamp')
            combined = combined.set_index('timestamp')
            return combined
        return pd.DataFrame(columns=['kwh','building'])

    def building_summary_table(self):
        rows = []
        for name,b in self.buildings.items():
            s = b.summary_stats()
            rows.append({
                "building": name,
                "mean_kwh": s.get('mean_kwh',0),
                "min_kwh": s.get('min_kwh',0),
                "max_kwh": s.get('max_kwh',0),
                "total_kwh": s.get('total_kwh',0)
            })
        return pd.DataFrame(rows).sort_values('total_kwh', ascending=False)

# --- Aggregation & Visualization Helpers ---
def calculate_daily_totals(df_combined: pd.DataFrame):
    return df_combined.groupby('building').resample('D')['kwh'].sum().reset_index().set_index('timestamp')

def calculate_weekly_aggregates(df_combined: pd.DataFrame):
    return df_combined.groupby('building').resample('W')['kwh'].sum().reset_index().set_index('timestamp')

def plot_dashboard(df_combined: pd.DataFrame, outpath: Path):
    buildings = df_combined['building'].unique()
    fig, axes = plt.subplots(3,1, figsize=(12,12), constrained_layout=True)
    # 1. Trend line - daily consumption per building (resample daily)
    daily = df_combined.groupby('building').resample('D')['kwh'].sum().reset_index().pivot(index='timestamp', columns='building', values='kwh')
    daily.plot(ax=axes[0])
    axes[0].set_title("Daily Consumption Trend (kWh)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("kWh")
    # 2. Bar chart - average weekly usage per building
    weekly_avg = df_combined.groupby('building').resample('W')['kwh'].sum().groupby('building').mean()
    axes[1].bar(weekly_avg.index, weekly_avg.values)
    axes[1].set_title("Average Weekly Usage (kWh) per Building")
    axes[1].set_ylabel("kWh")
    axes[1].tick_params(axis='x', rotation=45)
    # 3. Scatter - peak-hour consumption vs time (plot top N peaks)
    df_peaks = df_combined.reset_index().sort_values('kwh', ascending=False).groupby('building').head(30)
    axes[2].scatter(df_peaks['timestamp'], df_peaks['kwh'], alpha=0.6)
    axes[2].set_title("Peak Hour Consumption (sample of top readings)")
    axes[2].set_xlabel("Timestamp")
    axes[2].set_ylabel("kWh")
    plt.savefig(outpath)
    plt.close()
    logging.info("Dashboard saved to %s", outpath)

def generate_summary_text(df_combined: pd.DataFrame, building_summary: pd.DataFrame, outpath: Path):
    total_consumption = df_combined['kwh'].sum()
    highest_building = building_summary.iloc[0]['building'] if not building_summary.empty else "N/A"
    # Peak load time:
    if not df_combined.empty:
        peak_row = df_combined.reset_index().sort_values('kwh', ascending=False).iloc[0]
        peak_time = peak_row.name.isoformat()
        peak_building = peak_row['building']
        peak_value = peak_row['kwh']
    else:
        peak_time = peak_building = peak_value = "N/A"
    with outpath.open("w") as f:
        f.write("Campus Energy Usage Summary\n")
        f.write("===========================\n\n")
        f.write(f"Total campus consumption (kWh): {total_consumption:.2f}\n")
        f.write(f"Highest-consuming building: {highest_building}\n")
        f.write(f"Peak load: {peak_value} kWh at {peak_time} (Building: {peak_building})\n\n")
        f.write("Top buildings summary:\n")
        f.write(building_summary.to_string(index=False))
    logging.info("Summary written to %s", outpath)

# --- Main Pipeline ---
def main():
    # If no data present, generate sample CSVs
    if not any(DATA_DIR.glob("*.csv")):
        logging.info("No CSV files in data/. Generating sample CSVs.")
        generate_sample_building_csvs()

    bm = BuildingManager()
    for csvfile in DATA_DIR.glob("*.csv"):
        bm.ingest_csv(csvfile)

    combined = bm.to_combined_dataframe()
    if combined.empty:
        logging.error("No readings ingested. Exiting.")
        return

    # Export cleaned combined
    cleaned_out = OUT_DIR / "cleaned_energy_data.csv"
    combined.reset_index().to_csv(cleaned_out, index=False)
    logging.info("Cleaned combined CSV exported: %s", cleaned_out)

    # Building summary
    bsummary = bm.building_summary_table()
    bsummary_out = OUT_DIR / "building_summary.csv"
    bsummary.to_csv(bsummary_out, index=False)
    logging.info("Building summary exported: %s", bsummary_out)

    # Plot dashboard
    dashboard_out = OUT_DIR / "dashboard.png"
    plot_dashboard(combined, dashboard_out)

    # Generate textual summary
    summary_out = OUT_DIR / "summary.txt"
    generate_summary_text(combined, bsummary, summary_out)
    logging.info("Pipeline complete. Outputs written to %s", OUT_DIR)

if __name__ == "__main__":
    main() 