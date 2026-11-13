"""Unseen weather stations for local validation (2026-11-11; user: 「可以下載」 public weather data; spec
docs/plan/beat-01060-tree-20261111.md). NOAA Integrated Surface Database global-hourly 2023 (public domain; the AWS
open-data copy noaa-global-hourly-pds) for five stations far from the training five and not Minneapolis:
Riyadh (hot desert), Dubai (hot humid), Seattle (marine), London (cool oceanic), Beijing (continental monsoon).

1. Weather files in the official format (smart_buildings/weather/csv/*.csv): hourly rows Time (%Y%m%d-%H%M), StationName,
   Location, TempC, DewPointC, WindspeedKmph, Humidity, TempF. The official files carry station-local clock times
   (summer warmest hour 13-15, coolest 3-5) labelled as UTC, so the NOAA UTC observations are shifted by the station's
   standard-time offset. Per hour: the mean of the valid reports (quality not 3/7, value not 9999); gaps filled by
   linear interpolation in time. Humidity from temperature and dew point (Magnus, Alduchov & Eskridge 1996).
2. Scenarios: each of the 6 official buildings x 5 stations x {January, July}, a copy of the building's official
   training config.gin with only the weather path, the start (the 5th, 06:00 like the official windows), the
   convection coefficient (19.1 W/m2K for every new station: our choice, the official values are 8.8-26.7 by
   station) and the label/metrics path changed; floor_plan.npy and zone_map.npy copied next to it.
Downloads in data/raw/external_noaa_isd/ are read as data only.  .venv/bin/python -I -B pipeline/setup/new_stations_build.py"""
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/external_noaa_isd"
WOUT = ROOT / "data/processed/weather_new/csv"
SOUT = ROOT / "data/processed/newstations"
OFFICIAL = ROOT / "data/raw/kaggle/smart_buildings/training_scenarios"
STATIONS = {  # name: (ISD id, standard-time offset hours, label)
    "riyadh": ("40437099999", 3, "Riyadh-King.Khaled.Intl"),
    "dubai": ("41194099999", 4, "Dubai.Intl"),
    "seattle": ("72793024233", -8, "Seattle-Tacoma.Intl"),
    "london": ("03772099999", 0, "London-Heathrow"),
    "beijing": ("54511099999", 8, "Beijing-Capital.Intl"),
}
MONTHS = {"jan": "2023-01-05 06:00", "jul": "2023-07-05 06:00"}
CONVECTION = 19.1


def field(series, scale, missing):
    parts = series.astype(str).str.split(",", expand=True)
    val = pd.to_numeric(parts[0], errors="coerce")
    q = parts[1].astype(str)
    bad = (val.abs() >= missing) | q.isin(["3", "7"]) | val.isna()
    return (val / scale).where(~bad)


def wind(series):
    parts = series.astype(str).str.split(",", expand=True)
    sp = pd.to_numeric(parts[3], errors="coerce")
    q = parts[4].astype(str)
    return (sp / 10.0).where(~((sp >= 9999) | q.isin(["3", "7"]) | sp.isna()))


def rh(t, td):
    return 100.0 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))


def weather(name):
    sid, off, label = STATIONS[name]
    df = pd.read_csv(RAW / f"{sid}_2023.csv", usecols=["DATE", "TMP", "DEW", "WND"], dtype=str)
    t = pd.to_datetime(df["DATE"], utc=True)
    x = pd.DataFrame({"t": field(df["TMP"], 10.0, 999).to_numpy(), "td": field(df["DEW"], 10.0, 999).to_numpy(),
                      "w": wind(df["WND"]).to_numpy()},
                     index=pd.DatetimeIndex((t + pd.Timedelta(hours=off)).dt.tz_localize(None)))
    hourly = x.resample("1h").mean()
    idx = pd.date_range("2023-01-01 00:00", "2023-12-31 23:00", freq="1h")
    hourly = hourly.reindex(idx)
    gaps = int(hourly["t"].isna().sum())
    hourly = hourly.interpolate(method="time", limit_direction="both")
    hourly["td"] = np.minimum(hourly["td"], hourly["t"])
    out = pd.DataFrame({"Time": idx.strftime("%Y%m%d-%H%M"), "StationName": label, "Location": label,
                        "TempC": hourly["t"].round(2).to_numpy(), "DewPointC": hourly["td"].round(2).to_numpy(),
                        "WindspeedKmph": (hourly["w"] * 3.6).round(2).to_numpy(),
                        "Humidity": rh(hourly["t"], hourly["td"]).clip(1, 100).round(1).to_numpy(),
                        "TempF": (hourly["t"] * 9 / 5 + 32).round(2).to_numpy()})
    WOUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(WOUT / f"{name}.csv")
    tt = pd.to_datetime(out["Time"], format="%Y%m%d-%H%M")
    summer = out[tt.dt.month.isin([6, 7, 8])].assign(h=tt.dt.hour)
    prof = summer.groupby("h")["TempC"].mean()
    mon = out.assign(m=tt.dt.month).groupby("m")["TempC"].mean().round(1).tolist()
    print(f"{name:8s} hours missing before fill {gaps}; summer warmest hour {int(prof.idxmax())}, coolest "
          f"{int(prof.idxmin())}; monthly mean C {mon}")


def scenarios():
    n = 0
    for bdir in sorted(OFFICIAL.iterdir()):
        src = sorted(bdir.glob("*/config.gin"))[0]
        text = src.read_text(encoding="utf-8")
        old_label = f"{bdir.name}/{src.parent.name}"
        for st in STATIONS:
            for mon, start in MONTHS.items():
                lab = f"{bdir.name}/{st}_{mon}"
                d = SOUT / bdir.name / f"{st}_{mon}"
                d.mkdir(parents=True, exist_ok=True)
                g = text.replace(old_label, lab)
                g = re.sub(r"^convection_coefficient = .*$", f"convection_coefficient = {CONVECTION}", g, flags=re.M)
                g = re.sub(r"^weather_csv_path = .*$", f"weather_csv_path = '{(WOUT / (st + '.csv')).as_posix()}'", g,
                           flags=re.M)
                g = re.sub(r"^start_timestamp = .*$", f"start_timestamp = '{start}'", g, flags=re.M)
                g = re.sub(r'^metrics_path = .*$', f'metrics_path = "{(d / "metrics").as_posix()}"', g, flags=re.M)
                g = g.replace(f"smart_buildings/training_scenarios/{lab}/floor_plan.npy", (d / "floor_plan.npy").as_posix())
                g = g.replace(f"smart_buildings/training_scenarios/{lab}/zone_map.npy", (d / "zone_map.npy").as_posix())
                (d / "config.gin").write_text(g, encoding="utf-8", newline="\n")
                shutil.copyfile(src.parent / "floor_plan.npy", d / "floor_plan.npy")
                shutil.copyfile(src.parent / "zone_map.npy", d / "zone_map.npy")
                n += 1
    first = sorted(SOUT.glob("*/*/config.gin"))[0]
    diff = [l for l in first.read_text(encoding="utf-8").splitlines() if "newstations" in l or "weather_new" in l
            or l.startswith(("convection_coefficient", "start_timestamp", "label"))]
    print(f"{n} scenarios under {SOUT.relative_to(ROOT)}; changed lines of {first.parent.relative_to(SOUT)}:")
    for l in diff:
        print("   ", l)
    (SOUT / "manifest.json").write_text(json.dumps({"stations": {k: v[0] for k, v in STATIONS.items()},
                                                    "months": MONTHS, "convection": CONVECTION,
                                                    "source": "NOAA ISD global-hourly 2023 (noaa-global-hourly-pds)"},
                                                   indent=1) + "\n", encoding="utf-8")


def main():
    for name in STATIONS:
        weather(name)
    scenarios()


if __name__ == "__main__":
    main()
