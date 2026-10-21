
import argparse
import glob
import os
from pathlib import Path

import cfgrib
import s3fs
import xarray as xr


# ============================================================
# Configuration
# ============================================================
parser = argparse.ArgumentParser(
    description="Download ECMWF AIFS forecasts and create India NetCDF files."
)
parser.add_argument("--date", default="20261021", help="Forecast run date: YYYYMMDD")
parser.add_argument("--hour", default="00z", choices=["00z", "06z", "12z", "18z"])
parser.add_argument("--step-start", type=int, default=0)
parser.add_argument("--step-end", type=int, default=360)
parser.add_argument("--step-interval", type=int, default=6)
parser.add_argument("--output-dir", default="Input_NC_files")
parser.add_argument("--delete-originals", action="store_true")
args = parser.parse_args()

INPUT_DATE = args.date
RUN_HOUR = args.hour
STEP_START = args.step_start
STEP_END = args.step_end
STEP_INTERVAL = args.step_interval
OUTPUT_DIR = Path(args.output_dir)
DELETE_ORIGINALS = args.delete_originals

LAT_MIN, LAT_MAX = 6.0, 38.0
LON_MIN, LON_MAX = 66.0, 100.0

if STEP_INTERVAL <= 0 or STEP_START < 0 or STEP_END < STEP_START:
    raise ValueError("Invalid forecast-step configuration.")

if len(INPUT_DATE) != 8 or not INPUT_DATE.isdigit():
    raise ValueError("Date must be in YYYYMMDD format.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ECMWF S3 forecast directory
BASE_S3 = (
    f"ecmwf-forecasts/{INPUT_DATE}/{RUN_HOUR}/"
    "aifs-single/0p25/oper"
)

# The filename's initialization hour must match RUN_HOUR.
RUN_HOUR_CODE = RUN_HOUR[:2]
INIT_TIME = f"{INPUT_DATE}{RUN_HOUR_CODE}0000"

fs = s3fs.S3FileSystem(
    anon=True,
    client_kwargs={"region_name": "eu-central-1"},
)


# ============================================================
# Helper functions
# ============================================================
def subset_india(da):
    """Subset a DataArray to the specified India-region bounds."""
    lat = da.latitude
    lon = da.longitude

    lat_slice = (
        slice(LAT_MAX, LAT_MIN)
        if float(lat[0]) > float(lat[-1])
        else slice(LAT_MIN, LAT_MAX)
    )
    lon_slice = (
        slice(LON_MAX, LON_MIN)
        if float(lon[0]) > float(lon[-1])
        else slice(LON_MIN, LON_MAX)
    )

    return da.sel(latitude=lat_slice, longitude=lon_slice)


def clean(da, name=None):
    """Remove scalar coordinates and GRIB-specific attributes."""
    scalar_coords = [
        c for c in da.coords if da[c].ndim == 0
    ]
    da = da.drop_vars(scalar_coords, errors="ignore")
    da.attrs = {
        k: v for k, v in da.attrs.items()
        if not k.startswith("GRIB_")
    }

    if name:
        da.name = name

    return da


def grib_to_nc(grib_path, nc_path):
    """Extract selected forecast variables and save as NetCDF."""
    print(f"  Opening GRIB: {grib_path}")

    datasets = cfgrib.open_datasets(str(grib_path))

    def get_var(ds_list, idx, varname):
        # Prefer the expected dataset index, then search all datasets.
        if idx < len(ds_list) and varname in ds_list[idx]:
            return ds_list[idx][varname]

        for ds in ds_list:
            if varname in ds:
                return ds[varname]

        raise KeyError(
            f"Variable '{varname}' not found in GRIB datasets."
        )

    try:
        tcc = clean(get_var(datasets, 0, "tcc"), "tcc")
        u10 = clean(get_var(datasets, 1, "u10"), "u10")
        v10 = clean(get_var(datasets, 1, "v10"), "v10")
        t2m = clean(get_var(datasets, 2, "t2m"), "t2m")
        tp = clean(get_var(datasets, 11, "tp"), "tp")

        q = get_var(datasets, 6, "q")
        q1000 = clean(
            q.sel(isobaricInhPa=1000.0, method="nearest"),
            "q1000",
        )

        ds_out = xr.Dataset({
            "tcc": subset_india(tcc),
            "u10": subset_india(u10),
            "v10": subset_india(v10),
            "t2m": subset_india(t2m),
            "tp": subset_india(tp),
            "q1000": subset_india(q1000),
        })

        ds_out.attrs = {
            "title": "ECMWF AIFS forecast - India subset",
            "source": grib_path.name,
            "domain": (
                f"lat {LAT_MIN}-{LAT_MAX}N, "
                f"lon {LON_MIN}-{LON_MAX}E"
            ),
            "run_date": INPUT_DATE,
            "run_hour": RUN_HOUR,
        }

        print(f"  Saving NetCDF: {nc_path}")
        ds_out.to_netcdf(nc_path, engine="netcdf4")
        ds_out.close()

    finally:
        for ds in datasets:
            ds.close()


# ============================================================
# Download GRIB2 files and convert them to NetCDF
# ============================================================
steps = range(
    STEP_START, STEP_END + 1, STEP_INTERVAL
)
total = len(steps)

print(f"Forecast initialization: {INPUT_DATE} {RUN_HOUR}")
print(f"Processing {total} forecast steps")
print(f"Output folder: {OUTPUT_DIR.resolve()}")

for i, step in enumerate(steps, 1):
    fname = f"{INIT_TIME}-{step}h-oper-fc.grib2"
    s3_path = f"{BASE_S3}/{fname}"
    grib_local = Path(fname)
    nc_path = OUTPUT_DIR / fname.replace(
        ".grib2", "_india.nc"
    )

    print(f"\n[{i}/{total}] Forecast step: {step}h")

    # Resume support: skip completed outputs.
    if nc_path.exists() and nc_path.stat().st_size > 0:
        print(f"  Already exists; skipping: {nc_path}")
        continue

    try:
        print(f"  Downloading: s3://{s3_path}")
        fs.get(s3_path, str(grib_local))
        print(
            f"  Downloaded: "
            f"{grib_local.stat().st_size / 1e6:.1f} MB"
        )
    except Exception as exc:
        print(f"  Download failed: {exc}")
        if grib_local.exists():
            grib_local.unlink()
        continue

    try:
        grib_to_nc(grib_local, nc_path)
        print(f"  Saved: {nc_path}")
    except Exception as exc:
        print(f"  Conversion failed: {exc}")
        if nc_path.exists():
            nc_path.unlink()
    finally:
        if grib_local.exists():
            grib_local.unlink()
            print("  Temporary GRIB2 file deleted.")


# ============================================================
# Merge NetCDF files along a new time dimension
# ============================================================
MERGED_FILE = OUTPUT_DIR / (
    f"ecmwf_aifs_india_{INPUT_DATE}_{RUN_HOUR}_merged.nc"
)

nc_files = sorted(
    OUTPUT_DIR.glob("*-*h-oper-fc_india.nc"),
    key=lambda f: int(f.name.split("-")[-3].replace("h", "")),
)

if not nc_files:
    raise FileNotFoundError(
        f"No converted NetCDF files found in {OUTPUT_DIR}"
    )

print(f"\nFound {len(nc_files)} NetCDF files for merging.")

# Open one file at a time for metadata and concatenate lazily.
opened = []
try:
    for f in nc_files:
        opened.append(xr.open_dataset(f, engine="netcdf4"))

    # Each input represents a separate forecast step.
    ds_merged = xr.concat(
        opened,
        dim="time",
        combine_attrs="override",
    )

    ds_merged.attrs.update({
        "title": "ECMWF AIFS forecast - India subset (merged)",
        "source": "ECMWF Open Data S3, aifs-single 0p25",
        "run_date": INPUT_DATE,
        "run_hour": RUN_HOUR,
        "forecast_steps": (
            f"{STEP_START}h to {STEP_END}h "
            f"every {STEP_INTERVAL}h"
        ),
        "domain": "lat 6-38N, lon 66-100E",
        "n_files": len(nc_files),
    })

    print(ds_merged)

    encoding = {
        var: {"zlib": True, "complevel": 4}
        for var in ds_merged.data_vars
    }

    print(f"Saving merged file: {MERGED_FILE}")
    ds_merged.to_netcdf(
        MERGED_FILE,
        engine="netcdf4",
        encoding=encoding,
    )
    ds_merged.close()

finally:
    for ds in opened:
        ds.close()

print(
    f"\nMerged file saved: {MERGED_FILE} "
    f"({MERGED_FILE.stat().st_size / 1e6:.1f} MB)"
)

# Optional deletion of individual NetCDF files.
if DELETE_ORIGINALS:
    for f in nc_files:
        f.unlink(missing_ok=True)
    print("Individual NetCDF files deleted.")

print("\nFinal output files:")
for f in sorted(OUTPUT_DIR.glob("*.nc")):
    print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")

print("\nProcessing completed.")
