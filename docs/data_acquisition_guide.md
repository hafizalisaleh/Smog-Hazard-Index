# Satellite & Reanalysis Data Acquisition Guide for Students

**Project:** Machine Learning for Ground-Level $\text{PM}_{2.5}$ Estimation & Development of Smog Hazard Index ($\text{SHI}$)  
**Institution:** EHDS Lab / SUPARCO Projects

This guide provides step-by-step instructions for acquiring the satellite and reanalysis inputs required for the research project.

---

## 1. Study Domain Bounding Box (Punjab Pilot Area)

Use these spatial bounding coordinates when querying or clipping datasets:
- **Minimum Latitude (South):** `31.10° N`
- **Maximum Latitude (North):** `32.35° N`
- **Minimum Longitude (West):** `73.80° E`
- **Maximum Longitude (East):** `74.65° E`
- **Time Period:** `2025-06-01` to `2026-05-31` (1 Full Annual Cycle)

---

## 2. MODIS AOD: MCD19A2 (1 km Daily Resolution)

The **MCD19A2 Version 6.1** product provides Multi-Angle Implementation of Atmospheric Correction (MAIAC) land Aerosol Optical Depth (AOD) at 1 km resolution from both the Terra and Aqua satellite overpasses.

### Option A: NASA AppEEARS (Recommended for Beginners)
1. Navigate to: [https://appeears.earthdatacloud.nasa.gov/](https://appeears.earthdatacloud.nasa.gov/)
2. Log in with your free **NASA Earthdata** account.
3. Click **Extract** $\to$ **Area Sample**.
4. Define your area using the pilot bounding box coordinates above.
5. Select the date range: `2025-06-01` to `2026-05-31`.
6. Add dataset: `MCD19A2.061`.
7. Select Layers:
   - `Optical_Depth_055` (AOD at 550nm green band)
   - `AOD_QA` (Quality Assurance mask)
8. Output format: **GeoTIFF** or **NetCDF4**, coordinate system **Geographic (EPSG:4326)**.
9. Submit and download when processing completes.

### Option B: NASA LAADS DAAC
- Portal: [https://ladsweb.modaps.eosdis.nasa.gov/](https://ladsweb.modaps.eosdis.nasa.gov/)
- Tile identifier for Punjab, Pakistan: **h24v05** and **h24v06**.

---

## 3. Sentinel-5P TROPOMI: $\text{NO}_2$ and $\text{SO}_2$

Sentinel-5P carries the TROPOspheric Monitoring Instrument (TROPOMI), providing daily global measurements of key smog precursor trace gases.

### Access via Copernicus Data Space Ecosystem (CDSE)
1. Portal: [https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/)
2. Register for a free Copernicus user account.
3. Open the **Copernicus Browser**.
4. Set Search Filters:
   - Mission: **Sentinel-5P**
   - Products:
     - `L2__NO2___` (Offline / Reprocessed tropospheric $\text{NO}_2$ vertical column density)
     - `L2__SO2___` (Offline $\text{SO}_2$ total column)
5. Set Geographic Filter to Punjab Pilot Domain.
6. Processing / Quality Control Rule:
   - Filter observations by `qa_value > 0.75` for $\text{NO}_2$ (excludes cloud-covered and snow/ice contaminated pixels).
   - Filter by `qa_value > 0.50` for $\text{SO}_2$.

---

## 4. ECMWF ERA5 Meteorological Reanalysis

ERA5 is the fifth generation ECMWF atmospheric reanalysis, providing global hourly meteorological fields.

### Access via Copernicus Climate Data Store (CDS API)
1. Register at: [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)
2. Install the Python CDS API client:
   ```bash
   pip install cdsapi
   ```
3. Set up your CDS API key in `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api/v2
   key: <YOUR-UID>:<YOUR-API-KEY>
   ```
4. Python extraction script for Punjab Pilot area:

```python
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '2m_temperature',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'boundary_layer_height',
            'surface_pressure',
            '2m_dewpoint_temperature',
        ],
        'year': ['2025', '2026'],
        'month': [f'{m:02d}' for m in range(1, 13)],
        'day': [f'{d:02d}' for d in range(1, 32)],
        'time': [f'{h:02d}:00' for h in range(24)],
        'area': [32.35, 73.80, 31.10, 74.65],  # North, West, South, East
    },
    'data/raw/meteorology/era5_punjab_hourly.nc'
)
```

### Calculating Derived Meteorological Parameters:
- **Relative Humidity ($RH$):** Derived from $2\text{m Temperature } (T)$ and $2\text{m Dewpoint } (T_d)$ using the Magnus formula:
  $$RH \approx 100 \times \exp\left(\frac{17.625 T_d}{243.04 + T_d} - \frac{17.625 T}{243.04 + T}\right)$$
- **Total Wind Speed ($WS$):**
  $$WS = \sqrt{u_{10}^2 + v_{10}^2}$$
- **Wind Direction ($WD$):**
  $$WD = \left(270 - \text{atan2}(v_{10}, u_{10}) \times \frac{180}{\pi}\right) \pmod{360}$$

---

## 5. CAMS Global Reanalysis ($\text{PM}_{2.5} / \text{PM}_{10}$)

The Copernicus Atmosphere Monitoring Service (CAMS) reanalysis provides aerosol chemical transport reanalysis.
- Dataset: `cams-global-reanalysis-eac4`
- Variables: `particulate_matter_2.5um`, `particulate_matter_10um`, `total_aerosol_optical_depth_550nm`
- Access via CDS API: specify dataset name `'cams-global-reanalysis-eac4'`.

---

## 6. Directory Structure Placement

Save your downloaded files into the project directories:
- Satellite files $\to$ `data/raw/satellite/`
- Meteorology files $\to$ `data/raw/meteorology/`

The automated ingestion pipeline (`python scripts/02_assemble_dataset.py`) will automatically scan these folders and collocate them with ground monitoring records.
