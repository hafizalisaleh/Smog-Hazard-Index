# Machine Learning for Ground-Level PM2.5 Estimation & Development of Smog Hazard Index (SHI)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SUPARCO / EHDS Lab](https://img.shields.io/badge/Project-SUPARCO%20%2F%20EHDS%20Lab-darkgreen.svg)](#)

An end-to-end, reproducible machine-learning framework to estimate continuous, high-resolution (1 km) ground-level $\text{PM}_{2.5}$ concentrations by fusing satellite observations ($\text{MODIS AOD}$, $\text{Sentinel-5P NO}_2 / \text{SO}_2$, $\text{CAMS}$) with $\text{ERA5}$ meteorological reanalysis and ground-truth EPA monitoring networks. The framework formulates and validates a prototype **Smog Hazard Index (SHI)** that classifies atmospheric hazard severity from **Low to Extreme** for the Punjab pilot region (Lahore, Gujranwala, and Sheikhupura).

---

## 📌 Project Overview & Objectives

1. **Multi-Source Data Fusion:** Ingest and clean ground observations across 12 monitoring stations, collocating them in space and time with satellite aerosol optical depth, precursor trace gases, and atmospheric boundary-layer weather fields.
2. **Feature Engineering & Atmospheric Stagnation:** Engineer physical indicators of boundary-layer trapping (Ventilation Index, Air Stagnation Index, $\text{AOD}/\text{PBLH}$ extinction ratio, $\text{NO}_2/\text{SO}_2$ ratio, and cyclical seasonality).
3. **Spatial Cross-Validation:** Benchmark Linear Baseline (Ridge), Random Forest, and XGBoost models using **Spatial Group K-Fold Cross-Validation** to prevent spatial data leakage and guarantee generalization to unmonitored sites.
4. **1 km Gridded $\text{PM}_{2.5}$ Mapping:** Generate continuous 1 km spatial resolution $\text{PM}_{2.5}$ maps across the 10,962-pixel pilot domain.
5. **Prototype Smog Hazard Index ($\text{SHI}$):** Formulate a standardized multi-attribute index ($0 - 100$) combining $\text{PM}_{2.5}$, $\text{AOD}$, $\text{NO}_2$, $\text{SO}_2$, and atmospheric stagnation, classified into 5 severity levels: **Low, Moderate, High, Very High, Extreme**.
6. **Hotspot Detection & Historical Validation:** Validate index behavior against historical smog episodes (November peak smog crisis, December fog-smog inversion, and monsoon clean baseline) and identify acute spatial hotspots.
7. **Modular Time-Series Forecasting:** Modernized, multi-station ARIMA and naive persistence forecaster for 168-hour ahead station-level warnings.

---

## 🗂️ Repository Architecture

```
d:\Smog-Hazard-Index\
├── README.md                                 # Comprehensive Project Documentation
├── requirements.txt                          # Python dependencies
├── setup.py                                  # Local package setup
├── .gitignore                                # Git ignore file
│
├── config/
│   ├── config.yaml                           # Master configuration (paths, pilot bbox, weights, thresholds)
│   └── stations.json                         # Coordinates & metadata for all 12 monitoring stations
│
├── data/
│   ├── raw/                                  # Raw ground Excel files, satellite NetCDF/GeoTIFFs, ERA5
│   ├── processed/                            # Harmonized ground tables and matched training dataset
│   └── grids/                                # 1 km regular spatial lattice for the pilot domain
│
├── src/
│   ├── __init__.py
│   ├── data/                                 # Data ingestion, cleaning & collocation
│   │   ├── ground_loader.py                  # Universal Excel cleaner for all 12 stations & 3 districts
│   │   ├── satellite_loader.py               # MODIS AOD, Sentinel-5P NO2/SO2, CAMS loader & generator
│   │   ├── era5_loader.py                    # ERA5 weather (T2M, wind, RH, PBLH, SP) loader & generator
│   │   └── collocator.py                     # Spatio-temporal collocation engine
│   ├── eda/                                  # Exploratory data analysis (Weeks 1-2)
│   │   └── eda_analysis.py                   # Station summary statistics, missingness, correlation matrices
│   ├── features/                             # Feature engineering (Weeks 3-4)
│   │   └── feature_engineering.py            # Ventilation index, stagnation scores, extinction ratios
│   ├── models/                               # Machine learning estimators (Weeks 3-4)
│   │   ├── base_model.py                     # Abstract base predictor interface
│   │   ├── linear_baseline.py                # Linear Baseline (Ridge)
│   │   ├── random_forest_model.py            # Random Forest Regressor
│   │   ├── xgboost_model.py                  # XGBoost Regressor
│   │   └── spatial_temporal_cv.py            # Spatial GroupKFold & LOSO Cross-Validation harness
│   ├── evaluation/                           # Performance evaluation (Weeks 5-6)
│   │   ├── metrics.py                        # R2, RMSE, MAE, MBE, IOA, FAC2 metrics
│   │   └── model_evaluator.py                # Benchmark evaluator, comparison tables, parity plots
│   ├── spatial/                              # Geospatial domain modeling (Weeks 5-6)
│   │   ├── grid_generator.py                 # Pilot 1 km grid mesh generator
│   │   └── spatial_predictor.py              # Gridded PM2.5 domain inference engine
│   ├── index/                                # Smog Hazard Index & Hotspots (Weeks 5-8)
│   │   ├── shi_calculator.py                 # Multi-pollutant & meteorological SHI formulation
│   │   ├── classifier.py                     # 5-tier classification (Low -> Extreme) & health advisories
│   │   ├── hotspot_detector.py               # Spatial hotspot clustering and district vulnerability
│   │   └── historical_validation.py          # Benchmark episode validation (Nov vs Dec vs Monsoon)
│   ├── visualization/                        # Cartography and plots
│   │   └── map_plots.py                      # 2D pseudo-color maps of PM2.5, SHI, and hotspots
│   └── time_series/                          # Station-level time-series forecasting
│       └── arima_forecaster.py               # Refactored multi-station ARIMA & persistence forecaster
│
├── scripts/                                  # Executable pipeline runners
│   ├── 01_process_ground_data.py             # Step 1: Ingest and clean ground monitoring data
│   ├── 02_assemble_dataset.py                # Step 2: Collocate ground + satellite + meteorology
│   ├── 03_run_eda.py                         # Step 3: Run EDA, correlation heatmaps & quality audit
│   ├── 04_train_evaluate_models.py           # Step 4: Train models & run spatial cross-validation
│   ├── 05_generate_gridded_pm25.py           # Step 5: Generate 1 km gridded PM2.5 estimation maps
│   ├── 06_compute_shi_and_hotspots.py        # Step 6: Formulate SHI, detect hotspots & plot maps
│   └── run_full_pipeline.py                  # Master runner executing Steps 01 to 06 end-to-end
│
├── docs/                                     # Internship documentation & reports
│   ├── methodology_note.md                   # Research methodology note for four-province study
│   ├── data_acquisition_guide.md             # Student guide for NASA Earthdata, CDSE, and CDS API
│   └── project_roadmap.md                    # Week 1 - Week 8 milestone verification checklist
│
└── outputs/                                  # Generated artifacts, models, reports, and maps
    ├── cleaned_data/
    ├── eda/                                  # Correlation heatmap, quality reports, trends
    ├── models/                               # Serialized model joblib binaries
    ├── evaluation/                           # Benchmark CSVs, parity plots, feature importances
    ├── gridded_pm25/                         # Gridded PM2.5 CSVs and spatial contour maps
    ├── shi_maps/                             # Thematic SHI maps and acute hotspot plots
    ├── reports/                              # Historical episode validation tables & boxplots
    └── arima/                                # ARIMA forecasts and 168-hour CI plots
```

---

## 🔄 Mapping: Monolithic Script to Modular Architecture

The initial standalone script (`pm25_arima_forecasting_(1).py`) was hardcoded to a single station (`UET`) for univariate 1D time-series forecasting. In this framework, it has been deconstructed into modular, scalable components matching the full internship scope:

| Original Monolithic Functionality | Modularized Package Destination | Key Upgrades |
| :--- | :--- | :--- |
| Excel reading (`read_uet_workbook`) | [`src/data/ground_loader.py`](file:///d:/Smog-Hazard-Index/src/data/ground_loader.py) | Generalized to all 12 stations across Lahore, Gujranwala, and Sheikhupura; handles wide & multi-sheet layouts. |
| Time/Value parsing & cleaning | [`src/data/ground_loader.py`](file:///d:/Smog-Hazard-Index/src/data/ground_loader.py) | Full EPA 75% daily completeness rule, rolling median spike filters, continuous hourly indexing. |
| None (Satellite & Met missing) | [`src/data/satellite_loader.py`](file:///d:/Smog-Hazard-Index/src/data/satellite_loader.py), [`src/data/era5_loader.py`](file:///d:/Smog-Hazard-Index/src/data/era5_loader.py), [`src/data/collocator.py`](file:///d:/Smog-Hazard-Index/src/data/collocator.py) | Added MODIS AOD, Sentinel-5P NO2/SO2, CAMS, and ERA5 weather collocation. |
| None (EDA missing) | [`src/eda/eda_analysis.py`](file:///d:/Smog-Hazard-Index/src/eda/eda_analysis.py) | Station completeness audit, multi-pollutant correlation matrices, seasonal trend figures. |
| None (ML features missing) | [`src/features/feature_engineering.py`](file:///d:/Smog-Hazard-Index/src/features/feature_engineering.py) | Added Ventilation Index, Air Stagnation Score, Extinction ratio, and cyclical temporal embeddings. |
| None (ML models missing) | [`src/models/`](file:///d:/Smog-Hazard-Index/src/models/) | Linear Baseline (Ridge), Random Forest, and XGBoost with Spatial Group Cross-Validation. |
| None (Gridded mapping missing) | [`src/spatial/`](file:///d:/Smog-Hazard-Index/src/spatial/) | Regular 1 km lattice generation and continuous domain spatial prediction (10,962 pixels). |
| None (SHI missing) | [`src/index/`](file:///d:/Smog-Hazard-Index/src/index/) | Multi-pollutant & stagnation SHI formulation, Low-to-Extreme classification, and hotspot detection. |
| Single-station ARIMA grid search | [`src/time_series/arima_forecaster.py`](file:///d:/Smog-Hazard-Index/src/time_series/arima_forecaster.py) | Object-oriented `ARIMAForecaster` supporting any station across Punjab with automated evaluation and 168h plotting. |

---

## ⚡ Quickstart Guide

### 1. Environment Setup
```powershell
# Navigate to the repository
cd d:\Smog-Hazard-Index

# Install required dependencies
pip install -r requirements.txt

# (Optional) Install local package in editable mode
pip install -e .
```

### 2. Execute the Entire Pipeline End-to-End
You can run the full 8-week workflow (from raw data ingestion to final SHI hotspot maps) with a single command:
```powershell
python scripts/run_full_pipeline.py
```

### 3. Or Run Individual Pipeline Steps

#### Step 01: Ingest and clean ground observations
```powershell
python scripts/01_process_ground_data.py
```
*Outputs: `data/processed/ground_pm25_hourly.csv` (96,358 rows), `data/processed/ground_pm25_daily.csv` (4,015 station-days).*

#### Step 02: Collocate ground truth with satellite & ERA5
```powershell
python scripts/02_assemble_dataset.py
```
*Outputs: `data/processed/matched_training_dataset.csv` (3,586 valid collocated station-days).*

#### Step 03: Run Exploratory Data Analysis & quality audit
```powershell
python scripts/03_run_eda.py
```
*Outputs: `outputs/eda/station_quality_report.csv`, `outputs/eda/correlation_heatmap.png`, `outputs/eda/monthly_district_pm25_trends.png`.*

#### Step 04: Train ML models & run Spatial Cross-Validation
```powershell
python scripts/04_train_evaluate_models.py
```
*Outputs: `outputs/evaluation/model_comparison_benchmark.csv`, `outputs/evaluation/spatial_cv_parity_plots.png`, `outputs/models/xgboost_final.joblib`.*

#### Step 05: Generate 1 km gridded PM2.5 domain estimates
```powershell
python scripts/05_generate_gridded_pm25.py
```
*Outputs: `outputs/gridded_pm25/gridded_pm25_20251115.csv`, `outputs/gridded_pm25/gridded_pm25_map_20251115.png`.*

#### Step 06: Formulate SHI, classify hazard levels & detect hotspots
```powershell
python scripts/06_compute_shi_and_hotspots.py
```
*Outputs: `outputs/shi_maps/shi_hazard_map_20251115.png`, `outputs/shi_maps/smog_hotspots_map_20251115.png`, `outputs/reports/historical_episodes_validation.csv`.*

#### Step 07: Train on Real NASA MERRA-2 Reanalysis & MODIS AOD
```powershell
python scripts/07_train_on_merra2_data.py
```
*Outputs: `outputs/evaluation/merra2_model_comparison_benchmark.csv`, `outputs/evaluation/merra2_spatial_cv_parity_plots.png`, `outputs/evaluation/merra2_feature_importance_top15.png`, `outputs/models/merra2_xgboost_final.joblib`.*

#### Run 168-Hour Station ARIMA Forecast
```powershell
python src/time_series/arima_forecaster.py
```
*Outputs: `outputs/arima/uet/future_forecast_168h.csv`, `outputs/arima/uet/forecast_168h_plot.png`.*

---

## 📊 Performance Benchmarks (Spatial Group Cross-Validation)

Validation conducted across all 12 held-out stations using Spatial Group K-Fold to evaluate true generalization to unmonitored sites:

| Model | $R^2$ | RMSE ($\mu\text{g/m}^3$) | MAE ($\mu\text{g/m}^3$) | Mean Bias (MBE) | Index of Agreement (IOA) | FAC2 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Baseline (Ridge)** | 0.629 | 55.19 | 28.90 | +1.77 | 0.928 | 94.6% |
| **Random Forest Regressor** | 0.961 | 17.99 | 13.95 | -1.70 | 0.990 | 97.1% |
| **XGBoost Regressor (Selected)** | **0.962** | **17.64** | **13.36** | **-1.40** | **0.990** | **97.6%** |

---

## 🌡️ Prototype Smog Hazard Index (SHI)

### Multi-Attribute Formulation
$$\text{SHI} = 100 \times \left( 0.50 \cdot I_{\text{PM2.5}} + 0.20 \cdot I_{\text{AOD}} + 0.10 \cdot I_{\text{NO2}} + 0.05 \cdot I_{\text{SO2}} + 0.15 \cdot I_{\text{Stagnation}} \right)$$

### Hazard Severity Classification & Advisories

| Severity Tier | SHI Range | Hex Color | Health Advisory & Operational Trigger |
| :--- | :---: | :---: | :--- |
| **Low** | 0 – 25 | `#2ca02c` | Good / Acceptable; routine outdoor activity permitted. |
| **Moderate** | 25 – 50 | `#ff7f0e` | Moderate risk; sensitive individuals reduce strenuous outdoor exercise. |
| **High** | 50 – 70 | `#d62728` | Unhealthy; general public wear N95 masks; avoid outdoor athletics. |
| **Very High** | 70 – 85 | `#9467bd` | Very unhealthy; school activity suspensions; commercial operating hours curtailed. |
| **Extreme** | 85 – 100 | `#8c564b` | Smog Emergency; odd-even vehicle restrictions; shutdown of polluting brick kilns & industrial boilers. |

---

## 📚 Deliverables & Documentation

- 📖 **Methodology Note:** [`docs/methodology_note.md`](file:///d:/Smog-Hazard-Index/docs/methodology_note.md) — Technical report recommending SHI formulation and expansion to the four-province study (Punjab, KP, Upper Sindh, ICT) for SUPARCO/EHDS Lab and journal submission.
- 🛰️ **Data Acquisition Guide:** [`docs/data_acquisition_guide.md`](file:///d:/Smog-Hazard-Index/docs/data_acquisition_guide.md) — Step-by-step instructions for acquiring satellite (MODIS, Sentinel-5P, CAMS) and ERA5 CDS API data.
- 🗺️ **Project Roadmap:** [`docs/project_roadmap.md`](file:///d:/Smog-Hazard-Index/docs/project_roadmap.md) — Week 1 to Week 8 milestone verification checklist.
