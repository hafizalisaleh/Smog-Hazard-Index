# Technical Methodology Note: Ground-Level $\text{PM}_{2.5}$ Estimation and Smog Hazard Index ($\text{SHI}$) Formulation

**Project:** Machine Learning for Ground-Level $\text{PM}_{2.5}$ Estimation & Development of Smog Hazard Index ($\text{SHI}$)  
**Institution:** EHDS Lab in collaboration with SUPARCO  
**Study Scope:** Pilot Study (Lahore, Gujranwala, Sheikhupura) & Extension Roadmap to Four Provinces (Punjab, Khyber Pakhtunkhwa, Upper Sindh, Islamabad Capital Territory)

---

## 1. Executive Summary & Objective

Severe atmospheric smog has become an annual public health crisis across the Indo-Gangetic Plain and eastern Pakistan. Traditional ground-based air quality monitoring relies on sparse, point-based reference stations that fail to capture the high spatial heterogeneity of aerosol plumes.

This project develops an end-to-end, reproducible machine-learning framework to estimate continuous, ground-level $\text{PM}_{2.5}$ mass concentrations (1 km daily resolution) by integrating satellite aerosol optical depth ($\text{AOD}$), tropospheric trace gases ($\text{NO}_2$, $\text{SO}_2$), meteorological reanalysis ($\text{ERA5}$), and ground truth observations from EPA monitoring networks. 

Using the validated estimation fields, we formulate a prototype **Smog Hazard Index ($\text{SHI}$)** that combines particulate loading, column aerosol optical depth, gaseous precursors, and boundary-layer stagnation dynamics into a standardized 5-tier hazard scale (**Low $\to$ Extreme**).

---

## 2. Multi-Sensor Data Fusion Framework

| Category | Dataset / Sensor | Spatial Resolution | Temporal Frequency | Parameters Used |
| :--- | :--- | :--- | :--- | :--- |
| **Satellite Aerosol** | MODIS (Terra/Aqua) MCD19A2 | 1 km | Daily overpass (~10:30 & 13:30 LT) | $\text{AOD}_{550\text{nm}}$ (Green band), Cloud Mask |
| **Satellite Trace Gases**| Sentinel-5P TROPOMI | $3.5 \times 5.5\text{ km}^2$ | Daily | Tropospheric $\text{NO}_2$ column, $\text{SO}_2$ column |
| **Reanalysis Aerosol** | CAMS Global Reanalysis | 0.75° (~80 km) | 3-hourly / Daily | Reanalysis $\text{PM}_{2.5}$, $\text{PM}_{10}$ |
| **Meteorology** | ECMWF ERA5 | 0.25° (~28 km) | Hourly / Daily | 2m Temp ($T_{2\text{m}}$), $10\text{m}$ $U/V$ winds ($U_{10}, V_{10}$), Wind Speed ($WS$), Relative Humidity ($RH$), Boundary Layer Height ($PBLH$), Surface Pressure ($SP$) |
| **Ground Truth** | Punjab EPA Monitoring Stations | In-situ point | Hourly (aggregated to 24h daily) | $\text{PM}_{2.5}$ ($\mu\text{g/m}^3$) across 12 stations |

### Ground Truth Quality Assurance Protocol
Ground observations undergo systematic scientific filtering:
1. **Sensor Dropout Filter:** Removal of non-positive readings ($\le 0\ \mu\text{g/m}^3$).
2. **Sensor Spike Filter:** Removal of readings $\ge 700\ \mu\text{g/m}^3$ that exceed $5\times$ the local 14-day centred rolling median.
3. **Temporal Completeness:** Daily means are accepted only when $\ge 75\%$ of hours in a day ($\ge 18$ hours) are valid (US EPA / WHO completeness standard).
4. **Short-Gap Interpolation:** Internal missing gaps $\le 6$ consecutive hours are linearly interpolated; longer gaps are preserved as missing to avoid synthetic distortion.

---

## 3. Physical Feature Engineering & Stagnation Dynamics

Atmospheric smog in the Punjab basin is heavily governed by the interaction between emissions and meteorological ventilation. We engineer physically grounded indicators:

### 3.1. Ventilation Index ($\text{VI}$)
$$\text{VI} = WS \times PBLH \quad [\text{m}^2/\text{s}]$$
Low ventilation ($\text{VI} < 1000\ \text{m}^2/\text{s}$) denotes poor vertical mixing and weak horizontal transport, trapping particulates near the surface.

### 3.2. Air Stagnation Index ($\text{ASI}$)
Multi-criteria continuous score ($0 \text{ to } 1$) computed from:
- Wind stagnation: $f(WS) = \max(0, 1 - WS / 5.0)$
- Shallow boundary layer: $f(PBLH) = \max(0, 1 - PBLH / 1500.0)$
- Condensation factor: $f(RH) = RH / 100.0$

$$\text{ASI} = 0.45 \cdot f(WS) + 0.35 \cdot f(PBLH) + 0.20 \cdot f(RH)$$

### 3.3. Column-to-Surface Extinction Ratio
$$\text{Extinction Ratio} = \frac{\text{AOD}}{PBLH / 1000} \quad [\text{km}^{-1}]$$
Normalizes total vertical column optical depth by the depth of the active mixing layer, representing effective near-surface aerosol concentration.

---

## 4. Model Architecture & Spatial Cross-Validation

To ensure that performance metrics reflect true generalization to unmonitored geographic coordinates, models are evaluated using **Spatial Group K-Fold Cross-Validation** (grouped by monitoring station ID). Random K-Fold CV is strictly rejected due to spatial-temporal autocorrelation leading to severe data leakage.

### Model Benchmarking Summary (Spatial CV)

| Model Architecture | Cross-Validation $R^2$ | RMSE ($\mu\text{g/m}^3$) | MAE ($\mu\text{g/m}^3$) | Mean Bias (MBE) | Index of Agreement (IOA) | FAC2 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Baseline (Ridge)** | 0.629 | 55.19 | 28.90 | +1.77 | 0.928 | 94.6% |
| **Random Forest Regressor** | 0.961 | 17.99 | 13.95 | -1.70 | 0.990 | 97.1% |
| **XGBoost Regressor (Best)** | **0.962** | **17.64** | **13.36** | **-1.40** | **0.990** | **97.6%** |

*Finding:* XGBoost outperforms Random Forest and the Linear Baseline, achieving the lowest spatial RMSE ($17.64\ \mu\text{g/m}^3$) and near-zero normalized bias ($-1.18\%$).

---

## 5. Prototype Smog Hazard Index ($\text{SHI}$) Formulation

Unlike conventional single-pollutant Air Quality Indices (AQI), the **Smog Hazard Index ($\text{SHI}$)** incorporates particulate toxicity, regional optical turbidity, secondary aerosol gaseous precursors, and physical dispersion stagnation into a multi-attribute composite score scaled from $0$ to $100$:

$$\text{SHI} = 100 \times \sum_{i=1}^{5} w_i \cdot S_i$$

### Sub-Index Weights & Standardization Breakpoints

| Component | Variable | Weight ($w_i$) | Baseline ($S=0.0$) | Standard Target ($S=0.5$) | Extreme Crisis ($S=1.0$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Particulate Matter | Ground $\text{PM}_{2.5}$ | **0.50** | $5.0\ \mu\text{g/m}^3$ | $35.0\ \mu\text{g/m}^3$ (Pak-EPA) | $350.0\ \mu\text{g/m}^3$ |
| Aerosol Column | MODIS $\text{AOD}_{550\text{nm}}$ | **0.20** | 0.10 | 0.50 | 2.50 |
| Reactive Nitrogen | Sentinel-5P $\text{NO}_2$ | **0.10** | $2.0 \times 10^{-5}\ \text{mol/m}^2$ | $1.5 \times 10^{-4}\ \text{mol/m}^2$ | $4.5 \times 10^{-4}\ \text{mol/m}^2$ |
| Sulfur Dioxide | Sentinel-5P $\text{SO}_2$ | **0.05** | $1.0 \times 10^{-5}\ \text{mol/m}^2$ | $8.0 \times 10^{-5}\ \text{mol/m}^2$ | $2.5 \times 10^{-4}\ \text{mol/m}^2$ |
| Atmospheric Trapping | Stagnation Index | **0.15** | 0.00 (Vigorous mixing) | 0.50 (Moderate calm) | 1.00 (Inversion / calm) |

### Severity Categorization and Public Action Triggers

```
[0 -------- 25 -------- 50 -------- 70 -------- 85 -------- 100]
     Low       Moderate     High      Very High     Extreme
```

1. **Low ($\text{SHI} < 25$):** Good air quality; routine outdoor activities.
2. **Moderate ($25 \le \text{SHI} < 50$):** Acceptable; sensitive individuals monitor respiratory symptoms.
3. **High ($50 \le \text{SHI} < 70$):** Unhealthy; N95 masks advised; outdoor sports restricted.
4. **Very High ($70 \le \text{SHI} < 85$):** Very unhealthy; commercial operating hours curtailed; primary schools move online.
5. **Extreme ($85 \le \text{SHI} \le 100$):** Hazardous smog emergency; odd-even vehicle restrictions; shutdown of heavy-emitting brick kilns and non-compliant industrial boilers.

---

## 6. Recommendations for Full Four-Province Rollout

To expand from the pilot domain (Lahore, Gujranwala, Sheikhupura) to the full study across **Punjab, Khyber Pakhtunkhwa (KP), Upper Sindh, and Islamabad Capital Territory (ICT)**:

### 6.1. Regional Topographic & Atmospheric Stratification
- **ICT (Margalla Foothills):** Valley-trapping effects require localized topographic lapse rate corrections.
- **Khyber Pakhtunkhwa (Peshawar Basin):** High natural dust loading mixed with vehicular emissions; requires shortwave infrared AOD separation to isolate fine-mode vs coarse-mode dust.
- **Upper Sindh (Sukkur, Larkana, Shikarpur):** High thermal convection in summer, followed by severe agricultural biomass burning and cold-air drainage in winter.
- **Punjab Plain:** Dominated by post-monsoon rice straw burning (Oct-Nov) and dense nocturnal radiation fog (Dec-Jan).

### 6.2. Ground Sensor Calibration & Network Harmonization
- Deploy **low-cost sensor calibration transfers**: calibrate Plantower / Sensirion optical particle counters against EPA reference beta-attenuation monitors (BAM-1020) using relative-humidity hygroscopic growth corrections ($f(RH) = 1 + a(RH/100)^b$).

### 6.3. Cloud & Fog Gap-Filling Architecture
- In peak smog season, optical AOD is frequently obscured by cloud or heavy fog. Deploy a **two-stage ML architecture**:
  - *Stage 1 (AOD available):* Full feature model (Satellite AOD + Trace Gases + ERA5).
  - *Stage 2 (AOD cloudy):* Meteorological + CAMS reanalysis fallback model ensuring 100% daily gridded spatial coverage.

---

## 7. Journal Publication Roadmap

Target Journals: *Atmospheric Environment*, *Remote Sensing of Environment*, or *Science of The Total Environment*.
- **Title:** "High-Resolution Ground-Level $\text{PM}_{2.5}$ Estimation and Development of a Multi-Pollutant Smog Hazard Index ($\text{SHI}$) in Eastern Pakistan Using Satellite Observations and Machine Learning."
- **Key Contributions:**
  1. First high-resolution (1 km) multi-station machine learning retrieval combining MODIS MAIAC, Sentinel-5P, and ERA5 for the Lahore-Gujranwala-Sheikhupura industrial corridor.
  2. Spatial cross-validation showing spatial generalization to unmonitored sites ($R^2 = 0.96$).
  3. Novel formulation of the Smog Hazard Index ($\text{SHI}$) integrating meteorological stagnation with multi-pollutant satellite observations.
