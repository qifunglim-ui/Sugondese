# REXharge 2026 MVP (Theme 1)

This MVP demonstrates an AI-based load shifting and peak shaving workflow for Malaysian commercial facilities with EV charging and solar PV.

## What it does

1. Loads all organizer `.xlsx` load profiles in this folder.
2. Cleans mixed schema formats into a unified time-series.
3. Forecasts demand with a machine learning model.
4. Detects peak periods and simulates:
   - automated load shifting
   - battery peak shaving (charged by solar surplus)
5. Compares estimated billing impact:
   - old MD rate: RM30.30/kW
   - new MD rate: RM97.06/kW
6. Provides dashboard charts and CSV export.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local Streamlit URL shown in terminal.

## Competition deliverables mapping

- **Working MVP Prototype**: `app.py` Streamlit app
- **Technical Report Inputs**:
  - Forecast MAE/RMSE
  - Baseline vs optimized MD (kW)
  - Estimated cost savings under new tariff
  - Battery sizing hint for target peak cap
- **Demo Video**: screen record walkthrough of the dashboard controls and output metrics

## Recommended report assumptions section

- Peak window default in prototype: weekdays 14:00-22:00
- Energy rate default: RM0.40/kWh (editable in UI)
- Strategy priority:
  1. shift flexible loads out of peak
  2. shave remaining peaks using battery dispatch

You can tune these to align with final competition assumptions and site constraints.
