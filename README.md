# roi-ltv-estimator

A Python/Tkinter GUI to forecast LTV (D28/D90/D360) from a baseline ROI curve and compute the **required ROI** at early checkpoints (D0/D3/D7/D14/D28) to hit a chosen LTV target. Supports **paid-only** or **total** (paid × organic uplift) targets.

## Features
- Paste baseline ROI: `D0 D3 D7 D14 D28` (percent values).
- Extrapolate tail to **D90** and **D360** using:
  - Power-law: `log(1+ROI_t) = a + b·log(t+1)`
  - Log-polynomial (quadratic) for extra flexibility.
- Compute required ROI at early days to reach **LTV@D28/D90/D360**.
- Organic uplift:
  - Toggle whether the target includes organic uplift or is paid-only.
  - Optionally apply post-calc uplift to show total outcome.
- Clear table output and model diagnostics (params, log-R²).

## Quick start
```bash
python roi_estymator.py
