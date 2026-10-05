"""Audited PM2.5 Random Forest training and offline lag/lead features."""
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from .common import OUT, SITES, make_block_mask, timestamp_features
from .hour_median import hour_median
from .linear_interpolation import interpolate

def tabular_features(pm_masked, weather, cal, rows):
    # Offline imputation may use past AND future observations within a gap.
    lags = [-24, -6, -3, -2, -1, 0, 1, 2, 3, 6, 24]
    features = []
    for t, j, _ in rows:
        x = []
        for station in range(len(SITES)):
            for delta in lags:
                if station == j and delta == 0:
                    continue  # Never pass the target at its own hour.
                q = t + delta
                x.append(pm_masked[q, station] if 0 <= q < len(pm_masked) else np.nan)
        x.extend(weather[t])
        x.extend(cal[t])
        x.extend(float(j == site_index) for site_index in range(len(SITES)))
        features.append(x)
    return np.asarray(features, float)

def run_rf(times, pm, weather, masks):
    cal = timestamp_features(times)
    train_mask, train_rows = make_block_mask(pm, 0, 6552, 702, train=True)
    train_pm = pm.copy()
    train_pm[train_mask] = np.nan
    # Lags/leads must stay in the fitting split. In particular, the last
    # September windows must not consume October PM2.5 as future features.
    train_pm[6552:] = np.nan
    xtrain = tabular_features(train_pm, weather, cal, train_rows)
    ytrain = np.array([pm[t, j] for t, j, _ in train_rows])
    assert np.isfinite(ytrain).all()
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                          RandomForestRegressor(n_estimators=160, max_depth=16,
                                                min_samples_leaf=3, max_features=0.8,
                                                n_jobs=2, random_state=702))
    model.fit(xtrain, ytrain)
    joblib.dump(model, OUT/"random_forest.joblib")
    results = []
    for seed, (mask, rows) in masks.items():
        masked = pm.copy(); masked[mask] = np.nan
        # The RF comparison is evaluated using October PM2.5 context only;
        # no November/December measurements enter its future features.
        rf_context = masked.copy()
        rf_context[:6552] = np.nan
        rf_context[7296:] = np.nan
        pred = model.predict(tabular_features(rf_context, weather, cal, rows))
        baselines = {"Hour median": hour_median(pm, times, rows),
                     "Linear interpolation": interpolate(masked, rows),
                     "Random Forest": pred}
        for method, p in baselines.items():
            for (t, j, gap), yhat in zip(rows, p):
                results.append(dict(seed=seed, method=method, datetime_utc=times[t].isoformat(),
                                    site=SITES[j], gap_hours=gap, truth=pm[t, j], prediction=yhat))
    return results, len(train_rows)

