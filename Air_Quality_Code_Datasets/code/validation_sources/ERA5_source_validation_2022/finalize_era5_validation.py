import csv,json,hashlib,zipfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs'/'era5_source_validation'
source={r['datetime_utc']:r for r in csv.DictReader((OUT/'ERA5_2022_CST_year_original_UTC.csv').open())}
src=ROOT/'outputs/aqs_gmt_verified_2022/AQS_ERA5_2022_hourly_panel_UTC_GMT_VERIFIED.csv'
rows=list(csv.DictReader(src.open()));old={ (r['site'],r['datetime_utc']):dict(r) for r in rows}
a='2022-11-06T06:00:00Z';b='2022-11-06T07:00:00Z'
vars=['tp','u10','v10','d2m','t2m','sp','blh','temp_c','dewpoint_c','surface_pressure_hpa','precip_mm','wind_speed','relative_humidity']
tol={'u10':1e-5,'v10':1e-5,'t2m':1e-4,'d2m':1e-4,'sp':.01,'blh':.001,'tp':1e-9}
changed=[]
for r in rows:
 t=r['datetime_utc']
 if t in [a,b]:
  other=old[(r['site'],b if t==a else a)]
  for v in vars:
   k='era5_'+v
   if r[k]!=other[k]:changed.append({'site':r['site'],'datetime_utc':t,'column':k,'previous_value':r[k],'corrected_value':other[k]})
   r[k]=other[k]
 r['era5_grid_latitude']=source[t]['latitude'];r['era5_grid_longitude']=source[t]['longitude']
 r['era5_source_file']=source[t]['source_file'];r['era5_source_utc_verified']=1
 r['era5_spatial_basis']='shared single grid point'
 r['era5_autumn_order_corrected']=int(t in [a,b])
 for v,tolerance in tol.items():assert abs(float(r['era5_'+v])-float(source[t][v]))<=tolerance
 for k in ['datetime_utc','CO','NO2','O3','PM2.5','SO2']:assert r[k]==old[(r['site'],t)][k]
def write(name,data):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
write('AQS_ERA5_2022_UTC_SOURCE_VERIFIED.csv',rows);write('autumn_correction_log.csv',changed)
summary=json.loads((OUT/'results.json').read_text())
summary.update({'corrected_station_hours':10,'changed_weather_cells':len(changed),'post_correction_variable_hour_checks':8760*7,'post_correction_checks_passed':8760*7,'pollutant_values_unchanged':True})
(OUT/'results.json').write_text(json.dumps(summary,indent=2))
report='''# ERA5 source validation and autumn alignment correction

## Result

Original ERA5 source comparison is complete for all 8,760 hours in the 2022 CST-year panel, including the first six UTC hours of 2023. The previous panel reversed the instantaneous weather values at 2022-11-06 06:00 and 07:00 UTC. The corrected panel passes all 61,320 variable-hour comparisons against the uploaded source files within explicit CSV-rounding tolerances.

This correction affects two hours across five stations (10 station-hour rows). All pollutant measurements and AQS timestamps are unchanged. The old AQS GMT validation was correct, but it did not establish the ERA5 alignment. The original claim that all weather timestamps were publication-safe was too strong.

## Direct evidence

| UTC | Civil time | Original t2m (K) | Previous panel t2m (K) |
|---|---|---:|---:|
| 2022-11-06 06:00 | 01:00 CDT (-05:00) | 287.87548828125 | 287.39404 |
| 2022-11-06 07:00 | 01:00 CST (-06:00) | 287.39404296875 | 287.8755 |

The earlier reconstruction assigned the two timezone-free 01:00 rows in the wrong order. Original valid_time identifies the correct physical hour unambiguously. The corrected panel exchanges all weather fields, including their derived values, between these two hours. It retains existing CSV precision elsewhere. The separate original-UTC CSV contains the source numeric values at their decoded precision.

## Spatial provenance

All 71 supplied monthly ZIP containers were opened. Each contains NetCDF components for instantaneous and accumulated variables. Each component has one latitude (29.5 degrees north) and one longitude (-98.5 degrees east), on a grid with recorded 0.25-degree spacing. The uploaded source is a single grid point, not a multi-cell county average. Its series matches the supplied panel except for the corrected autumn reversal and ordinary rounding.

Use the description: shared ERA5 meteorological covariates from the grid point at 29.5 N, 98.5 W. Do not call these station-specific weather measurements or a Bexar County spatial average. This is suitable as a documented shared covariate design; a study claiming station-specific meteorology requires a different spatial extraction.

## Coverage and checks

- 71 of 72 expected monthly files for 2019–2024 received. October 2023 is absent. This does not affect 2022.
- Every supplied month contains its expected number of consecutive hourly timestamps. Instantaneous and accumulated components have identical time coordinates and grids. All seven decoded weather variables are finite.
- Source time units: seconds since 1970-01-01; proleptic Gregorian calendar. Conversion uses UTC, not a local-clock inference.
- Full numeric comparison was performed for the 2022 CST-year panel, not the entire 2019–2024 historical CSV.
- Year definition: 2022-01-01 06:00 UTC through 2023-01-01 05:00 UTC inclusive. There are 8,760 distinct UTC hours and 43,800 station-hour rows.
- Compared variables: u10, v10, t2m, d2m, sp, blh and tp. Temperature tolerance 0.0001 K; wind components 0.00001 m/s; pressure 0.01 Pa; boundary-layer height 0.001 m; precipitation 0.000000001 m. Tolerances accommodate exported decimal rounding, not hour-scale mismatches.
- Precipitation happened to have no detectable difference at the two reversed hours. Instantaneous variables each failed at precisely those two hours before correction and pass afterward.

## Units and remaining scientific checks

Source metadata identifies u10/v10 in m/s, t2m/d2m in kelvin, sp in pascals, blh in metres and tp in metres. tp has accumulation step type; the other six have instantaneous step type. Unit conversions previously checked are retained. Relative humidity is derived, not a downloaded variable; its exact generation formula remains undocumented without the CSV-generation script. Accumulation interval and forecast availability must be documented before precipitation is used in a forecasting design. Exact matching at valid_time does not mean hourly-average AQS and instantaneous ERA5 have identical temporal support.

This resolves source UTC alignment and grid identity. It is not a declaration that the full research dataset, missingness policy or modelling protocol is publication-ready. The prior AQS metadata audit still applies; no imputation has been performed. Original source records and reported negative pollutant values remain unchanged.

## Files

- AQS_ERA5_2022_UTC_SOURCE_VERIFIED.csv: corrected panel with grid coordinates, source filenames and correction flags.
- ERA5_2022_CST_year_original_UTC.csv: source weather values at original UTC.
- autumn_correction_log.csv: every changed weather cell.
- ERA5_panel_source_comparison.csv and ERA5_comparison_summary.csv: BEFORE-correction comparison, including failures.
- results.json: before-correction summaries plus after-correction verification totals.
- uploaded_file_inventory.csv: monthly coverage and SHA-256 source hashes.
- read_era5_nc.py, validate_era5_sources.py, finalize_era5_validation.py: reproducible analysis scripts. Run using the same runtime with NumPy and its VTK NetCDF shared library; arrange input upload and downloads directories as in the scripts. Source files themselves are not duplicated in this package.
'''
(OUT/'ERA5_validation_report.md').write_text(report)
for n in ['read_era5_nc.py','validate_era5_sources.py','finalize_era5_validation.py']:shutil.copy2(ROOT/n,OUT/n)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='sha256_manifest.json'}
(OUT/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(OUT.parent/'ERA5_source_validation_2022.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.iterdir():z.write(p,p.name)
with zipfile.ZipFile(OUT.parent/'ERA5_source_validation_2022.zip') as z:assert z.testzip() is None
print(json.dumps({k:summary[k] for k in ['corrected_station_hours','changed_weather_cells','post_correction_checks_passed','pollutant_values_unchanged']}))
