"""Reproduce the 2022 AQS audit without modifying source files or imputing values."""
import csv, json, math, hashlib, shutil, zipfile, gzip
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'/'aqs_audit_2022'
OUT.mkdir(parents=True,exist_ok=True)
SRC=ROOT/'downloads'/'aqs_bexar_2022_original.json.gz'
PANEL=ROOT/'downloads'/'AQS_ERA5_2022_hourly_panel_UTC.csv'
with gzip.open(SRC, 'rt') as f: raw=json.load(f)
assert raw['Header'][0]['status']=='Success'
records=raw['Data']
assert len(records)==raw['Header'][0]['rows']
with gzip.open(str(PANEL)+'.gz', 'rt') as f: panel=list(csv.DictReader(f))
params={'42101':'CO','42602':'NO2','44201':'O3','88101':'PM2.5','42401':'SO2'}
null_codes={'AF','AI','AL','AM','AN','AZ','BA','BC','BF'}
groups=defaultdict(list)
allgroups=defaultdict(list)
monitors={}
for r in records:
 t=r['date_local']+' '+r['time_local']+':00'
 k=(int(r['site_number']),t,r['parameter_code'])
 allgroups[k[:2]].append(r)
 if r['sample_duration']=='1 HOUR':groups[k].append(r)
 m=(r['site_number'],r['parameter_code'],r['poc'],r['sample_duration'])
 monitors[m]={k:r[k] for k in ['site_number','parameter_code','poc','sample_duration','method_code','method','units_of_measure','latitude','longitude','datum']}
 local=datetime.fromisoformat(t)
 gmt=datetime.fromisoformat(r['date_gmt']+'T'+r['time_gmt'])
 assert gmt-local==timedelta(hours=6)

def writecsv(name, rows):
 rows=list(rows)
 if not rows:return
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

writecsv('aqs_original_records_2022.csv',records)
writecsv('aqs_monitor_metadata_2022.csv',monitors.values())
writecsv('aqs_daily_records_not_used.csv',(r for r in records if r['sample_duration']!='1 HOUR'))
audit524=[];long=[];disagreements=[];counts=Counter();reasons=Counter();keyset=set()
for p in panel:
 site=int(p['site']);local=p['datetime_local_standard'][:19]
 assert (site,p['datetime_utc']) not in keyset
 keyset.add((site,p['datetime_utc']))
 if p['aqs_source_row_present']=='0':
  rr=allgroups[(site,local)]
  assert rr and all(r['sample_measurement'] is None for r in rr)
  reasons.update(r['qualifier'] for r in rr)
  audit524.append({'site':f'{site:04d}','datetime_local_standard':local,'datetime_utc':p['datetime_utc'],'epa_records':len(rr),'numeric_records':0,'qualifiers':' | '.join(sorted(set(r['qualifier'] for r in rr))),'finding':'EPA records exist; all measurements null','export_cause':'Consistent with dropping all-null station-hours; export code not available'})
 for code,pol in params.items():
  rr=groups.get((site,local,code),[])
  assert len(rr)<=1, 'Ambiguous hourly POC requires explicit selection'
  r=rr[0] if rr else {}
  v=r.get('sample_measurement');q=r.get('qualifier') or '';qc=q.split(' - ')[0]
  numeric=v is not None
  if numeric:assert math.isfinite(v)
  status='observed_qualified' if numeric and q else 'observed_valid' if numeric else 'missing_null_qualified' if rr else 'no_hourly_record_in_requested_parameter'
  counts[status]+=1
  old=p[pol]
  matches=(old=='' and v is None) or (old!='' and v is not None and math.isclose(float(old),v,rel_tol=0,abs_tol=1e-9))
  row={'site':f'{site:04d}','datetime_utc':p['datetime_utc'],'datetime_local_standard':local,'parameter_code':code,'pollutant':pol,'original_panel_measurement':old,'sample_measurement':v,'qualifier':q,'method_code':r.get('method_code',''),'poc':r.get('poc',''),'sample_duration':r.get('sample_duration',''),'units_of_measure':r.get('units_of_measure',''),'latitude':r.get('latitude',''),'longitude':r.get('longitude',''),'aqs_record_present':int(bool(rr)),'status':status,'observed_valid':int(numeric and not q),'observed_qualified':int(numeric and bool(q)),'missing_sensor_outage':int(not numeric and qc=='AN'),'structural_non_monitoring':0 if rr else '', 'structural_non_monitoring_candidate':int(not rr),'invalid_measurement':int(not numeric and qc in {'AM','AL','AI'}),'recoverable_for_imputation':0 if numeric or not rr else '', 'imputation_review_required':int(bool(rr) and not numeric),'original_panel_value_matches':int(matches),'dataset_release_status':'PROVISIONAL_NO_IMPUTATION'}
  long.append(row)
  if not matches:disagreements.append(row)
assert len(audit524)==524
assert len(long)==219000
writecsv('aqs_524_station_hours_audit.csv',audit524)
writecsv('aqs_2022_hourly_status_PROVISIONAL.csv',long)
writecsv('panel_value_disagreements.csv',disagreements)

# ERA5 calculations are checked against the supplied CSV, not original product metadata.
weather=list(csv.DictReader(next((ROOT/'downloads').glob('ERA5_hourly_formatted*.csv')).open()))
formulas={'temp_c':lambda r:float(r['t2m'])-273.15,'dewpoint_c':lambda r:float(r['d2m'])-273.15,'surface_pressure_hpa':lambda r:float(r['sp'])/100,'precip_mm':lambda r:float(r['tp'])*1000,'wind_speed':lambda r:math.hypot(float(r['u10']),float(r['v10']))}
errors={k:max(abs(float(r[k])-fn(r)) for r in weather) for k,fn in formulas.items()}
weathercols=[k for k in panel[0] if k.startswith('era5_') and k not in ['era5_datetime_local','era5_utc_offset_hours','era5_is_dst','era5_local_clock_occurrence','era5_weather_matched']]
bytime=defaultdict(set)
for p in panel:bytime[p['datetime_utc']].add(tuple(p[k] for k in weathercols))
summary={'release_status':'PROVISIONAL; publication freeze and experiments blocked','source_request_time':raw['Header'][0]['request_time'],'original_aqs_records':len(records),'hourly_aqs_records':sum(r['sample_duration']=='1 HOUR' for r in records),'daily_records_excluded':sum(r['sample_duration']!='1 HOUR' for r in records),'panel_station_hours':len(panel),'disputed_station_hours':len(audit524),'all_disputed_hours_have_only_null_measurements':True,'disputed_source_record_qualifiers':dict(reasons),'status_counts':dict(counts),'panel_value_disagreements':len(disagreements),'era5_derived_column_max_absolute_error':errors,'era5_hours_identical_across_all_stations':sum(len(v)==1 for v in bytime.values()),'era5_unique_hours':len(bytime),'imputation_performed':False}
(OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2))
report=f'''# 2022 AQS–ERA5 data-quality audit

## Decision

This is a reproducible PROVISIONAL audit snapshot, not a publication-ready dataset. No imputation or model fitting was performed. Original input files were not changed.

## AQS findings

- {len(records):,} original EPA records retained with all returned metadata.
- {summary['hourly_aqs_records']:,} one-hour records; {summary['daily_records_excluded']} daily PM2.5 records kept separately, never averaged into hourly observations.
- All 524 disputed station-hours contain EPA records but no numeric measurement. Qualifier reasons are recorded individually in a 524-row audit. These are not missing valid observations or UTC errors.
- Exact export logic is not available. Dropping all-null station-hours is consistent with the evidence but not proven as the original implementation.
- {len(disagreements):,} pollutant-cell differences between the prior panel and the current hourly EPA records. If nonzero, inspect panel_value_disagreements.csv; no silent replacement was performed.
- Every original AQS local-to-GMT timestamp differs by exactly six hours, tested including minutes and date rollover.

## Status-column definitions

The long table has 219,000 station-hour-pollutant rows. Flags are pollutant-specific, not one status applied to an entire station-hour. Empty flag values mean unresolved, not false.

- observed_valid: finite reported number with no qualifier. This is a screening label, not independent certification of scientific validity. Negative reported numbers are retained.
- observed_qualified: reported number with a qualifier. A qualifier alone is not a reason to delete it.
- missing_sensor_outage: null measurement explicitly marked AN (machine malfunction). Maintenance/calibration are not conflated with machine failure.
- invalid_measurement: null record marked AM, AL or AI (void or insufficient data). No missing numeric values are invented.
- structural_non_monitoring: unresolved where no hourly record was returned; candidate flag marks those combinations. Absence in these five parameter codes does not prove the pollutant was never monitored under another code (e.g. PM2.5 88502). Monitor-history/parameter inventory is required to confirm it.
- recoverable_for_imputation: false for observed values or unmonitored candidates; unresolved for monitored nulls pending gap-policy validation. No gap is declared scientifically recoverable just because its neighbors exist.

## ERA5 provenance

The provided download notebooks request reanalysis-era5-single-levels, hourly, seven variables, area [29.60, -98.65, 29.30, -98.30] in north/west/south/east order. This documents requested coverage, not the actual grid selection or reduction that generated this CSV. AQS sites 0052 and 0059 lie outside that requested latitude interval; station-specific extraction needs review.

All {summary['era5_unique_hours']:,} panel UTC hours have identical weather values across the five stations. This proves a shared weather series, not whether it originated from a single grid cell or an area average.

The supplied formatted ERA5 CSV has no grid coordinates or original timezone-aware UTC. Its UTC was reconstructed under an America/Chicago interpretation. AQS GMT verification does NOT validate ERA5 chronology, spatial extraction or precipitation accumulation semantics. Earlier wording that ERA5 original UTC was retained was incorrect.

Two available NetCDF files were located, but their time coverage and relevance could not be inspected with the installed readers. They have not been used as provenance proof.

Formula residuals across supplied ERA5 CSV (maximum absolute difference):
{json.dumps(errors,indent=2)}

Checked arithmetic: kelvin minus 273.15; pascals divided by 100; metres times 1000; wind speed sqrt(u10^2+v10^2). Small residuals may reflect stored precision. Actual source units, humidity formula, temporal accumulation and coordinate selection remain to be verified from source metadata and processing code.

## Required before final freeze and experiments

Supply the original 2022 ERA5 NetCDF/GRIB and the script/notebook that generated the formatted CSV. Include 2023-01-01 00:00–05:00 UTC because this panel represents the 2022 CST year (2022-01-01 06:00 UTC through 2023-01-01 05:00 UTC), not the UTC calendar year. Resolve any pollutant-cell differences and confirm monitor inventory before freezing.

After validation, fit transformations only on chronological training data. Evaluate imputation on withheld observed values using train-derived outage masks. Keep offline interpolation/smoothing separate from causal forecasting. Do not let post-forecast observations fill forecasting inputs. Proposed January–September/October/November–December split is a starting protocol, not a completed experiment. Seasonal generalization requires further evaluation. DeepMVI and baseline results are not yet generated.

## Sources

- EPA response request time: {raw['Header'][0]['request_time']}. Full original response is included in the archive.
- EPA qualifier definitions: https://aqs.epa.gov/aqsweb/documents/codetables/qualifiers.html
- ERA5 documentation: https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation
- Supplied panel and ERA5 formatted CSV; download_era5_bexar_hourly_2019_2024 notebooks. Source hashes are included for reproducibility.
'''
(OUT/'READ_ME_data_quality_report.md').write_text(report)
shutil.copy2(Path(__file__),OUT/'audit_2022.py')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SRC,PANEL,*OUT.glob('*')] if p.is_file()}
(OUT/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(OUT.parent/'AQS_2022_audit_PROVISIONAL.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.iterdir():z.write(p,p.name)
 z.write(SRC,'downloads/aqs_bexar_2022_original.json')
 z.write(PANEL,'downloads/AQS_ERA5_2022_hourly_panel_UTC.csv')
 f=next((ROOT/'downloads').glob('ERA5_hourly_formatted*.csv'));z.write(f,'downloads/'+f.name)
print(json.dumps(summary,indent=2))
