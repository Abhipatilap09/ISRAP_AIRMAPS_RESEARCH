from pathlib import Path
import pandas as pd,numpy as np,zipfile,json,hashlib
root=Path(__file__).resolve().parent
pieces=[]
with zipfile.ZipFile(root/'raw/hourly_88101_2024.zip') as z:
 with z.open(z.namelist()[0]) as f:
  for d in pd.read_csv(f,chunksize=250000,low_memory=False):
   s=d[(d['State Code']==48)&(d['County Code']==29)&d['Site Num'].isin([32,59,1069])&(d.POC==2)].copy()
   if len(s):pieces.append(s)
a=pd.concat(pieces,ignore_index=True);a.to_csv(root/'raw/aqs_2024_bexar_original.csv',index=False)
a['datetime_utc']=pd.to_datetime(a['Date GMT']+' '+a['Time GMT'],utc=True)
assert ((a.datetime_utc-pd.to_datetime(a['Date Local']+' '+a['Time Local'],utc=True))==pd.Timedelta(hours=6)).all()
assert a['Parameter Code'].eq(88101).all() and a['Units of Measure'].eq('Micrograms/cubic meter (LC)').all()
assert not a.duplicated(['datetime_utc','Site Num']).any()
w=pd.read_csv(root/'weather_original_utc.csv');w.datetime_utc=pd.to_datetime(w.datetime_utc,utc=True);w=w.set_index('datetime_utc');rows=[]
for site in [32,59,1069]:
 s=a[a['Site Num']==site].set_index('datetime_utc').reindex(w.index);v=pd.to_numeric(s['Sample Measurement'],errors='coerce');q=s.Qualifier.fillna('').astype(str).str.strip();valid=np.isfinite(v)&q.eq('');w[f'PM25_{site}']=np.where(valid,v,np.nan)
 rows.append({'site':site,'original_records':int((a['Site Num']==site).sum()),'observed_valid':int(valid.sum()),'qualified_numeric':int((np.isfinite(v)&~q.eq('')).sum()),'negative_unqualified':int((valid&(v<0)).sum()),'below_negative_MDL':int((valid&(v < -pd.to_numeric(s.MDL,errors='coerce'))).sum()),'minimum':float(v[valid].min()),'method_codes':s['Method Code'].dropna().unique().tolist()})
w.to_csv(root/'panel_2024.csv');audit={'source_url':'https://aqs.epa.gov/aqsweb/airdata/hourly_88101_2024.zip','sha256':hashlib.sha256((root/'raw/hourly_88101_2024.zip').read_bytes()).hexdigest(),'hours':len(w),'hourly_archive':True,'GMT_minus_local_hours':6,'duplicate_station_hours':0,'screening':'finite numeric value with empty Qualifier, identical rule to 2022; negative measurements retained','stations':rows};(root/'data_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2),flush=True)
# Compare pilot ERA5 against original 2022 source using stated rounding tolerances.
pilot=pd.read_csv(root.parent/'source/pkg/research_doc_sources/EPA_AQS_ERA5_2022_QUALITY_LABELLED.csv.gz',low_memory=False);pilot=pilot[pilot.site==32].copy();pilot.datetime_utc=pd.to_datetime(pilot.datetime_utc,utc=True);pilot=pilot.set_index('datetime_utc')
orig=pd.read_csv(root.parent/'validation_sources/ERA5_source_validation_2022/ERA5_2022_CST_year_original_UTC.csv');orig.datetime_utc=pd.to_datetime(orig.datetime_utc,utc=True);orig=orig.set_index('datetime_utc').reindex(pilot.index)
tol={'u10':1e-5,'v10':1e-5,'t2m':1e-4,'d2m':1e-4,'sp':.01,'blh':.001,'tp':1e-9};diff=[]
for v,t in tol.items():
 e=abs(pilot['era5_'+v]-orig[v]);bad=e>t
 for date in e[bad].index:diff.append({'datetime_utc':date.isoformat(),'variable':v,'absolute_difference':e[date]})
pd.DataFrame(diff).to_csv(root/'pilot_weather_source_differences.csv',index=False);assert all(pd.Timestamp(x['datetime_utc'])>=pd.Timestamp('2022-11-01',tz='UTC') for x in diff)
print('2022 original source differences outside training and October:',len(diff),flush=True)
