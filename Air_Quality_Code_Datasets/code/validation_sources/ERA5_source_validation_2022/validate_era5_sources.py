import csv,json,zipfile,tempfile,calendar,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from read_era5_nc import read
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'/'era5_source_validation';OUT.mkdir(parents=True,exist_ok=True)
variables=['u10','v10','t2m','d2m','sp','blh','tp']
inventory=[];source={};metadata={}
with tempfile.TemporaryDirectory() as td:
 for f in sorted((ROOT/'upload').glob('era5_bexar_*.nc')):
  year,month=map(int,f.stem.split('_')[-2:]);parts={}; times=None;coords=[]
  with zipfile.ZipFile(f) as z:
   for name in z.namelist():
    p=Path(td)/Path(name).name;p.write_bytes(z.read(name));d=read(p)
    t=d['valid_time']['values'].astype('int64')
    assert d['valid_time']['attributes']['units']=='seconds since 1970-01-01'
    assert len(t)==24*calendar.monthrange(year,month)[1] and np.all(np.diff(t)==3600)
    assert datetime.fromtimestamp(int(t[0]),timezone.utc)==datetime(year,month,1,tzinfo=timezone.utc)
    if times is not None:assert np.array_equal(t,times)
    times=t
    lat=d['latitude']['values'];lon=d['longitude']['values'];assert lat.size==lon.size==1
    coords.append((float(lat[0]),float(lon[0])))
    for v in variables:
     if v in d:
      assert d[v]['dimensions']==['valid_time','latitude','longitude']
      parts[v]=d[v]['values'].reshape(-1);assert np.isfinite(parts[v]).all()
      metadata[v]={k:d[v]['attributes'].get(k) for k in ['units','GRIB_stepType','GRIB_dataType','GRIB_gridType','GRIB_iDirectionIncrementInDegrees','GRIB_jDirectionIncrementInDegrees']}
  assert set(parts)==set(variables) and len(set(coords))==1
  inventory.append({'file':f.name,'year':year,'month':month,'hours':len(times),'latitude':coords[0][0],'longitude':coords[0][1],'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
  if year==2022 or (year==2023 and month==1):
   for i,t in enumerate(times):source[int(t)]={v:parts[v][i] for v in variables}|{'file':f.name,'latitude':coords[0][0],'longitude':coords[0][1]}
def write(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write('uploaded_file_inventory.csv',inventory)
panel=list(csv.DictReader((ROOT/'downloads'/'AQS_ERA5_2022_hourly_panel_UTC.csv').open()))
unique={r['datetime_utc']:r for r in panel};assert len(unique)==8760
tol={'u10':1e-5,'v10':1e-5,'t2m':1e-4,'d2m':1e-4,'sp':.01,'blh':.001,'tp':1e-9}
comparison=[];canonical=[]
for iso,p in unique.items():
 t=int(datetime.fromisoformat(iso.replace('Z','+00:00')).timestamp());r=source[t]
 canonical.append({'datetime_utc':iso,'latitude':r['latitude'],'longitude':r['longitude'],'source_file':r['file'],**{v:float(r[v]) for v in variables}})
 for v in variables:
  old=float(p['era5_'+v]);new=float(r[v]);e=abs(old-new)
  comparison.append({'datetime_utc':iso,'variable':v,'old_panel_value':old,'original_era5_value':new,'absolute_difference':e,'tolerance':tol[v],'within_tolerance':int(e<=tol[v])})
write('ERA5_2022_CST_year_original_UTC.csv',canonical)
write('ERA5_panel_source_comparison.csv',comparison)
checks=[{'variable':v,'hours':8760,'within_tolerance':sum(r['within_tolerance'] for r in comparison if r['variable']==v),'maximum_absolute_difference':max(r['absolute_difference'] for r in comparison if r['variable']==v),'tolerance':tol[v]} for v in variables]
write('ERA5_comparison_summary.csv',checks)
missing=[f'{y}-{m:02d}' for y in range(2019,2025) for m in range(1,13) if not any(r['year']==y and r['month']==m for r in inventory)]
summary={'uploaded_files':len(inventory),'missing_months':missing,'grid_coordinates':sorted(set((r['latitude'],r['longitude']) for r in inventory)),'2022_station_hours':len(panel),'2022_unique_hours':len(unique),'comparisons':checks,'metadata':metadata}
(OUT/'results.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
