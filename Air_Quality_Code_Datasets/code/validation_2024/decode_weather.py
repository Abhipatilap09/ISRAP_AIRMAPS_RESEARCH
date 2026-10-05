import sys,zipfile,tempfile,json
from pathlib import Path
import pandas as pd,numpy as np
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'validation_sources/ERA5_source_validation_2022'))
from read_era5_nc import read
rows=[];meta={}
for f in sorted((root.parent/'validation_sources/NEW_NASA_RESEARCH').glob('*.nc')):
 with tempfile.TemporaryDirectory() as td:
  parts={};times=None
  with zipfile.ZipFile(f) as z:
   for name in z.namelist():
    p=Path(td)/Path(name).name;p.write_bytes(z.read(name));d=read(p)
    assert d['valid_time']['attributes']['units']=='seconds since 1970-01-01'
    t=pd.to_datetime(d['valid_time']['values'].astype('int64'),unit='s',utc=True)
    if times is not None:assert times.equals(t)
    times=t
    assert float(d['latitude']['values'][0])==29.5 and float(d['longitude']['values'][0])==-98.5
    for v in ['u10','v10','t2m','d2m','sp','tp','blh']:
     if v in d:parts['era5_'+v]=d[v]['values'].reshape(-1);meta[v]=d[v]['attributes']
  rows.append(pd.DataFrame(parts,index=times))
a=pd.concat(rows).sort_index();assert len(a)==8784 and a.index.is_unique and np.isfinite(a).all().all();assert (a.index[1:]-a.index[:-1]==pd.Timedelta(hours=1)).all();a.index.name='datetime_utc';a.to_csv(root/'weather_original_utc.csv');(root/'weather_metadata.json').write_text(json.dumps(meta,indent=2));print('Original source 2024 weather decoded and validated')
