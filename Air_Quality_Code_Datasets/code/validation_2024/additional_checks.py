import sys,json
from pathlib import Path
import numpy as np,pandas as pd,joblib
root=Path(__file__).resolve().parent;pkg=root.parent/'source/pkg';sys.path.insert(0,str(pkg))
from pm25_target_experiment import load_panel,make_block_mask,timestamp_features,tabular_features,interpolate,hour_median,SITES
T,P,W,_=load_panel();saved=pd.read_csv(pkg/'october_masked_predictions.csv');rf=joblib.load(pkg/'random_forest.joblib.gz');checks=[]
for seed in [11,22,33]:
 mask,rows=make_block_mask(P,6552,7296,seed);masked=P.copy();masked[mask]=np.nan;ctx=masked.copy();ctx[:6552]=np.nan;ctx[7296:]=np.nan
 preds={'Random Forest':rf.predict(tabular_features(ctx,W,timestamp_features(T),rows)),'Hour median':hour_median(P,T,rows),'Linear interpolation':interpolate(masked,rows)}
 for method,p in preds.items():
  lookup={(T[t].isoformat(),SITES[j]):v for (t,j,g),v in zip(rows,p)};s=saved[(saved.method==method)&(saved.seed==seed)];e=[abs(lookup[(x.datetime_utc,x.site)]-x.prediction) for x in s.itertuples()];assert max(e)<1e-8;checks.append(dict(method=method,seed=seed,n=len(e),max_prediction_difference=max(e)))
(root/'baseline_replay.json').write_text(json.dumps(checks,indent=2))
d=pd.read_csv(root/'predictions_2024.csv');d['day']=d.datetime_utc.str[:10];d['ae']=abs(d.truth-d.prediction);days=sorted(d.day.unique());rng=np.random.default_rng(407);starts=rng.integers(0,len(days),(2000,int(np.ceil(len(days)/7))));ids=((starts[:,:,None]+np.arange(7))%len(days)).reshape(2000,-1)[:,:len(days)];boot={}
for method,x in d.groupby('method'):
 a=x.groupby('day').agg(n=('ae','size'),s=('ae','sum')).reindex(days,fill_value=0);boot[method]=a.s.to_numpy()[ids].sum(1)/a.n.to_numpy()[ids].sum(1)
pd.DataFrame([{'method':k,'MAE_CI_lower':np.quantile(v,.025),'MAE_CI_upper':np.quantile(v,.975)} for k,v in boot.items()]).to_csv(root/'week_block_bootstrap.csv',index=False)
pd.DataFrame([{'comparison':'station Transformer minus '+k,'lower':np.quantile(boot['PM2.5 station-temporal Transformer']-v,.025),'upper':np.quantile(boot['PM2.5 station-temporal Transformer']-v,.975)} for k,v in boot.items() if k!='PM2.5 station-temporal Transformer']).to_csv(root/'week_paired_MAE_differences.csv',index=False)
# Examples chosen chronologically, one for each gap length; no favorable-error selection.
keys=['seed','datetime_utc','site','gap_hours'];examples=[]
for gap in [1,3,6]:
 s=d[(d.seed==11)&(d.site==59)&(d.gap_hours==gap)];key=s.sort_values('datetime_utc').iloc[0];x=s[s.datetime_utc==key.datetime_utc];examples.append({'datetime_utc':key.datetime_utc,'site':59,'gap_hours':gap,'truth':key.truth,**dict(zip(x.method,x.prediction))})
pd.DataFrame(examples).to_csv(root/'examples_2024.csv',index=False)
print('All five October methods independently replayed; weekly block uncertainty calculated.');print(pd.read_csv(root/'week_paired_MAE_differences.csv').to_string(index=False));print(pd.DataFrame(examples).to_string(index=False))
a=pd.read_csv(root/'raw/aqs_2024_bexar_original.csv');a['datetime_utc']=pd.to_datetime(a['Date GMT']+' '+a['Time GMT'],utc=True).map(lambda x:x.isoformat());a=a.rename(columns={'Site Num':'site'});q=d.merge(a[['datetime_utc','site','MDL']],on=['datetime_utc','site'],how='left',validate='many_to_one');assert q.MDL.notna().all();out=[]
for mode,sub in [('all screened targets',q),('exclude below negative MDL targets',q[q.truth>= -q.MDL]),('exclude all negative targets',q[q.truth>=0])]:
 for method,x in sub.groupby('method'):out.append(dict(sensitivity=mode,method=method,n=len(x),MAE=abs(x.truth-x.prediction).mean(),RMSE=((x.truth-x.prediction)**2).mean()**.5))
pd.DataFrame(out).to_csv(root/'target_quality_sensitivity.csv',index=False)
