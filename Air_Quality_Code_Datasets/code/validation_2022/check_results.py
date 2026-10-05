from pathlib import Path
import sys,json
import pandas as pd,numpy as np
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root.parent/'source/pkg'))
from pm25_target_experiment import load_panel,make_block_mask,SITES
T,P,W,_=load_panel();T=T[7296:];P=P[7296:];d=pd.read_csv(root/'predictions_2022_nov_dec.csv');keys=['seed','datetime_utc','site','gap_hours'];reference=set()
for seed in [11,22,33]:
 mask,rows=make_block_mask(P,0,len(P),seed)
 reference.update((seed,T[t].isoformat(),SITES[j],g,float(P[t,j])) for t,j,g in rows)
for method,s in d.groupby('method'):
 assert len(s)==1865 and not s.duplicated(keys).any();assert set(s[keys+['truth']].itertuples(index=False,name=None))==reference;assert np.isfinite(s.prediction).all()
summary=pd.read_csv(root/'comparison_2022_nov_dec.csv').set_index('method')
for method,s in d.groupby('method'):
 assert np.isclose(abs(s.truth-s.prediction).mean(),summary.loc[method,'MAE']);assert np.isclose(np.sqrt(((s.truth-s.prediction)**2).mean()),summary.loc[method,'RMSE'])
d['day']=d.datetime_utc.str[:10];d['ae']=abs(d.truth-d.prediction);days=sorted(d.day.unique());rng=np.random.default_rng(407);starts=rng.integers(0,len(days),(2000,int(np.ceil(len(days)/7))));ids=((starts[:,:,None]+np.arange(7))%len(days)).reshape(2000,-1)[:,:len(days)];boot={}
for method,s in d.groupby('method'):
 a=s.groupby('day').agg(n=('ae','size'),total=('ae','sum')).reindex(days,fill_value=0);boot[method]=a.total.to_numpy()[ids].sum(1)/a.n.to_numpy()[ids].sum(1)
pd.DataFrame([dict(method=k,MAE_CI_lower=np.quantile(v,.025),MAE_CI_upper=np.quantile(v,.975)) for k,v in boot.items()]).to_csv(root/'week_block_bootstrap.csv',index=False)
pd.DataFrame([dict(comparison='station Transformer minus '+k,lower=np.quantile(boot['PM2.5 station-temporal Transformer']-v,.025),upper=np.quantile(boot['PM2.5 station-temporal Transformer']-v,.975)) for k,v in boot.items() if k!='PM2.5 station-temporal Transformer']).to_csv(root/'week_paired_MAE_differences.csv',index=False)
d['month']=d.datetime_utc.str[:7];d['se']=(d.truth-d.prediction)**2;a=d.groupby(['method','month']).agg(n=('truth','size'),MAE=('ae','mean'),MSE=('se','mean'));a['RMSE']=np.sqrt(a.MSE);a.drop(columns='MSE').to_csv(root/'by_utc_month.csv')
examples=[]
for g in [1,3,6]:
 a=d[(d.seed==11)&(d.site==59)&(d.gap_hours==g)];r=a.sort_values('datetime_utc').iloc[0];a=a[a.datetime_utc==r.datetime_utc];examples.append(dict(datetime_utc=r.datetime_utc,site=59,gap_hours=g,truth=r.truth,**dict(zip(a.method,a.prediction))))
pd.DataFrame(examples).to_csv(root/'examples.csv',index=False)
checks=dict(equal_keys_all_five_methods=True,masked_instances_per_method=1865,unique_station_timestamp_pairs=len(d[['datetime_utc','site']].drop_duplicates()),metrics_recomputed=True,mask_seeds=[11,22,33],scalers_checked_by_evaluate=True,October_neural_replay=True,training_period='Jan-Sep',selection_period='Oct',evaluation_period='Nov-Dec');(root/'checks.json').write_text(json.dumps(checks,indent=2))
print(pd.read_csv(root/'week_paired_MAE_differences.csv').to_string(index=False));print('All mask and metric checks passed.')
