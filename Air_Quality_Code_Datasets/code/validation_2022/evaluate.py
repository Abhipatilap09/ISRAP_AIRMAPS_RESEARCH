import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch,joblib
from torch import nn
ROOT=Path(__file__).resolve().parent
PKG=ROOT.parent/'source/pkg';sys.path.insert(0,str(PKG))
from pm25_target_experiment import load_panel,make_block_mask,timestamp_features,tabular_features,interpolate,SITES,WEATHER
from timexer_imputation import TimeXerImputer
torch.set_num_threads(2)
class StationTemporalTransformer(nn.Module):
 def __init__(self):
  super().__init__();self.project=nn.Linear(15,64);self.pos=nn.Parameter(torch.zeros(1,24,64));self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,128,.1,batch_first=True,norm_first=True),2);self.out=nn.Linear(64,3)
 def forward(self,x):return self.out(self.encoder(self.project(x)+self.pos))

def net_predictions(model,c,pm,w,times,mask):
 z=(pm-c['pm_mean'])/c['pm_std'];wz=(w-c['weather_mean'])/c['weather_std'];cyc=timestamp_features(times)[:,:2];obs=np.isfinite(z)&~mask;z[~obs]=0
 x=np.concatenate([z,obs.astype(float),wz,cyc],axis=1).reshape(-1,24,15)
 assert (x.reshape(-1,15)[mask.any(1),:3][mask[mask.any(1)]]==0).all()
 with torch.no_grad():out=torch.cat([model(a) for a in torch.tensor(x,dtype=torch.float32).split(32)]).numpy().reshape(-1,3)
 return out*c['pm_std']+c['pm_mean']

c=torch.load(PKG/'transformer_checkpoint.pt',weights_only=False);m=StationTemporalTransformer();m.load_state_dict(c['state_dict']);m.eval()
tc=torch.load(PKG/'timexer_results/patch_6/timexer_checkpoint.pt',weights_only=False);tm=TimeXerImputer(6);tm.load_state_dict(tc['state_dict']);tm.eval()
oldtimes,oldpm,oldw,_=load_panel()
for key,expected in [('pm_mean',np.nanmean(oldpm[:6552],0)),('pm_std',np.nanstd(oldpm[:6552],0)),('weather_mean',oldw[:6552].mean(0)),('weather_std',oldw[:6552].std(0))]:
 assert np.allclose(c[key],expected) and np.allclose(tc[key],expected)
# Independently replay the October checkpoint predictions before transfer.
replay=[]
for model,ck,method,path in [(m,c,'PM2.5 station-temporal Transformer',PKG/'october_masked_predictions.csv'),(tm,tc,'TimeXer imputation adaptation',PKG/'timexer_results/patch_6/october_masked_predictions_timexer.csv')]:
 saved=pd.read_csv(path);saved=saved[saved.method==method];errors=[]
 for seed in [11,22,33]:
  mask,rows=make_block_mask(oldpm,6552,7296,seed);pred=net_predictions(model,ck,oldpm[6552:7296],oldw[6552:7296],oldtimes[6552:7296],mask[6552:7296])
  lookup={(oldtimes[t].isoformat(),SITES[j]):pred[t-6552,j] for t,j,g in rows}
  for r in saved[saved.seed==seed].itertuples():errors.append(abs(lookup[(r.datetime_utc,r.site)]-r.prediction))
 replay.append({'method':method,'n':len(errors),'max_prediction_difference':max(errors)});assert max(errors)<1e-4
(ROOT/'checkpoint_replay.json').write_text(json.dumps(replay,indent=2));print('October checkpoint replay passed',replay,flush=True)
times=oldtimes[7296:];pm=oldpm[7296:].copy();w=oldw[7296:].copy();assert len(times)==1464;assert times[0]==pd.Timestamp('2022-11-01 06:00:00Z')
rf=joblib.load(PKG/'random_forest.joblib.gz');cal=timestamp_features(times);hour=(times-pd.Timedelta(hours=6)).hour.to_numpy();oh=(oldtimes-pd.Timedelta(hours=6)).hour.to_numpy();med={(j,h):np.nanmedian(oldpm[:6552,j][oh[:6552]==h]) for j in range(3) for h in range(24)}
results=[]
for seed in [11,22,33]:
 mask,rows=make_block_mask(pm,0,len(pm),seed);masked=pm.copy();masked[mask]=np.nan
 predictions={'Hour median':np.array([med[j,hour[t]] for t,j,g in rows]),'Linear interpolation':interpolate(masked,rows),'Random Forest':rf.predict(tabular_features(masked,w,cal,rows))}
 for model,ck,method in [(m,c,'PM2.5 station-temporal Transformer'),(tm,tc,'TimeXer imputation adaptation')]:
  p=net_predictions(model,ck,pm,w,times,mask);predictions[method]=np.array([p[t,j] for t,j,g in rows])
 for method,p in predictions.items():
  assert np.isfinite(p).all()
  for (t,j,g),v in zip(rows,p):results.append(dict(seed=seed,method=method,datetime_utc=times[t].isoformat(),site=SITES[j],gap_hours=g,truth=pm[t,j],prediction=float(v)))
 print('Evaluated seed',seed,'hidden',len(rows),flush=True)
d=pd.DataFrame(results);d.to_csv(ROOT/'predictions_2022_nov_dec.csv',index=False);d['ae']=abs(d.truth-d.prediction);d['se']=(d.truth-d.prediction)**2
keys=['seed','datetime_utc','site','gap_hours'];sets=[set(x[keys].itertuples(index=False,name=None)) for _,x in d.groupby('method')];assert all(s==sets[0] for s in sets)
for group,name in [(['method'],'comparison_2022_nov_dec'),(['method','site'],'by_station'),(['method','gap_hours'],'by_gap'),(['method','seed'],'by_seed')]:
 a=d.groupby(group).agg(n=('truth','size'),MAE=('ae','mean'),MSE=('se','mean'));a['RMSE']=np.sqrt(a.MSE);a.drop(columns='MSE').to_csv(ROOT/(name+'.csv'))
d['day']=d.datetime_utc.str[:10];days=sorted(d.day.unique());rng=np.random.default_rng(404);ids=rng.integers(0,len(days),size=(2000,len(days)));boot={}
for method,x in d.groupby('method'):
 a=x.groupby('day').agg(n=('ae','size'),s=('ae','sum')).reindex(days,fill_value=0);boot[method]=a.s.to_numpy()[ids].sum(1)/a.n.to_numpy()[ids].sum(1)
ci=[{'method':k,'MAE_CI_lower':np.quantile(v,.025),'MAE_CI_upper':np.quantile(v,.975)} for k,v in boot.items()];pd.DataFrame(ci).to_csv(ROOT/'day_block_bootstrap.csv',index=False)
paired=[]
for k,v in boot.items():
 if k!='PM2.5 station-temporal Transformer':
  diff=boot['PM2.5 station-temporal Transformer']-v;paired.append({'comparison':'station Transformer minus '+k,'lower':np.quantile(diff,.025),'upper':np.quantile(diff,.975)})
pd.DataFrame(paired).to_csv(ROOT/'paired_MAE_differences.csv',index=False)
print(pd.read_csv(ROOT/'comparison_2022_nov_dec.csv').to_string(index=False),flush=True)
