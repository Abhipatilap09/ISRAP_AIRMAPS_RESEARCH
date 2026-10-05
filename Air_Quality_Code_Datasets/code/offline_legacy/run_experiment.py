"""Offline imputation experiment. Author architectures with explicit inductive adapters.
Run from workspace: PYTHONPATH=deps python3 outputs/imputation_2022/run_experiment.py
"""
from pathlib import Path
import sys,time,json,hashlib,copy,importlib.util
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'methods/DeepMVI'))
from model import OurModel
# Import author MPIN modules without executing their benchmark script.
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
MP=module(ROOT/'methods/MPIN/utils/DynamicGNN.py','mpin_gnn')
REG=module(ROOT/'methods/MPIN/utils/regressor.py','mpin_regressor')
torch.set_num_threads(2)
SOURCE=ROOT.parent/'datasets/merged/aqs_era5_2022_quality_labelled.csv.gz'
source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
d=pd.read_csv(SOURCE);gaps=pd.read_csv(ROOT.parent/'datasets/merged/missing_gap_report.csv')
pols=['CO','NO2','O3','PM2.5','SO2']; times=sorted(d.datetime_utc.unique()); T=len(times)
groups={s:g.sort_values('datetime_utc') for s,g in d.groupby('site')}
channels=[(s,p) for s,g in groups.items() for p in pols if g[p].notna().any()]
weather=['era5_u10','era5_v10','era5_t2m','era5_d2m','era5_sp','era5_tp','era5_blh']
X=np.column_stack([groups[s][p].to_numpy() for s,p in channels]+[next(iter(groups.values()))[w].to_numpy() for w in weather])
P=len(channels); C=X.shape[1]; split=next(iter(groups.values())).evaluation_split.to_numpy(); nt=int((split=='train').sum()); nv=int((split=='validation').sum())
mu=np.nanmean(X[:nt],axis=0);sd=np.nanstd(X[:nt],axis=0);sd[sd<1e-10]=1
Z=(X-mu)/sd
hours=(pd.to_datetime(times,utc=True)-pd.Timedelta(hours=6)).hour.to_numpy()
lengths=[]
for s,p in channels:
 a=gaps[(gaps.site==s)&(gaps.pollutant==p)&(gaps.split=='train')&(gaps.hours<=8)].hours.to_numpy(dtype=int)
 lengths.append(a if len(a) else np.array([1]))
SEEDS=[11,22,33,44,55]; STEPS=400; WINDOW=168; BATCH=8
names=['hour_median','linear_offline','kalman_offline','DeepMVI_random_adapted','DeepMVI_outage_adapted','MPIN_frozen_adapted']
config={'seeds':SEEDS,'steps':STEPS,'window_hours':WINDOW,'batch':BATCH,'training_hours':nt,'validation_hours':nv,'channels':channels,'weather':weather,'source_sha256':source_hash,'primary_gap_hours':[1,8],'test_scoring':False,'scope':'offline retrospective reconstruction','selection':'lowest mean seed MAE by pollutant; within 1 percent prefer simpler method in listed order','methods':names}
(OUT/'protocol.json').write_text(json.dumps(config,indent=2))
np.savez(OUT/'training_scaling.npz',mean=mu,std=sd)

def smooth(y,median,variance,noise):
 n=len(y);mf=np.empty(n);pf=np.empty(n);m=median;p=variance
 for i,v in enumerate(y):
  p+=noise
  if np.isfinite(v):k=p/(p+noise);m+=k*(v-m);p*=1-k
  mf[i]=m;pf[i]=p
 a=mf.copy()
 for i in range(n-2,-1,-1):a[i]=mf[i]+pf[i]/(pf[i]+noise)*(a[i+1]-mf[i])
 return a

stats=[]
for j in range(P):
 y=X[:nt,j];diff=np.diff(y);diff=diff[np.isfinite(diff)];median=float(np.nanmedian(y));v=max(float(np.nanvar(y)),1e-12);noise=max(float(np.mean(diff**2))/2,v*1e-6,1e-12)
 hm=np.array([np.nanmedian(y[hours[:nt]==h]) for h in range(24)]);hm=np.where(np.isfinite(hm),hm,median);stats.append((median,v,noise,hm))

def baselines(raw,hh):
 out={n:np.zeros((len(raw),P)) for n in names[:3]}
 for j,(median,v,q,hm) in enumerate(stats):
  out['hour_median'][:,j]=hm[hh]
  out['linear_offline'][:,j]=pd.Series(raw[:,j]).interpolate(limit_area='inside').to_numpy()
  out['kalman_offline'][:,j]=smooth(raw[:,j],median,v,q)
 return out

def masking(raw,seed):
 rng=np.random.default_rng(seed);mask=np.zeros((len(raw),P),bool);lens=np.zeros_like(mask,dtype=int)
 for j in range(P):
  obs=np.isfinite(raw[:,j]);target=int(obs.sum()*.1)
  for attempt in range(20000):
   if mask[:,j].sum()>=target:break
   L=int(rng.choice(lengths[j]));a=int(rng.integers(1,len(raw)-L))
   if obs[a-1:a+L+1].all() and not mask[a-1:a+L+1,j].any():mask[a:a+L,j]=True;lens[a:a+L,j]=L
 return mask,lens

def make_deep():
 return OurModel([C],kernel_size=12,block_size=1,nhead=2,use_embed=True,use_context=True,use_local=True)

def deep_forward(model,arr,cols):
 # arr has already had ALL evaluation targets removed, including sibling context.
 arr=torch.nan_to_num(arr,nan=0.0);b,w,c=arr.shape;ii=torch.arange(b)
 own=arr[ii,:,cols];siblings=arr.clone();siblings[ii,:,cols]=0
 context=[cols[:,None],torch.zeros(b,dtype=torch.long),torch.arange(w).repeat(b,1),torch.zeros(b,w//12,w)]
 return model.core(own,siblings,context,test=True)

class MPIN(torch.nn.Module):
 def __init__(self):
  super().__init__();self.gnns=torch.nn.ModuleList([MP.StaticGraphSAGE(C,256,k=10),MP.StaticGraphSAGE(C,256,k=10)]);self.reg=REG.MLPNet(256,C)
 def forward(self,raw):
  obs=torch.isfinite(raw);x=torch.nan_to_num(raw,nan=0.0)
  # Exact Euclidean 10-NN, matching the original graph definition, no compiled extension.
  with torch.no_grad():
   dist=torch.cdist(x,x);dist.fill_diagonal_(float('inf'));nei=dist.topk(min(10,len(x)-1),largest=False).indices
   edge=torch.stack([nei.reshape(-1),torch.arange(len(x)).repeat_interleave(nei.shape[1])])
  out=[];z=x
  for gnn in self.gnns:
   emb,_=gnn(z,edge);pred=self.reg(emb);out.append(pred);z=torch.where(obs,x,pred)
  return out

logs=[];models={}
for seed in SEEDS:
 for method in names[3:]:
  key=f'{method}_{seed}';ckpt=OUT/(key+'.pt');torch.manual_seed(seed);rng=np.random.default_rng(seed);start=time.time()
  net=MPIN() if method.startswith('MPIN') else make_deep()
  if ckpt.exists():
   net.load_state_dict(torch.load(ckpt,weights_only=True));models[key]=net;logs.append({'method':method,'seed':seed,'steps':STEPS,'checkpoint_reused':True});continue
  opt=torch.optim.Adam(net.parameters(),lr=.001);net.train()
  for step in range(STEPS):
   if method.startswith('MPIN'):
    a=int(rng.integers(0,nt-WINDOW+1));arr=torch.tensor(Z[a:a+WINDOW],dtype=torch.float32);obs=torch.isfinite(arr)
    # Frozen inductive adaptation uses author two-stage observed reconstruction loss.
    outputs=net(arr);loss=sum(torch.abs(v[obs]-arr[obs]).mean() for v in outputs)
   else:
    starts=rng.integers(0,nt-WINDOW+1,size=BATCH);cols=rng.integers(0,P,size=BATCH)
    arr=np.stack([Z[a:a+WINDOW].copy() for a in starts]);truth=arr[np.arange(BATCH),:,cols].copy();hidden=np.zeros((BATCH,WINDOW),bool)
    for b,j in enumerate(cols):
     if 'random' in method:hidden[b]=(rng.random(WINDOW)<.1)&np.isfinite(truth[b])
     else:
      for k in range(50):
       if hidden[b].sum()>=WINDOW*.1:break
       L=int(rng.choice(lengths[j]));a=int(rng.integers(1,WINDOW-L))
       if np.isfinite(truth[b,a-1:a+L+1]).all():hidden[b,a:a+L]=True
     arr[b,hidden[b],j]=np.nan
    if not hidden.any():continue
    pred=deep_forward(net,torch.tensor(arr,dtype=torch.float32),torch.tensor(cols));hm=torch.tensor(hidden);truth=torch.tensor(truth,dtype=torch.float32)
    loss=torch.abs(pred[hm]-truth[hm]).mean()
   assert torch.isfinite(loss),'Nonfinite training loss'
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
   if step%100==0:print(key,step,round(loss.item(),5),flush=True)
  net.eval();torch.save(net.state_dict(),ckpt);models[key]=net
  logs.append({'method':method,'seed':seed,'steps':STEPS,'final_loss':float(loss),'training_seconds':time.time()-start})
  pd.DataFrame(logs).to_csv(OUT/'training_log.csv',index=False)

@torch.no_grad()
def deep_predict(net,raw,mask):
 z=(raw-mu)/sd;rows,cols=np.where(mask);out=np.full((len(raw),P),np.nan);net.eval()
 for a in range(0,len(rows),32):
  rr=rows[a:a+32];cc=cols[a:a+32];starts=np.clip(rr-WINDOW//2,0,len(raw)-WINDOW)
  arr=torch.tensor(np.stack([z[s:s+WINDOW] for s in starts]),dtype=torch.float32)
  pred=deep_forward(net,arr,torch.tensor(cc)).numpy();out[rr,cc]=pred[np.arange(len(rr)),rr-starts]*sd[cc]+mu[cc]
 return out

@torch.no_grad()
def mp_predict(net,raw):
 # Fixed 168-hour graph windows; parameters frozen after training.
 z=(raw-mu)/sd;out=np.zeros((len(raw),P));net.eval()
 for a in range(0,len(raw),WINDOW):
  b=min(a+WINDOW,len(raw));s=max(0,b-WINDOW);v=net(torch.tensor(z[s:b],dtype=torch.float32))[-1].numpy();out[a:b]=v[a-s:,:P]*sd[:P]+mu[:P]
 return out

records=[]
for seed in SEEDS:
 raw=X[nt:nt+nv].copy();mask,lens=masking(raw,seed);raw[:,:P][mask]=np.nan
 pred=baselines(raw,hours[nt:nt+nv])
 for method in names[3:]:
  net=models[f'{method}_{seed}'];pred[method]=mp_predict(net,raw) if method.startswith('MPIN') else deep_predict(net,raw,mask)
 rr,cc=np.where(mask)
 for method,v in pred.items():
  assert np.isfinite(v[rr,cc]).all()
  for r,c in zip(rr,cc):
   site,pol=channels[c];records.append({'seed':seed,'method':method,'site':site,'pollutant':pol,'units':groups[site].iloc[0][pol+'_units'],'datetime_utc':times[nt+r],'gap_hours':int(lens[r,c]),'truth':float(X[nt+r,c]),'prediction':float(v[r,c])})
 print('Scored validation seed',seed,'hidden',int(mask.sum()),flush=True)
p=pd.DataFrame(records);p.to_csv(OUT/'validation_predictions.csv',index=False)
p['abs_error']=abs(p.prediction-p.truth);p['sq_error']=(p.prediction-p.truth)**2
score=p.groupby(['pollutant','units','method','seed']).agg(MAE=('abs_error','mean'),MSE=('sq_error','mean'),n=('truth','size')).reset_index();score['RMSE']=np.sqrt(score.MSE);score.to_csv(OUT/'validation_scores_by_seed.csv',index=False)
summary=score.groupby(['pollutant','units','method']).agg(MAE_mean=('MAE','mean'),MAE_sd=('MAE','std'),RMSE_mean=('RMSE','mean'),RMSE_sd=('RMSE','std'),n_evaluations=('n','sum')).reset_index();summary.to_csv(OUT/'imputation_comparison.csv',index=False)
p['gap_group']=pd.cut(p.gap_hours,[0,1,4,8],labels=['1','2-4','5-8']);gs=p.groupby(['pollutant','units','method','seed','gap_group'],observed=True).agg(MAE=('abs_error','mean'),MSE=('sq_error','mean'),n=('truth','size')).reset_index();gs['RMSE']=np.sqrt(gs.MSE);gs.to_csv(OUT/'validation_scores_by_gap.csv',index=False)
selected={}
for pol,part in summary.groupby('pollutant'):
 best=part.MAE_mean.min();cand=part[part.MAE_mean<=best*1.01].method.to_list();selected[pol]=min(cand,key=names.index)
(OUT/'selected_methods.json').write_text(json.dumps(selected,indent=2));print('Selected',selected,flush=True)
# Freeze selection before touching test-period inputs. No test truth hidden or scored.
# Predict real gaps separately within each split; do not propagate imputed values as observations.
all_candidates={n:np.full((T,P),np.nan) for n in names}; eligibility=np.column_stack([groups[s][p+'_pilot_imputation_eligible'].to_numpy()==1 for s,p in channels])
for which in ['train','validation','test']:
 ix=np.flatnonzero(split==which);raw=X[ix].copy();mask=eligibility[ix]&~np.isfinite(raw[:,:P]);pred=baselines(raw,hours[ix])
 for method in names[3:]:
  # Use fixed seed 11, chosen in advance, for generated data; all 5 seeds were scored.
  net=models[f'{method}_11'];pred[method]=mp_predict(net,raw) if method.startswith('MPIN') else deep_predict(net,raw,mask)
 for method in names:all_candidates[method][ix]=pred[method]
 print('Real gap inference',which,int(mask.sum()),flush=True)
result=d.copy();audit=[]
for pol in pols:
 result[pol+'_imputed']=result[pol];result[pol+'_was_imputed']=0;result[pol+'_imputation_method']='none';result[pol+'_imputation_scope']='offline_reconstruction'
for j,(site,pol) in enumerate(channels):
 group=groups[site];idx=group.index.to_numpy();fill=eligibility[:,j]&~np.isfinite(X[:,j]);method=selected[pol];v=all_candidates[method][:,j];assert np.isfinite(v[fill]).all()
 result.loc[idx[fill],pol+'_imputed']=v[fill];result.loc[idx[fill],pol+'_was_imputed']=1;result.loc[idx[fill],pol+'_imputation_method']=method
 for r in np.flatnonzero(fill):
  rec={'site':site,'pollutant':pol,'datetime_utc':times[r],'evaluation_split':split[r],'gap_id':group.iloc[r][pol+'_gap_id'],'original_status':group.iloc[r][pol+'_status'],'selected_method':method,'filled_value':float(v[r])}
  rec.update({m:float(all_candidates[m][r,j]) for m in names});audit.append(rec)
pd.testing.assert_frame_equal(result[d.columns],d)
for pol in pols:
 observed=d[pol].notna();assert result.loc[observed,pol+'_imputed'].equals(d.loc[observed,pol])
 structural=d[pol+'_structural_non_monitoring']==1;assert result.loc[structural,pol+'_imputed'].isna().all()
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_hash
result.to_csv(OUT/'EPA_AQS_ERA5_2022_OFFLINE_IMPUTED.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'imputation_audit_all_methods.csv',index=False)
metrics={'rows':len(result),'columns':len(result.columns),'filled_measurements':len(audit),'remaining_missing_measurements':sum(result[p+'_imputed'].isna().sum() for p in pols),'structural_missing_measurements':sum((d[p+'_structural_non_monitoring']==1).sum() for p in pols),'hidden_evaluations_per_method':len(p)//len(names),'selected_methods':selected,'source_preserved':True,'test_scored':False}
(OUT/'result_summary.json').write_text(json.dumps(metrics,indent=2,default=int));print(metrics,flush=True)
