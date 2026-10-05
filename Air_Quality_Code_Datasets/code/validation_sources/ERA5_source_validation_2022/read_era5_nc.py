"""Read numeric NetCDF variables using the installed VTK NetCDF C library."""
import ctypes as C
from pathlib import Path
import numpy as np
import os
libpath=Path(os.environ['CODEX_PRIMARY_RUNTIME_ROOT'])/'dependencies/python/lib/python3.12/site-packages/vtkmodules/libvtknetcdf-9.3.so'
lib=C.CDLL(str(libpath))
def call(name,*args):
 code=getattr(lib,'vtknetcdf_nc_'+name)(*args)
 if code:raise RuntimeError(f'NetCDF {name}: {code}')
def read(path):
 fid=C.c_int();call('open',str(path).encode(),0,C.byref(fid))
 out={}
 try:
  n=C.c_int();call('inq_nvars',fid,C.byref(n))
  for i in range(n.value):
   name=C.create_string_buffer(257);typ=C.c_int();nd=C.c_int();dims=(C.c_int*100)();na=C.c_int()
   call('inq_var',fid,i,name,C.byref(typ),C.byref(nd),dims,C.byref(na))
   shape=[];dimnames=[]
   for j in range(nd.value):
    sz=C.c_size_t();dn=C.create_string_buffer(257);call('inq_dim',fid,dims[j],dn,C.byref(sz));shape.append(sz.value);dimnames.append(dn.value.decode())
   attrs={}
   for j in range(na.value):
    an=C.create_string_buffer(257);call('inq_attname',fid,i,j,an)
    at=C.c_int();sz=C.c_size_t();call('inq_att',fid,i,an,C.byref(at),C.byref(sz))
    if at.value==2:
     val=C.create_string_buffer(sz.value+1);call('get_att_text',fid,i,an,val);attrs[an.value.decode()]=val.value.decode()
    elif at.value!=12:
     val=(C.c_double*sz.value)();call('get_att_double',fid,i,an,val);attrs[an.value.decode()]=list(val)
   if typ.value in (2,12):continue
   a=np.empty(shape or (),dtype='float64');call('get_var_double',fid,i,a.ctypes.data_as(C.POINTER(C.c_double)))
   fill=attrs.get('_FillValue',[])
   if fill:a=np.where(a==fill[0],np.nan,a)
   a=a*attrs.get('scale_factor',[1])[0]+attrs.get('add_offset',[0])[0]
   out[name.value.decode()]={'values':a,'dimensions':dimnames,'attributes':attrs}
 finally:call('close',fid)
 return out
if __name__=='__main__':
 import zipfile,tempfile,json
 with tempfile.TemporaryDirectory() as td:
  with zipfile.ZipFile('upload/era5_bexar_2022_01.nc') as z:
   for name in z.namelist():
    p=Path(td)/Path(name).name;p.write_bytes(z.read(name));d=read(p)
    print(json.dumps({k:{'shape':v['values'].shape,'range':[float(np.nanmin(v['values'])),float(np.nanmax(v['values']))],'dimensions':v['dimensions'],'attrs':v['attributes']} for k,v in d.items()},indent=2))
