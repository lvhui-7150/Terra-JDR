# 对应问题与章节：问题三；第5.1—5.2节
# 功能：DEM上包络、链路预算、运动轨迹与连续区间认证
# 重点函数或流程：Terrain、trajectory、certify_interval、certify_solution
# 数据路径相对本包根目录；时刻和时长用秒，距离用米，能量用kWh。
# 运行方法及输入输出见同目录《代码与图表索引.md》；默认复核不搜索新方案。

"""Conservative continuous-interval communications checker on a DEM envelope.
Distance is bounded on each linear trajectory interval. LOS is certified over the
entire swept triangle, not merely at equally spaced time samples. An uncertified
interval is NOT automatically a true outage: it may be conservatism.
The source DEM is piecewise constant in native cells. A 30-m projected envelope
assigns every source-cell elevation to every intersected projected bounding-box
cell (0.10 m numerical padding), so it never intentionally lowers terrain.
"""
from __future__ import annotations
import math,json,itertools
from pathlib import Path
from functools import lru_cache
import numpy as np
import rasterio
from rasterio.features import rasterize
from affine import Affine
from pyproj import Transformer
from transport import DATA,ARCS,TYPES,ROOT,write_csv
FWD=Transformer.from_crs(4326,32649,always_xy=True)
INV=Transformer.from_crs(32649,4326,always_xy=True)
NODE={n['id']:n for n in DATA['nodes']}
XYZ={n['id']:np.array([*FWD.transform(n['lon'],n['lat']),n['elev_m']+(0 if n['id']=='O01' else 30)]) for n in DATA['nodes']}
GW=XYZ['O01'].copy();GW[2]+=20
FSPL_BASE=32.45+20*math.log10(2400)
BUDGET={'direct':122.,'access':116.,'backhaul':126.}
DLOS={k:1000*10**((v-FSPL_BASE)/20) for k,v in BUDGET.items()}
DOBS={k:v/math.sqrt(10) for k,v in DLOS.items()}

class Terrain:
    def __init__(self,source:Path,cache:Path|None=None):
        self.ds=rasterio.open(source);self.original=self.ds.read(1);self.native_inv=~self.ds.transform
        if cache and cache.exists():
            q=np.load(cache);self.z=q['z'];self.x0=float(q['x0']);self.y0=float(q['y0']);self.res=float(q['res'])
        else:
            self.build_envelope()
            if cache:np.savez_compressed(cache,z=self.z,x0=self.x0,y0=self.y0,res=self.res)
    def build_envelope(self):
        res=30.;pad=.10;h,w=self.original.shape
        lon0,lat0=self.ds.transform*(0,0);lon1,lat1=self.ds.transform*(w,h)
        corner=np.array([FWD.transform(lo,la) for lo in [lon0,lon1] for la in [lat0,lat1]])
        self.x0=math.floor((corner[:,0].min()-100)/res)*res;self.y0=math.floor((corner[:,1].min()-100)/res)*res
        W=math.ceil((corner[:,0].max()+100-self.x0)/res);H=math.ceil((corner[:,1].max()+100-self.y0)/res)
        out=np.full((H,W),-np.inf,dtype=np.float32)
        for start in range(0,h,80):
            end=min(start+80,h)
            cc,rr=np.meshgrid(np.arange(w+1),np.arange(start,end+1))
            lo,la=self.ds.transform*(cc,rr);xx,yy=FWD.transform(lo,la)
            xmin=np.minimum.reduce([xx[:-1,:-1],xx[1:,:-1],xx[:-1,1:],xx[1:,1:]])-pad
            xmax=np.maximum.reduce([xx[:-1,:-1],xx[1:,:-1],xx[:-1,1:],xx[1:,1:]])+pad
            ymin=np.minimum.reduce([yy[:-1,:-1],yy[1:,:-1],yy[:-1,1:],yy[1:,1:]])-pad
            ymax=np.maximum.reduce([yy[:-1,:-1],yy[1:,:-1],yy[:-1,1:],yy[1:,1:]])+pad
            c0=np.floor((xmin-self.x0)/res).astype(int);c1=np.floor((xmax-self.x0)/res).astype(int)
            r0=np.floor((ymin-self.y0)/res).astype(int);r1=np.floor((ymax-self.y0)/res).astype(int)
            zz=self.original[start:end]
            for dc in range(int((c1-c0).max())+1):
                for dr in range(int((r1-r0).max())+1):
                    valid=(c0+dc<=c1)&(r0+dr<=r1)&np.isfinite(zz)&(zz!=-32767)
                    np.maximum.at(out,((r0+dr)[valid],(c0+dc)[valid]),zz[valid])
        self.z=out;self.res=res
    def native_height(self,xy):
        lo,la=INV.transform(float(xy[0]),float(xy[1]));c,r=self.native_inv*(lo,la);r=int(math.floor(r));c=int(math.floor(c))
        if not(0<=r<self.original.shape[0] and 0<=c<self.original.shape[1]):return math.inf
        return float(self.original[r,c])
    def sample_native_max(self,a,b,step=10.):
        d=float(np.linalg.norm(np.asarray(a)[:2]-np.asarray(b)[:2]));t=np.linspace(0,1,max(2,math.ceil(d/step)+1))
        pp=np.asarray(a)[:2]+t[:,None]*(np.asarray(b)[:2]-np.asarray(a)[:2]);lo,la=INV.transform(pp[:,0],pp[:,1]);cc,rr=self.native_inv*(lo,la)
        return float(self.original[np.floor(rr).astype(int),np.floor(cc).astype(int)].max())
    def los_sweep(self,q,p0,p1):
        """Return a safe lower bound of clearance for all rays q -> p(t).
        Triangle rasterization is all-touched. The height bound uses radial min/max
        over each whole projected cell, so cells outside the exact beam only make
        the certificate more conservative, never falsely permissive.
        """
        pts=np.array([q[:2],p0[:2],p1[:2]],dtype=float)
        cc=(pts[:,0]-self.x0)/self.res;rr=(pts[:,1]-self.y0)/self.res
        c0=max(0,int(math.floor(cc.min()))-1);c1=min(self.z.shape[1]-1,int(math.floor(cc.max()))+1)
        r0=max(0,int(math.floor(rr.min()))-1);r1=min(self.z.shape[0]-1,int(math.floor(rr.max()))+1)
        if cc.min()<0 or rr.min()<0 or cc.max()>=self.z.shape[1] or rr.max()>=self.z.shape[0]:return -math.inf
        cr=np.column_stack([cc-c0,rr-r0]);area=abs(np.cross(cr[1]-cr[0],cr[2]-cr[0]))
        if area<1e-7:
            far=p0 if np.linalg.norm(p0[:2]-q[:2])>=np.linalg.norm(p1[:2]-q[:2]) else p1
            geo={'type':'LineString','coordinates':[(float(cc[0]-c0),float(rr[0]-r0)),((far[0]-self.x0)/self.res-c0,(far[1]-self.y0)/self.res-r0)]}
        else:geo={'type':'Polygon','coordinates':[[tuple(x) for x in [*cr,cr[0]]]]}
        mask=rasterize([(geo,1)],out_shape=(r1-r0+1,c1-c0+1),transform=Affine.identity(),all_touched=True,dtype='uint8')
        ir,ic=np.nonzero(mask);ir=ir+r0;ic=ic+c0
        if not len(ir):
            ir=np.array([int(rr[0])]);ic=np.array([int(cc[0])])
        z=self.z[ir,ic]
        if np.any(~np.isfinite(z)):return -math.inf
        xa=self.x0+ic*self.res;xb=xa+self.res;ya=self.y0+ir*self.res;yb=ya+self.res
        dxmin=np.maximum(np.maximum(xa-q[0],q[0]-xb),0);dymin=np.maximum(np.maximum(ya-q[1],q[1]-yb),0)
        rmin=np.hypot(dxmin,dymin)
        rmax=np.hypot(np.maximum(abs(xa-q[0]),abs(xb-q[0])),np.maximum(abs(ya-q[1]),abs(yb-q[1])))
        v=p1[:2]-p0[:2];den=float(v@v)
        tt=float(np.clip((q[:2]-p0[:2])@v/den,0,1)) if den>1e-16 else 0.
        dmin=float(np.linalg.norm(p0[:2]+tt*v-q[:2]));dmax=max(float(np.linalg.norm(p0[:2]-q[:2])),float(np.linalg.norm(p1[:2]-q[:2])))
        zmin=min(p0[2],p1[2]);dz=zmin-q[2]
        if dz>=0:ratio=np.minimum(1.,rmin/max(dmax,1e-12))
        else:ratio=np.minimum(1.,rmax/max(dmin,1e-12))
        lower=q[2]+dz*ratio
        # Coincident horizontal endpoints are a vertical column, min endpoint height applies.
        if dmax<1e-8:lower=np.full_like(z,min(q[2],p0[2],p1[2]),dtype=float)
        return float(np.min(lower-z))
    def link_certificate(self,q,p0,p1,kind):
        dist=max(float(np.linalg.norm(q-p0)),float(np.linalg.norm(q-p1)))
        fspl=FSPL_BASE+20*math.log10(max(dist/1000,1e-12))
        if dist<=DOBS[kind]-1e-7:return dict(mode='worst_case_obstruction',margin_db=BUDGET[kind]-fspl-10,distance_upper_m=dist,clearance_lower_m=None)
        if dist>DLOS[kind]+1e-7:return None
        clear=self.los_sweep(q,p0,p1)
        if clear>.05:return dict(mode='LOS_swept_region',margin_db=BUDGET[kind]-fspl,distance_upper_m=dist,clearance_lower_m=clear)
        return None
    def point_link(self,q,p,kind):return self.link_certificate(q,p,p,kind)


# 把完整运输架次分解为线性运动与交接静止区间。
def trajectory(r):
    """Exact piecewise-linear trajectory using the declared transport arcs."""
    g=TYPES[r['type']];segs=[]
    for idx,leg in enumerate(r['legs']):
        a,b=leg['origin'],leg['destination'];arc=ARCS[a,b];pa=XYZ[a];pb=XYZ[b];pc=pa.copy();pc[2]=arc['cruise_m'];pd=pb.copy();pd[2]=arc['cruise_m']
        t=r['start_s']+leg['start_offset_s'];dt=arc['climb_m']/g['up_mps']
        segs.append(dict(phase='climb',leg=idx+1,start_s=t,end_s=t+dt,p0=pa,p1=pc));t+=dt
        dt=arc['distance_m']/g['cruise_mps'];segs.append(dict(phase='cruise',leg=idx+1,start_s=t,end_s=t+dt,p0=pc,p1=pd));t+=dt
        dt=arc['descent_m']/g['down_mps'];segs.append(dict(phase='descent',leg=idx+1,start_s=t,end_s=t+dt,p0=pd,p1=pb));t+=dt
        if b!='O01':
            st=next(s for s in r['stops'] if s['node']==b);end=r['start_s']+st['delivery_offset_s'];segs.append(dict(phase='handoff',leg=idx+1,start_s=t,end_s=end,p0=pb,p1=pb))
    return segs

# 先以距离最坏值检查链路，再用扫掠区域净空认证；未认证区间递归二分。
def certify_interval(ter,p0,p1,t0,t1,relays,min_dt=.05,depth=0):
    # Priority: certify direct first; when direct availability is uncertain,
    # 'direct_or_relay' means a relay backup exists, with online direct preference.
    dc=ter.link_certificate(GW,p0,p1,'direct')
    if dc:return [dict(start_s=t0,end_s=t1,provider='G01',**dc)],[]
    for r in relays:
        if r.get('service_start_s',-math.inf)>t0+1e-8 or r.get('service_end_s',math.inf)<t1-1e-8:continue
        rc=ter.link_certificate(np.array(r['xyz']),p0,p1,'access')
        if rc:
            rc['margin_db']=min(rc['margin_db'],r['backhaul_margin_db'])
            return [dict(start_s=t0,end_s=t1,provider=r['id'],**rc)],[]
    if t1-t0<=min_dt or depth>=20:
        return [],[dict(start_s=t0,end_s=t1,p0=p0.tolist(),p1=p1.tolist(),reason='No interval certificate; not necessarily a true outage')]
    tm=(t0+t1)/2;pm=(p0+p1)/2
    c1,f1=certify_interval(ter,p0,pm,t0,tm,relays,min_dt,depth+1);c2,f2=certify_interval(ter,pm,p1,tm,t1,relays,min_dt,depth+1)
    return c1+c2,f1+f2

# 逐架次、逐阶段覆盖全部轨迹，返回连续证书与未认证区间。
def certify_solution(ter,solution,relays,min_dt=.5):
    certificates=[];failed=[]
    for r in solution['records']:
        for seg in trajectory(r):
            t0,t1=seg['start_s'],seg['end_s'];events=sorted(set([t0,t1]+[x for re in relays for x in [re.get('service_start_s',-1),re.get('service_end_s',-1)] if t0<x<t1]))
            for a,b in zip(events[:-1],events[1:]):
                pa=seg['p0']+(seg['p1']-seg['p0'])*((a-t0)/(t1-t0));pb=seg['p0']+(seg['p1']-seg['p0'])*((b-t0)/(t1-t0))
                cc,ff=certify_interval(ter,pa,pb,a,b,relays,min_dt)
                for c in cc:certificates.append(dict(sortie=r['sortie'],phase=seg['phase'],leg=seg['leg'],**c))
                for f in ff:failed.append(dict(sortie=r['sortie'],phase=seg['phase'],leg=seg['leg'],**f))
    return certificates,failed
