# 对应问题与章节：问题一；第2—3节
# 功能：航段地形净空、安全载荷与单点数量状态动态规划
# 重点函数或流程：main、supercover、energy
# 数据路径相对本包根目录；时刻和时长用秒，距离用米，能量用kWh。
# 运行方法及输入输出见同目录《代码与图表索引.md》；默认复核不搜索新方案。

"""Reproducible Q1 baseline, NOT a Q2-Q4 solver.

Energy assumptions (explicit closure of energy subterms):
  horizontal = battery usable kWh * distance / equivalent range(load)
  climb = (empty mass + payload) * 9.81 * climb / efficiency / 3.6e6
Operation time includes preparation, loading, flight, and complete handoff.
Flight horizontal geometry: WGS84 / UTM zone 49N; DEM is queried on its native grid.
A projected straight segment is inverse-transformed at <=step-m intervals;
EVERY native raster cell touched by each intervening polyline segment is visited.
Default 5 m polyline spacing. This is not a claim of exact continuous GIS geometry.
"""
from __future__ import annotations
import argparse, csv, itertools, json, math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
import rasterio
from pyproj import Transformer

CATEGORIES = ['医疗物资','饮用水','应急食品','生活卫生用品']

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# 枚举线段穿越的原生栅格，保留边界接触像元以避免漏掉高地。
def supercover(x0: float,y0: float,x1: float,y1: float):
    """Cells touched by a line in continuous raster coordinates, including ties."""
    events=[0.,1.]
    for a,b in [(x0,x1),(y0,y1)]:
        if abs(b-a)>1e-14:
            for k in range(math.ceil(min(a,b)),math.floor(max(a,b))+1):
                t=(k-a)/(b-a)
                if 0<t<1: events.append(t)
    ev=sorted(set(events))
    probes=ev+[(a+b)/2 for a,b in zip(ev[:-1],ev[1:])]
    result=set()
    for t in probes:
        x=x0+(x1-x0)*t; y=y0+(y1-y0)*t
        xs={math.floor(x)}; ys={math.floor(y)}
        if abs(x-round(x))<1e-8: xs|={round(x)-1,round(x)}
        if abs(y-round(y))<1e-8: ys|={round(y)-1,round(y)}
        result.update((r,c) for r in ys for c in xs)
    return result

# 载荷相关的水平能耗加含电池质量的爬升势能；返回kWh。
def energy(g: dict, arc: dict, payload: float) -> float:
    if not 0 <= payload <= g['capacity_kg']+1e-9:
        return math.inf
    L=g['range_empty_m']-(g['range_empty_m']-g['range_full_m'])*(payload/g['capacity_kg'])**1.5
    return (g['energy_kwh']*arc['distance_m']/L
            +(g['empty_mass_kg']+payload)*9.81*arc['climb_m']/g['up_eff']/3.6e6)

def flight_time(g: dict,arc: dict) -> float:
    return arc['distance_m']/g['cruise_mps']+arc['climb_m']/g['up_mps']+arc['descent_m']/g['down_mps']

def charge_time(soc: float, full: float) -> float:
    if not 0<=soc<=1: raise ValueError('SOC out of range')
    return full*(.65*(.9-soc)/.9+.35) if soc<.9 else full*.35*(1-soc)/.1

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dem',type=Path,required=True)
    p.add_argument('--inputs',type=Path,default=Path(__file__).resolve().parents[2]/'data/processed/transport.json')
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[2]/'results/optimization/q1_recomputed')
    p.add_argument('--step-m',type=float,default=5.)
    a=p.parse_args()
    if a.step_m<=0: p.error('--step-m must be positive')
    d=json.loads(a.inputs.read_text(encoding='utf-8')); a.out.mkdir(parents=True,exist_ok=True)
    nodes=d['nodes']; types=d['types']; boxes=d['boxes']; node_by={n['id']:n for n in nodes}
    if len(boxes)!=len({b['id'] for b in boxes}): raise ValueError('Duplicate box IDs')
    if any(b['node'] not in node_by for b in boxes): raise ValueError('Unknown service node')
    fwd=Transformer.from_crs('EPSG:4326','EPSG:32649',always_xy=True)
    inv=Transformer.from_crs('EPSG:32649','EPSG:4326',always_xy=True)
    xy={n['id']:np.array(fwd.transform(n['lon'],n['lat'])) for n in nodes}
    geom={}; node_checks=[]
    with rasterio.open(a.dem) as ds:
        dem=ds.read(1); trans=~ds.transform
        if ds.crs.to_epsg()!=4326: raise ValueError('Expected source DEM EPSG:4326')
        valid=np.isfinite(dem)&(dem!=-32767)
        if ds.nodata is not None: valid &= dem!=ds.nodata
        meta={'height':ds.height,'width':ds.width,'crs':str(ds.crs),'tag_nodata':ds.nodata,
              'invalid_pixels':int((~valid).sum()),'min_m':float(dem[valid].min()),'max_m':float(dem[valid].max())}
        for n in nodes:
            r,c=ds.index(n['lon'],n['lat'])
            if not(0<=r<ds.height and 0<=c<ds.width and valid[r,c]): raise ValueError('Invalid node terrain')
            node_checks.append({'node':n['id'],'table_elev_m':n['elev_m'],'native_cell_elev_m':float(dem[r,c]),'difference_m':n['elev_m']-float(dem[r,c])})
        for i,j in itertools.combinations(nodes,2):
            start=xy[i['id']]; end=xy[j['id']]; dist=float(np.linalg.norm(end-start))
            ts=np.linspace(0,1,max(2,math.ceil(dist/a.step_m)+1))
            points=start[None,:]+ts[:,None]*(end-start)[None,:]
            lon,lat=inv.transform(points[:,0],points[:,1]); cols,rows=trans*(lon,lat)
            cells=set()
            for k in range(len(ts)-1): cells |= supercover(cols[k],rows[k],cols[k+1],rows[k+1])
            if any(not(0<=r<ds.height and 0<=c<ds.width) for r,c in cells): raise ValueError('Arc outside DEM')
            if any(not valid[r,c] for r,c in cells): raise ValueError('Arc intersects NoData; do not assume zero')
            zmax=max(float(dem[r,c]) for r,c in cells); H=zmax+50
            zi=i['elev_m']+(0 if i['id']=='O01' else 30)
            zj=j['elev_m']+(0 if j['id']=='O01' else 30)
            if H<max(zi,zj): raise ValueError('Prescribed cruise lower than work height')
            for ni,nj,za,zb in [(i,j,zi,zj),(j,i,zj,zi)]:
                geom[ni['id'],nj['id']]={'from':ni['id'],'to':nj['id'],'distance_m':dist,'terrain_max_m':zmax,'cruise_m':H,'climb_m':H-za,'descent_m':H-zb,'cells':len(cells)}
    write_csv(a.out/'nodes_check.csv',node_checks)
    write_csv(a.out/'arcs_240.csv',list(geom.values()))
    capacities=[]; batches=[]; scenarios=[]
    grouped=defaultdict(list)
    for b in boxes: grouped[b['node']].append(b)
    for rho in [.10,.15,.20,.25,.30]:
        total=(0,0.,0.); all_chosen=[]
        for sid,bs in sorted(grouped.items()):
            out=geom['O01',sid]; back=geom[sid,'O01']
            amounts=tuple(sum(b['category']==c for b in bs) for c in CATEGORIES)
            props={b['category']:(b['mass_kg'],b['volume_m3']) for b in bs}
            patterns=[]
            for g in types:
                def rt(q): return energy(g,out,q)+energy(g,back,0.)
                limit=(1-rho)*g['energy_kwh']
                if rt(0)>limit:
                    qmax=None
                else:
                    lo=0.; hi=g['capacity_kg']
                    for _ in range(70):
                        mid=(lo+hi)/2
                        if rt(mid)<=limit: lo=mid
                        else: hi=mid
                    qmax=lo
                capacities.append({'reserve_pct':round(rho*100),'node':sid,'type':g['id'],'safe_payload_kg':qmax,'nominal_payload_kg':g['capacity_kg']})
                if qmax is None: continue
                for counts in itertools.product(*(range(v+1) for v in amounts)):
                    n=sum(counts)
                    if n==0: continue
                    mass=sum(c*props[cat][0] for c,cat in zip(counts,CATEGORIES) if c)
                    vol=sum(c*props[cat][1] for c,cat in zip(counts,CATEGORIES) if c)
                    if mass>qmax+1e-8 or vol>g['volume_m3']+1e-10: continue
                    e=rt(mass)
                    fly=flight_time(g,out)+flight_time(g,back)
                    dur=g['prep_s']+g['load_per_box_s']*n+fly+g['handoff_base_s']+g['handoff_per_box_s']*n
                    patterns.append((counts,g['id'],mass,vol,e,dur,fly,1-e/g['energy_kwh']))
            @lru_cache(None)
            def solve(state):
                if not any(state): return (0,0.,0.),()
                best=(math.inf,math.inf,math.inf); chosen=()
                first=next(k for k,c in enumerate(state) if c)
                for idx,patt in enumerate(patterns):
                    counts=patt[0]
                    if counts[first]==0 or any(c>s for c,s in zip(counts,state)): continue
                    prev,tail=solve(tuple(s-c for s,c in zip(state,counts)))
                    obj=(prev[0]+1,prev[1]+patt[4],prev[2]+patt[5])
                    if obj<best: best=obj; chosen=(idx,)+tail
                return best,chosen
            obj,chosen=solve(amounts)
            if not math.isfinite(obj[0]):
                raise ValueError(f'Q1 infeasible at node {sid}, reserve {rho}; preserve this case explicitly')
            total=tuple(u+v for u,v in zip(total,obj))
            slots={cat:sorted([b['id'] for b in bs if b['category']==cat]) for cat in CATEGORIES}
            for idx in chosen:
                counts,typ,mass,vol,e,dur,fly,soc=patterns[idx]; ids=[]
                for cnt,cat in zip(counts,CATEGORIES):
                    ids.extend(slots[cat][:cnt]); slots[cat]=slots[cat][cnt:]
                row={'sortie':f'Q1-{round(rho*100):02d}-{len(all_chosen)+1:03d}','node':sid,'type':typ,'boxes':';'.join(ids),'mass_kg':mass,'volume_m3':vol,'flight_s':fly,'operation_s':dur,'energy_kwh':e,'return_soc_pct':100*soc}
                all_chosen.append(row)
            assert all(not v for v in slots.values())
        scenarios.append({'reserve_pct':round(rho*100),'sorties':total[0],'energy_kwh':total[1],'sum_operation_s':total[2],**dict(Counter(x['type'] for x in all_chosen))})
        if abs(rho-.20)<1e-8: batches=all_chosen
    write_csv(a.out/'payload_sensitivity_225.csv',capacities)
    write_csv(a.out/'q1_batches_assumptions.csv',batches)
    write_csv(a.out/'q1_sensitivity.csv',scenarios)
    delivered=[bid for row in batches for bid in row['boxes'].split(';')]
    assert sorted(delivered)==sorted(b['id'] for b in boxes)
    hard={b['id']:min(([b['first_deadline_s']] if b['first_batch']=='是' else [])+([b['expected_s']] if b['category']=='医疗物资' else []))
          for b in boxes if b['first_batch']=='是' or b['category']=='医疗物资'}
    deadlines=[{'box':b['id'],'node':b['node'],'category':b['category'],'hard_deadline_s':hard.get(b['id']),'expected_s':b['expected_s'],'priority':b['priority']} for b in boxes]
    write_csv(a.out/'box_deadlines.csv',deadlines)
    summary={'scope':'Data audit plus exact per-node Q1 count-state DP under documented assumptions. Scope: single-site capability and batching.',
             'coordinate_crs':'EPSG:32649','polyline_step_m':a.step_m,'terrain':meta,
             'boxes':len(boxes),'mass_kg':sum(b['mass_kg'] for b in boxes),'volume_m3':sum(b['volume_m3'] for b in boxes),
             'first_batch_boxes':sum(b['first_batch']=='是' for b in boxes),'hard_boxes':len(hard),'hard_deadlines_histogram':dict(Counter(hard.values())),
             'q1_scenarios':scenarios,'validated_q1_exact_box_coverage':True,
             'charge_examples':{'20pct_to100_A_s':charge_time(.2,1800),'90pct_to100_A_s':charge_time(.9,1800)}}
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
