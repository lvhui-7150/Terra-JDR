# 对应问题与章节：问题一至三；第2、4节
# 功能：共用航段能耗、架次时钟、机体/电池调度及路线邻域
# 重点函数或流程：energy、flight、charge、route、schedule、optimize
# 数据路径相对本包根目录；时刻和时长用秒，距离用米，能量用kWh。
# 运行方法及输入输出见同目录《代码与图表索引.md》；默认复核不搜索新方案。

"""Continuous-time route physics and machine/battery list scheduler.
Energy uses the explicitly stated horizontal-range and climb-potential closure.
The numerical optimizer is a reproducible local-search heuristic, not a global proof.
"""
from __future__ import annotations
import csv, itertools, json, math, random
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2]
DATA=json.loads((ROOT/'data/processed/transport.json').read_text(encoding='utf-8'))
BOX={b['id']:b for b in DATA['boxes']}
TYPES={g['id']:g for g in DATA['types']}
FLEET={g:[r['id'] for r in DATA['fleet'] if r['type']==g] for g in TYPES}
BAT={g:[f'B{g}{i+1:02d}' for i in range(next(r['count'] for r in DATA['battery_inventory'] if r['type']==g))] for g in TYPES}
CHG={r['type']:r['full_charge_s'] for r in DATA['battery_inventory']}
with (ROOT/'data/processed/arcs_240.csv').open(encoding='utf-8-sig') as f:
    ARCS={(r['from'],r['to']):{k:(v if k in ['from','to'] else float(v)) for k,v in r.items()} for r in csv.DictReader(f)}

def deadline(b):
    ds=([b['expected_s']] if b['category']=='医疗物资' else [])+([b['first_deadline_s']] if b['first_batch']=='是' else [])
    return min(ds,default=math.inf)
HARD={bid:deadline(b) for bid,b in BOX.items()}

# 载荷相关的水平能耗加含电池质量的爬升势能；返回kWh。
def energy(g,arc,q):
    if q< -1e-8 or q>g['capacity_kg']+1e-8:return math.inf
    L=g['range_empty_m']-(g['range_empty_m']-g['range_full_m'])*(max(0.,q)/g['capacity_kg'])**1.5
    return g['energy_kwh']*arc['distance_m']/L+(g['empty_mass_kg']+q)*9.81*arc['climb_m']/g['up_eff']/3.6e6

# 根据爬升、水平巡航、下降三段速度计算航段秒数。
def flight(g,a):return a['distance_m']/g['cruise_mps']+a['climb_m']/g['up_mps']+a['descent_m']/g['down_mps']
# 由返航SOC反算充满所需秒数，保留题设分段充电规律。
def charge(s,full):return full*(.65*(.9-s)/.9+.35) if s<.9 else full*.35*(1-s)/.1

def write_csv(p,rows):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    if rows:
        with p.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

# 枚举短路线访问顺序，累计准备、装载、交接和飞行时钟；逐点扣减载荷。
@lru_cache(maxsize=160000)
def route(typ:str,bids:tuple[str,...],seq:tuple[str,...]=()):
    """Best short-node permutation; >5 nodes uses a nearest-neighbor insertion seed.
    The cap only changes the ordering heuristic: it does not prohibit longer routes.
    """
    if not bids:return None
    g=TYPES[typ];bs=[BOX[x] for x in bids]
    q=sum(b['mass_kg'] for b in bs);v=sum(b['volume_m3'] for b in bs)
    if q>g['capacity_kg']+1e-9 or v>g['volume_m3']+1e-9:return None
    nodes=sorted(set(b['node'] for b in bs));bn={n:[b for b in bs if b['node']==n] for n in nodes}
    if seq:
        if set(seq)!=set(nodes) or len(seq)!=len(nodes):return None
        orders=[seq]
    elif len(nodes)<=5:orders=itertools.permutations(nodes)
    else:
        todo=set(nodes);order=[];last='O01'
        while todo:
            nxt=min(todo,key=lambda n:ARCS[last,n]['distance_m']);order.append(nxt);todo.remove(nxt);last=nxt
        orders=[tuple(order),tuple(reversed(order)),tuple(sorted(nodes,key=lambda n:min(b['expected_s'] for b in bn[n])))]
    best=None;bestkey=None
    for order in orders:
        load=q;last='O01';ee=0.;t=g['prep_s']+len(bids)*g['load_per_box_s'];takeoff=t;stops=[];legs=[];times={}
        for n in (*order,'O01'):
            arc=ARCS[last,n];en=energy(g,arc,load);ee+=en;fl=flight(g,arc)
            legs.append(dict(origin=last,destination=n,start_offset_s=t,end_offset_s=t+fl,payload_kg=load,energy_kwh=en))
            t+=fl
            if n!='O01':
                arrival=t;t+=g['handoff_base_s']+len(bn[n])*g['handoff_per_box_s']
                stops.append(dict(node=n,arrival_offset_s=arrival,delivery_offset_s=t,boxes=[b['id'] for b in bn[n]]))
                for b in bn[n]:times[b['id']]=t
                load-=sum(b['mass_kg'] for b in bn[n])
            last=n
        if ee>(1-g['reserve_pct']/100)*g['energy_kwh']+1e-9:continue
        # Penalize intrinsically impossible early deliveries, then airborne/handling duration.
        h=sum(max(0,times[b]-HARD[b])*BOX[b]['priority'] for b in bids)
        key=(h,t,ee,sum(times[b]/BOX[b]['expected_s'] for b in bids))
        if bestkey is None or key<bestkey:
            soc=1-ee/g['energy_kwh'];bestkey=key
            best=dict(type=typ,boxes=list(bids),nodes=list(order),mass_kg=q,volume_m3=v,energy_kwh=ee,return_soc=soc,
                      duration_s=t,takeoff_offset_s=takeoff,charge_s=charge(soc,CHG[typ]),deliveries=times,stops=stops,legs=legs)
    return best

# 分别管理同型机体与电池的最早可用时刻，避免把两者绑定成同一资源。
def schedule(tasks,release=None,details=False):
    ua={u:0. for vv in FLEET.values() for u in vv};ba={b:0. for vv in BAT.values() for b in vv}
    records=[];hardlate=0.;softlate=0.;total_e=0.;cmax=0.;lastdelivery=0.;wsum=0.
    for typ,bids in tasks:
        r=route(typ,bids)
        if r is None:return None
        u=min(FLEET[typ],key=lambda u:(ua[u],u));b=min(BAT[typ],key=lambda b:(ba[b],b))
        s=max(ua[u],ba[b],(release(r) if release else 0.))
        end=s+r['duration_s'];ua[u]=end;ba[b]=end+r['charge_s'];total_e+=r['energy_kwh'];cmax=max(cmax,end)
        for bid,dt in r['deliveries'].items():
            dt+=s
            hardlate+=max(0,dt-HARD[bid])*BOX[bid]['priority']
            softlate+=max(0,dt-BOX[bid]['expected_s'])*BOX[bid]['priority']
            lastdelivery=max(lastdelivery,dt);wsum+=dt*BOX[bid]['priority']
        if details:records.append(dict(r,sortie=f'T{len(records)+1:03d}',uav=u,battery=b,start_s=s,takeoff_s=s+r['takeoff_offset_s'],return_s=end,battery_ready_s=ba[b]))
    return dict(objective=(hardlate,softlate,cmax,total_e,len(tasks)),last_delivery_s=lastdelivery,weighted_delivery_s=wsum,records=records)

def initial(seed=0):
    rng=random.Random(seed);tasks=[]
    for node in [n['id'] for n in DATA['nodes'][1:]]:
        todo=sorted([b['id'] for b in DATA['boxes'] if b['node']==node],key=lambda x:(min(HARD[x],BOX[x]['expected_s']),-BOX[x]['priority'],x))
        while todo:
            typ=rng.choices(['A','B','C'],[2,2,1])[0];chosen=[]
            for bid in todo:
                proposed=tuple(sorted(chosen+[bid]))
                if route(typ,proposed):chosen.append(bid)
            if not chosen:continue
            tasks.append((typ,tuple(sorted(chosen))));todo=[b for b in todo if b not in chosen]
    tasks.sort(key=lambda r:(min(min(HARD[b],BOX[b]['expected_s']) for b in r[1]),-sum(BOX[b]['priority'] for b in r[1])/route(*r)['duration_s']))
    return tasks

def scal(obj):
    h,l,c,e,n=obj
    return h*500 + l*10 + c + e*.001+n*.00001

# 有限预算的路线邻域搜索；输出可行上界，不提供整体全局最优证明。
def optimize(iterations=50000,seed=20260923,initial_tasks=None,release=None,compatible=None):
    rng=random.Random(seed);cur=initial_tasks or initial(seed)
    cs=schedule(cur,release);best=list(cur);bs=cs;logs=[]
    for it in range(iterations):
        # Reheat periodically but keep the best feasible solution separately.
        cyc=it%5000
        if cyc==0:
            if it:cur=list(best);cs=bs
            logs.append(dict(iteration=it,objective=bs['objective']))
        temp=120*(1-cyc/5000)**2+0.2
        nxt=list(cur);kind=rng.random();i=rng.randrange(len(nxt));j=rng.randrange(len(nxt))
        if kind<.15:
            ti,bi=nxt[i];nxt[i]=(rng.choice(['A','B','C']),bi)
        elif kind<.35:
            if i==j:continue
            el=nxt.pop(i);nxt.insert(j,el)
        elif kind<.60:
            if i==j:continue
            ti,bi=nxt[i];tj,bj=nxt[j]
            # Bias same-neighborhood moves without excluding arbitrary nodes.
            bid=rng.choice(bi);bi=tuple(b for b in bi if b!=bid);bj=tuple(sorted(bj+(bid,)))
            nxt[j]=(tj,bj)
            if bi:nxt[i]=(ti,bi)
            else:nxt.pop(i)
        elif kind<.79:
            if i==j:continue
            ti,bi=nxt[i];tj,bj=nxt[j];a=rng.choice(bi);b=rng.choice(bj)
            nxt[i]=(ti,tuple(sorted(tuple(x for x in bi if x!=a)+(b,))))
            nxt[j]=(tj,tuple(sorted(tuple(x for x in bj if x!=b)+(a,))))
        elif kind<.90:
            if i==j:continue
            ti,bi=nxt[i];tj,bj=nxt[j]
            nxt[i]=(rng.choice([ti,tj,'C']),tuple(sorted(bi+bj)));nxt.pop(j)
        else:
            ti,bi=nxt[i]
            if len(bi)<=1:continue
            chosen=set(rng.sample(bi,rng.randrange(1,len(bi))));b1=tuple(b for b in bi if b not in chosen);b2=tuple(b for b in bi if b in chosen)
            nxt[i]=(rng.choice([ti,'A','B','C']),b1);nxt.insert(min(j,len(nxt)),(rng.choice([ti,'A','B','C']),b2))
        if compatible and any(not compatible(bids) for _,bids in nxt):continue
        ns=schedule(nxt,release)
        if ns is None:continue
        if ns['objective']<bs['objective']:
            best=list(nxt);bs=ns
        delta=scal(ns['objective'])-scal(cs['objective'])
        if delta<0 or rng.random()<math.exp(-min(700,max(0,delta/temp))):cur=nxt;cs=ns
    return best,schedule(best,release,True),logs

def export(tasks,solution,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'schedule.json').write_text(json.dumps(solution,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'tasks.json').write_text(json.dumps(tasks,ensure_ascii=False,indent=2),encoding='utf-8')
    rs=[];ds=[];legs=[];bats=[]
    for r in solution['records']:
        rs.append({k:r[k] for k in ['sortie','uav','type','battery','start_s','takeoff_s','return_s','battery_ready_s','mass_kg','volume_m3','energy_kwh','return_soc']}|{'nodes':';'.join(r['nodes']),'boxes':';'.join(r['boxes'])})
        bats.append(dict(battery=r['battery'],type=r['type'],sortie=r['sortie'],start_s=r['start_s'],flight_return_s=r['return_s'],charge_start_s=r['return_s'],charge_end_s=r['battery_ready_s'],return_soc=r['return_soc'],next_task_required_soc=1.))
        for bid,dt in r['deliveries'].items():
            d=BOX[bid];t=r['start_s']+dt
            ds.append(dict(box=bid,node=d['node'],sortie=r['sortie'],delivery_s=t,hard_deadline_s=HARD[bid] if math.isfinite(HARD[bid]) else '',expected_s=d['expected_s'],hard_lateness_s=max(0.,t-HARD[bid]),soft_lateness_s=max(0.,t-d['expected_s']),priority=d['priority']))
        for k,leg in enumerate(r['legs'],1):legs.append(dict(sortie=r['sortie'],leg=k,**leg,start_s=leg['start_offset_s']+r['start_s'],end_s=leg['end_offset_s']+r['start_s']))
    write_csv(out/'sorties.csv',rs);write_csv(out/'deliveries.csv',ds);write_csv(out/'battery_cycles.csv',bats);write_csv(out/'legs.csv',legs)

if __name__=='__main__':
    import argparse,time
    p=argparse.ArgumentParser();p.add_argument('--iterations',type=int,default=50000);p.add_argument('--seed',type=int,default=20260923);p.add_argument('--out',type=Path,default=ROOT/'results/optimization/q2');p.add_argument('--initial',type=Path)
    a=p.parse_args();start=time.time()
    init=None
    if a.initial:init=[(g,tuple(bs)) for g,bs in json.loads(a.initial.read_text(encoding='utf-8'))]
    tasks,sol,logs=optimize(a.iterations,a.seed,init);sol['search']={'seed':a.seed,'iterations':a.iterations,'elapsed_s':time.time()-start,'method':'simulated annealing/list scheduling','global_optimality_proven':False,'history':logs}
    export(tasks,sol,a.out);print(json.dumps({k:v for k,v in sol.items() if k!='records'},ensure_ascii=False,indent=2))
