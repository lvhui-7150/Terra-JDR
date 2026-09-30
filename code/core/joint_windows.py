# 对应问题与章节：问题三；第5.3节
# 功能：中继航路几何、固定服务窗与初始可行计划辅助函数
# 重点函数或流程：path_geometry、requirements、relay_record
# 数据路径相对本包根目录；时刻和时长用秒，距离用米，能量用kWh。
# 运行方法及输入输出见同目录《代码与图表索引.md》；默认复核不搜索新方案。

"""Continuous joint route/resource search under four physically timed relay service windows.
A heuristic restricted family, not a global optimization proof. All accepted routes
have a continuous-interval communication certificate against G01 + their relay.
Relay cruise altitude is exactly the source-cell path maximum + 50m (no high-hover
relaxation). Aircraft and energy-component availability are independent.
"""
from __future__ import annotations
import argparse,time,copy
import transport as tr
from communication import *
from single_site_model import supercover
from functools import lru_cache
BOX=tr.BOX

GROUP={'S002':'P1','S004':'P1','S005':'P2','S008':'P2','S009':'P2','S003':'P3','S007':'P3','S015':'P3','S010':'P4','S012':'P4','S013':'P4','S014':'P4'}
REGION=dict(GROUP, S001='P1', S006='P3', S011='P3')
POINTS={x['id']:x for x in json.load(open(ROOT/'data/processed/relay_candidates.json'))}
TER=None;GEOM={};WINDOWS={};RELAY_RECORDS=[]

def path_geometry(pos):
    O=XYZ['O01'];pos=np.asarray(pos);dist=float(np.linalg.norm(pos[:2]-O[:2]));ts=np.linspace(0,1,math.ceil(dist/5)+1)
    pp=O[:2]+ts[:,None]*(pos[:2]-O[:2]);lo,la=INV.transform(pp[:,0],pp[:,1]);cc,rr=TER.native_inv*(lo,la)
    cells=set()
    for k in range(len(ts)-1):cells|=supercover(cc[k],rr[k],cc[k+1],rr[k+1])
    zm=max(float(TER.original[r,c]) for r,c in cells);H=zm+50
    assert H>=pos[2]-1e-7,('hover above prescribed cruise',H,pos[2])
    outup=H-O[2];outdown=H-pos[2];inup=H-pos[2];indown=H-O[2]
    out=outup/4+dist/15+outdown/3;back=inup/4+dist/15+indown/3
    E=1.15*2*dist/15/3600+23.5*9.81*(outup+inup)/.72/3600000
    return dict(distance_m=dist,terrain_max_m=zm,cruise_m=H,outbound_s=out,return_flight_s=back,flight_energy_kwh=E,
                out_climb_m=outup,out_descent_m=outdown,in_climb_m=inup,in_descent_m=indown,source_cells=len(cells))

def init():
    global TER,GEOM
    TER=Terrain(ROOT/'data/processed/dem.tif',ROOT/'data/processed/terrain_envelope.npz')
    for p in POINTS.values():p['geometry']=path_geometry(p['xyz'])
    cache=ROOT/'results/optimization/joint_arc_certificates.json'
    if cache.exists():GEOM=json.load(open(cache));return
    # Each continuous geometric phase is parameterized on [0,1].
    for pid in [None,*POINTS]:
        allowed=['O01']+[n for n in XYZ if n!='O01' and (n not in GROUP or GROUP[n]==pid)]
        rel=[] if pid is None else [POINTS[pid]]
        for a,b in itertools.permutations(allowed,2):
            arc=ARCS[a,b];pa=XYZ[a];pb=XYZ[b];pc=pa.copy();pc[2]=arc['cruise_m'];pd=pb.copy();pd[2]=arc['cruise_m'];res=[];ok=True
            for phase,p0,p1 in [('climb',pa,pc),('cruise',pc,pd),('descent',pd,pb)]:
                cert=[];fail=[]
                for aa,bb in zip(np.linspace(0,1,33)[:-1],np.linspace(0,1,33)[1:]):
                    ca,fa=certify_interval(TER,p0+aa*(p1-p0),p0+bb*(p1-p0),float(aa),float(bb),rel,min_dt=1/4096)
                    cert+=ca;fail+=fa
                if fail:ok=False;break
                res.append(dict(phase=phase,certificates=cert))
            if b!='O01':
                cert,fail=certify_interval(TER,pb,pb,0.,1.,rel,min_dt=1/4096)
                if fail:ok=False
                res.append(dict(phase='handoff',certificates=cert))
            GEOM[f'{pid}|{a}|{b}']=res if ok else None
    json.dump(GEOM,open(cache,'w'),ensure_ascii=False,indent=2)
    print('Geometry cache',len(GEOM),'accepted',sum(x is not None for x in GEOM.values()),flush=True)

@lru_cache(maxsize=160000)
def requirements(typ,bids):
    if len({REGION[BOX[b]['node']] for b in bids})>1:return None
    r=tr.route(typ,bids)
    if not r:return None
    pids={GROUP[n] for n in r['nodes'] if n in GROUP}
    if len(pids)>1:return None
    pid=next(iter(pids),None);g=TYPES[typ];amin=math.inf;bmax=-math.inf
    for leg in r['legs']:
        a,b=leg['origin'],leg['destination'];certs=GEOM.get(f'{pid}|{a}|{b}')
        if certs is None:return None
        arc=ARCS[a,b];ds=[arc['climb_m']/g['up_mps'],arc['distance_m']/g['cruise_mps'],arc['descent_m']/g['down_mps']]
        if b!='O01':
            st=next(x for x in r['stops'] if x['node']==b);ds.append(st['delivery_offset_s']-st['arrival_offset_s'])
        t=leg['start_offset_s']
        for seg,dt in zip(certs,ds):
            for c in seg['certificates']:
                if c['provider']!='G01':amin=min(amin,t+c['start_s']*dt);bmax=max(bmax,t+c['end_s']*dt)
            t+=dt
    return r,pid,amin,bmax

def compatible(bids):return len({REGION[BOX[b]['node']] for b in bids})<=1

def relay_record(pid,start,end,rid,eid,idx):
    p=POINTS[pid];gg=p['geometry'];ready=start+180+gg['outbound_s']+30
    assert end>=ready
    energy=gg['flight_energy_kwh']+(end-ready+30)*1.1/3600;soc=1-energy/3.2
    assert soc>=.2,('relay energy',pid,energy)
    ret=end+gg['return_flight_s'];charge=tr.charge(soc,1800)
    return dict(id=f'RE{idx:02d}',point_id=pid,uav=rid,component=eid,start_s=start,takeoff_s=start+180,arrival_s=ready-30,
                service_start_s=ready,service_end_s=end,return_s=ret,uav_ready_s=ret+300,component_ready_s=ret+charge,
                lon=p['lon'],lat=p['lat'],ground_m=p['ground_m'],hover_agl_m=p['hover_agl_m'],xyz=p['xyz'],
                cruise_m=gg['cruise_m'],energy_kwh=energy,return_soc=soc,charge_s=charge,backhaul_margin_db=p['backhaul_margin_db'],geometry=gg)

def set_windows(north_end=1600.,west_end=4000.,east_end=4800.,northwest_end=7000.):
    global WINDOWS,RELAY_RECORDS
    a=relay_record('P1',0,north_end,'R01','BR01',1)
    b=relay_record('P4',0,east_end,'R02','BR02',2)
    c=relay_record('P3',math.ceil(a['uav_ready_s']),west_end,'R01','BR03',3)
    d=relay_record('P2',math.ceil(b['uav_ready_s']),northwest_end,'R02','BR04',4)
    RELAY_RECORDS=[a,b,c,d];WINDOWS={x['point_id']:(x['service_start_s'],x['service_end_s']) for x in RELAY_RECORDS}
    return RELAY_RECORDS

# 分别管理同型机体与电池的最早可用时刻，避免把两者绑定成同一资源。
def schedule(tasks,release=None,details=False):
    ua={u:0. for vv in tr.FLEET.values() for u in vv};ba={b:0. for vv in tr.BAT.values() for b in vv}
    records=[];hardlate=softlate=comm_late=total_e=cmax=lastdelivery=wsum=0.
    for typ,bids in tasks:
        qq=requirements(typ,bids)
        if qq is None:return None
        r,pid,aa,bb=qq;lower=0.;upper=math.inf
        if pid and math.isfinite(aa):
            L,U=WINDOWS[pid];lower=max(0.,L-aa+1.);upper=U-bb-1.
        u=min(tr.FLEET[typ],key=lambda u:(ua[u],u));b=min(tr.BAT[typ],key=lambda b:(ba[b],b))
        s=max(ua[u],ba[b],lower);comm_late+=max(0.,s-upper)
        end=s+r['duration_s'];ua[u]=end;ba[b]=end+r['charge_s'];total_e+=r['energy_kwh'];cmax=max(cmax,end)
        for bid,dt in r['deliveries'].items():
            dt+=s;hardlate+=max(0.,dt-tr.HARD[bid])*BOX[bid]['priority'];softlate+=max(0.,dt-BOX[bid]['expected_s'])*BOX[bid]['priority'];lastdelivery=max(lastdelivery,dt);wsum+=dt*BOX[bid]['priority']
        if details:records.append(dict(r,sortie=f'T{len(records)+1:03d}',uav=u,battery=b,start_s=s,takeoff_s=s+r['takeoff_offset_s'],return_s=end,battery_ready_s=ba[b],relay_point=pid,comm_required_start_offset_s=aa if math.isfinite(aa) else None,comm_required_end_offset_s=bb if math.isfinite(bb) else None,allowed_start_lower_s=lower,allowed_start_upper_s=upper if math.isfinite(upper) else None))
    return dict(objective=(hardlate+50*comm_late,softlate,max(cmax,max(x['return_s'] for x in RELAY_RECORDS)),total_e,len(tasks)),
                hard_weighted_lateness=hardlate,comm_window_lateness_s=comm_late,last_delivery_s=lastdelivery,weighted_delivery_s=wsum,records=records)

def initial(seed=1):
    rng=tr.random.Random(seed);tasks=[]
    order=['S002','S004','S001','S006','S007','S003','S015','S011','S010','S012','S013','S014','S005','S008','S009']
    for n in order:
        bs=sorted([b for b in BOX if BOX[b]['node']==n],key=lambda b:(min(tr.HARD[b],BOX[b]['expected_s']),-BOX[b]['priority']))
        assert any(requirements(g,(bs[0],)) for g in 'ABC'), f'No certified singleton for {n}'
        while bs:
            gt=rng.choices(['A','B','C'],[1,2,2])[0];chosen=[]
            for b in bs:
                if requirements(gt,tuple(sorted(chosen+[b]))):chosen.append(b)
            if not chosen:continue
            tasks.append((gt,tuple(sorted(chosen))));bs=[b for b in bs if b not in chosen]
    return tasks

def run(iterations,seed,init_tasks=None):
    tr.schedule=schedule
    return tr.optimize(iterations,seed,init_tasks or initial(seed),compatible=compatible)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--iterations',type=int,default=200000);ap.add_argument('--seed',type=int,default=41);ap.add_argument('--north-end',type=float,default=1850);ap.add_argument('--west-end',type=float,default=8000);ap.add_argument('--east-end',type=float,default=4000);ap.add_argument('--northwest-end',type=float,default=9000);ap.add_argument('--initial',type=Path);ap.add_argument('--out',type=Path,default=ROOT/'results/optimization/joint_trial');args=ap.parse_args();t0=time.time();init();rs=set_windows(args.north_end,args.west_end,args.east_end,args.northwest_end)
    print('WINDOWS',WINDOWS,flush=True)
    initial_tasks=None
    if args.initial:initial_tasks=[(g,tuple(bs)) for g,bs in json.load(open(args.initial))]
    tasks,sol,logs=run(args.iterations,args.seed,initial_tasks);sol['search']={'iterations':args.iterations,'seed':args.seed,'elapsed_s':time.time()-t0,'global_optimality_proven':False,'history':logs};sol['relay_records']=rs
    # Retain the original transport-objective convention for independent transport verification.
    sol['joint_objective']=sol['objective'];sol['objective']=(sol['hard_weighted_lateness'],sol['objective'][1],max(r['return_s'] for r in sol['records']),sol['objective'][3],len(tasks))
    tr.export(tasks,sol,args.out);print({k:v for k,v in sol.items() if k not in ['records','search','relay_records']},flush=True)
