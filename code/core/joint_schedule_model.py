# 对应问题与章节：问题二与三；第4—5节
# 功能：联合调度模型，支持中继服务窗口拆分及组件复用
# 重点函数或流程：geometry、obligations、build_chains、solve
# 数据路径相对本包根目录；时刻和时长用秒，距离用米，能量用kWh。
# 运行方法及输入输出见同目录《代码与图表索引.md》；默认复核不搜索新方案。

"""Strict-rule continuous scheduling with independent aircraft/battery chains.
Candidate routes and relay locations are fixed in each subproblem; missions,
responsibility assignments and resource orders are optimized, not globally exact.
"""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import argparse, copy, json, math, time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
import transport as tr
import communication as cm
import joint_windows as jw
from robust_comm import MarginTerrain

ROOT=tr.ROOT
def dump(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

class Model:
    def __init__(self):self.lo=[];self.hi=[];self.integ=[];self.rows=[];self.lb=[];self.ub=[];self.names=[]
    def var(self,name,lo=0.,hi=np.inf,integer=False):
        i=len(self.lo);self.lo.append(lo);self.hi.append(hi);self.integ.append(int(integer));self.names.append(name);return i
    def row(self,d,lo=-np.inf,hi=np.inf):self.rows.append(d);self.lb.append(lo);self.ub.append(hi)
    def solve(self,c,limit,gap=1e-5):
        rr=[];cc=[];vv=[]
        for i,d in enumerate(self.rows):
            for j,v in d.items():
                if v:rr.append(i);cc.append(j);vv.append(v)
        A=coo_matrix((vv,(rr,cc)),shape=(len(self.rows),len(self.lo))).tocsc()
        res=milp(np.asarray(c),integrality=np.asarray(self.integ),bounds=Bounds(self.lo,self.hi),constraints=LinearConstraint(A,self.lb,self.ub),options=dict(time_limit=limit,mip_rel_gap=gap))
        if res.x is not None:
            violation=max(0,float(np.max(np.asarray(self.lb)-A@res.x)),float(np.max(A@res.x-np.asarray(self.ub))),float(np.max(np.asarray(self.lo)-res.x)),float(np.max(res.x-np.asarray(self.hi))))
            assert violation<2e-5,violation
        return res

def geometry(margin=.3):
    base=cm.Terrain(ROOT/'data/processed/dem.tif',ROOT/'data/processed/terrain_envelope.npz');ter=MarginTerrain(base,margin)
    jw.TER=base
    pts={p['id']:copy.deepcopy(p) for p in json.loads((ROOT/'data/processed/relay_candidates.json').read_text(encoding='utf-8'))}
    for p in pts.values():
        p['geometry']=jw.path_geometry(p['xyz'])
        bh=ter.point_link(cm.GW,np.asarray(p['xyz']),'backhaul')
        assert bh is not None,p['id'];p['backhaul_margin_db']=bh['margin_db']
    jw.POINTS=pts
    return ter,pts

# 构造每个运输架次在候选中继点的通信责任区间。
def obligations(record,ter,pts,spacing=150.):
    r=copy.deepcopy(record);r['start_s']=0.;cert=[]
    # Stable prior responsibility first; alternate sites are available for new routes.
    prior=r.get('relay_point')
    if prior is None:
        votes={k:sum(tr.BOX[b]['priority'] for b in r['boxes'] if jw.REGION.get(tr.BOX[b]['node'])==k) for k in pts}
        prior=max(votes,key=votes.get)
    keys=([prior] if prior in pts else [])+[k for k in pts if k!=prior]
    rel=[pts[k] for k in keys]
    for seg in cm.trajectory(r):
        dist=float(np.linalg.norm(seg['p1']-seg['p0']));cuts=np.linspace(0,1,max(1,math.ceil(dist/spacing))+1)
        for a,b in zip(cuts,cuts[1:]):
            pa=seg['p0']+a*(seg['p1']-seg['p0']);pb=seg['p0']+b*(seg['p1']-seg['p0'])
            ta=seg['start_s']+a*(seg['end_s']-seg['start_s']);tb=seg['start_s']+b*(seg['end_s']-seg['start_s'])
            ca,fa=cm.certify_interval(ter,pa,pb,float(ta),float(tb),rel,min_dt=.05)
            if fa:return None
            cert.extend(dict(phase=seg['phase'],leg=seg['leg'],**c) for c in ca)
    needs={}
    for c in cert:
        if c['provider']=='G01':continue
        k=c['provider'];old=needs.get(k,[math.inf,-math.inf]);needs[k]=[min(old[0],c['start_s']),max(old[1],c['end_s'])]
    r['needs']=needs;r['relative_certificates']=cert
    return r

def import_seed(name,ter,pts):
    """读取随包提供的名义参考路线；B是兼容调用中的参考解标识。"""
    if name not in ['B','nominal']:
        raise ValueError('可用参考解为nominal；其他方案请使用joint_search、batch_search或position_search的source参数。')
    return [obligations(r,ter,pts) for r in json.loads((ROOT/'results/optimization/joint_nominal.json').read_text(encoding='utf-8'))['records']]

# 后继弧和链起点共同描述每类资源，链数由库存限制。
def build_chains(m,indices,starts,latest,earliest,durations,num,prefix):
    roots={j:m.var(f'{prefix}_root_{j}',0,1,True) for j in indices};arcs={}
    for i in indices:
        for j in indices:
            if i==j or earliest[i]+durations[i]>latest[j]+1e-7:continue
            v=m.var(f'{prefix}_{i}_{j}',0,1,True);arcs[i,j]=v
            big=max(0.,latest[i]+durations[i]-earliest[j])
            m.row({starts[j]:1,starts[i]:-1,v:-big},lo=durations[i]-big)
    for j in indices:m.row({roots[j]:1,**{v:1 for (i,k),v in arcs.items() if k==j}},lo=1,hi=1)
    for i in indices:m.row({v:1 for (k,j),v in arcs.items() if k==i},hi=1)
    m.row({v:1 for v in roots.values()},hi=num)
    return roots,arcs

# 把整数解中的资源链还原为实际设备编号。
def decode_chains(x,roots,arcs,names,starts):
    heads=sorted((i for i,v in roots.items() if x[v]>.5),key=lambda i:x[starts[i]]);nxt={i:j for (i,j),v in arcs.items() if x[v]>.5};out={}
    assert len(heads)<=len(names)
    for name,i in zip(names,heads):
        while True:
            assert i not in out;out[i]=name
            if i not in nxt:break
            i=nxt[i]
    return out

def solve(records,pts,visits,alpha=.03,hard_buffer=300.,common_delay=120.,horizon=8500.,limit=60.,second_limit=30.,objective='finish',mean_cap=None,energy_cap=None,fixed_orders=None,split_windows=True,reuse_components=True):
    if records is None or any(r is None for r in records):return None,dict(status='uncertified_geometry')
    if any((1+alpha)*r['energy_kwh']> .8*tr.TYPES[r['type']]['energy_kwh']+1e-8 for r in records):return None,dict(status='transport_energy_guard_infeasible')
    n=len(records);kcount=len(visits)
    if not reuse_components:assert kcount<=6
    m=Model();s={};lo={};hi={};dur={};bdur={};wait_const=0.;weight=sum(b['priority'] for b in tr.BOX.values());coeff={}
    for i,r in enumerate(records):
        dur[i]=r['duration_s'];soc=1-(1+alpha)*r['energy_kwh']/tr.TYPES[r['type']]['energy_kwh'];bdur[i]=dur[i]+tr.charge(soc,tr.CHG[r['type']])
        hi[i]=min([horizon-dur[i]]+[min(tr.HARD[b]-max(hard_buffer,common_delay),tr.BOX[b]['expected_s']-common_delay)-dt for b,dt in r['deliveries'].items()]);lo[i]=0.
        if hi[i]<0:return None,dict(status='intrinsic_deadline_infeasible')
        s[i]=m.var(f's_{i}',0,hi[i]);coeff[i]=sum(tr.BOX[b]['priority'] for b in r['boxes'])/weight
        wait_const+=sum(tr.BOX[b]['priority']*dt for b,dt in r['deliveries'].items())/weight
    C=m.var('joint_return',0,horizon);rs={};re={};choices={};ready={};ret={};power=1.1/3600
    for k,pid in enumerate(visits):
        p=pts[pid];gg=p['geometry'];ready[k]=180+gg['outbound_s']+30;ret[k]=gg['return_flight_s']
        rs[k]=m.var(f'relay_start_{k}',0,horizon-ready[k]-ret[k]);re[k]=m.var(f'relay_end_{k}',ready[k],horizon-ret[k])
        maxserve=(2.56/(1+alpha)-gg['flight_energy_kwh'])/power-30
        m.row({re[k]:1,rs[k]:-1},lo=ready[k],hi=ready[k]+maxserve)
        m.row({C:1,re[k]:-1},lo=ret[k])
    for i,r in enumerate(records):
        m.row({C:1,s[i]:-1},lo=dur[i])
        windows=[]
        for pid,(a,b) in r['needs'].items():
            spans=[]
            if split_windows:
                for ct in sorted(r['relative_certificates'],key=lambda ct:ct['start_s']):
                    if ct['provider']!=pid:continue
                    aa,bb=ct['start_s'],ct['end_s']
                    if spans and aa<=spans[-1][1]+1e-7:spans[-1][1]=max(bb,spans[-1][1])
                    else:spans.append([aa,bb])
            else:spans=[[a,b]]
            for w,(aa,bb) in enumerate(spans):windows.append((pid,pid if len(spans)==1 else f'{pid}:{w}',aa,bb))
        for pid,label,a,b in windows:
            options=[k for k,p in enumerate(visits) if p==pid]
            if not options:return None,dict(status='missing_relay_site',point=pid)
            vv=[]
            for k in options:
                v=m.var(f'assign_{i}_{label}_{k}',0,1,True);choices[i,label,k]=v;vv.append(v)
                big=horizon+ready[k]+b+2
                m.row({s[i]:1,rs[k]:-1,v:-big},lo=ready[k]-a+1-big)
                m.row({re[k]:1,s[i]:-1,v:-big},lo=b+1-big)
            m.row({v:1 for v in vv},lo=1,hi=1)
    for k in rs:m.row({v:1 for (i,p,j),v in choices.items() if j==k},lo=1)
    chains={}
    for g in tr.TYPES:
        ids=[i for i,r in enumerate(records) if r['type']==g]
        for kind,times,names in [('air',dur,tr.FLEET[g]),('battery',bdur,tr.BAT[g])]:chains[g,kind]=(*build_chains(m,ids,s,hi,lo,times,len(names),g+kind),names)
    rroots={k:m.var(f'relay_root_{k}',0,1,True) for k in rs};rarcs={}
    for i in rs:
        for j in rs:
            if i==j:continue
            v=m.var(f'relay_arc_{i}_{j}',0,1,True);rarcs[i,j]=v;big=horizon+ret[i]+300
            m.row({rs[j]:1,re[i]:-1,v:-big},lo=ret[i]+300-big)
    for j in rs:m.row({rroots[j]:1,**{v:1 for (i,k),v in rarcs.items() if k==j}},lo=1,hi=1)
    for i in rs:m.row({v:1 for (k,j),v in rarcs.items() if k==i},hi=1)
    m.row({v:1 for v in rroots.values()},hi=2)
    croots={};carcs={}
    if reuse_components and kcount>6:
        # Guarded consumed energy is affine in service duration. Recharge is a
        # continuous two-segment concave function; disjunctive equalities encode
        # both segments exactly instead of assuming six distinct components.
        qc={}
        for k,pid in enumerate(visits):
            en=m.var(f'component_energy_{k}',0,2.56)
            ec=(1+alpha)*(pts[pid]['geometry']['flight_energy_kwh']+power*(30-ready[k]))
            ep=(1+alpha)*power
            m.row({en:1,re[k]:-ep,rs[k]:ep},lo=ec,hi=ec)
            z=m.var(f'charge_above_10pct_{k}',0,1,True);qc[k]=m.var(f'charge_s_{k}',0,1800)
            m.row({en:1,z:-2.56},hi=.32);m.row({en:1,z:-.32},lo=0)
            a0=1800*.35/.1/3.2;a1=1800*.65/.9/3.2;b1=1800*(.35-.65*.1/.9)
            m.row({qc[k]:1,en:-a0,z:-6000},hi=0);m.row({qc[k]:1,en:-a0,z:6000},lo=0)
            m.row({qc[k]:1,en:-a1,z:6000},hi=b1+6000);m.row({qc[k]:1,en:-a1,z:-6000},lo=b1-6000)
        croots={k:m.var(f'component_root_{k}',0,1,True) for k in rs}
        for i in rs:
            for j in rs:
                if i==j:continue
                v=m.var(f'component_arc_{i}_{j}',0,1,True);carcs[i,j]=v;big=horizon+ret[i]+1800
                m.row({rs[j]:1,re[i]:-1,qc[i]:-1,v:-big},lo=ret[i]-big)
        for j in rs:m.row({croots[j]:1,**{v:1 for (i,k),v in carcs.items() if k==j}},lo=1,hi=1)
        for i in rs:m.row({v:1 for (k,j),v in carcs.items() if k==i},hi=1)
        m.row({v:1 for v in croots.values()},hi=6)
    # Identical visits at the same site can be ordered without removing any unlabeled solution.
    for pid in set(visits):
        kk=[k for k,p in enumerate(visits) if p==pid]
        for a,b in zip(kk,kk[1:]):m.row({rs[b]:1,rs[a]:-1},lo=0)
    econst=sum(r['energy_kwh'] for r in records)+sum(pts[p]['geometry']['flight_energy_kwh']+power*(30-ready[k]) for k,p in enumerate(visits))
    ecoeff={re[k]:power for k in rs};ecoeff.update({rs[k]:-power for k in rs})
    if mean_cap is not None:m.row({s[i]:coeff[i] for i in s},hi=mean_cap-wait_const)
    if energy_cap is not None:m.row(ecoeff,hi=energy_cap-econst)
    if fixed_orders:
        for g,kind in chains:
            roots,arcs,_=chains[g,kind]
            chosen=fixed_orders[g,kind]
            for ij,v in arcs.items():m.lo[v]=m.hi[v]=int(ij in chosen)
    cost=np.zeros(len(m.lo))
    if objective=='finish':cost[C]=1
    else:
        for i in s:cost[s[i]]=coeff[i]
    st=time.time();res=m.solve(cost,limit)
    info=dict(stage1_status=int(res.status),message=res.message,runtime_s=time.time()-st,variables=len(m.lo),binaries=sum(m.integ),rows=len(m.rows),objective=objective,split_communication_windows=split_windows,reusable_components=reuse_components,scope='Fixed routes, sites and visit multiplicities; split communication duties and independent resource chains. Above six missions, six rechargeable components may be reused. No full-problem optimality claim.')
    if res.x is None:return None,info
    x=res.x;info.update(primal=float(res.fun)+(wait_const if objective!='finish' else 0),lower_bound=float(res.mip_dual_bound)+(wait_const if objective!='finish' else 0),gap=float(res.mip_gap))
    info['solver_gap']=info['gap'];info['gap']=max(0.,info['primal']-info['lower_bound'])/max(1e-12,abs(info['primal']))
    if objective=='finish':m.hi[C]=float(x[C])+.0001
    else:m.row({s[i]:coeff[i] for i in s},hi=float(res.fun)+.0001)
    cost2=np.zeros(len(m.lo))
    if objective=='finish':
        for i in s:cost2[s[i]]=coeff[i]
    else:cost2[C]=1
    fixed=copy.deepcopy(m)
    for j,z in enumerate(fixed.integ):
        if z:fixed.lo[j]=fixed.hi[j]=round(x[j])
    lp2=fixed.solve(cost2,10)
    if lp2.x is not None:
        x=lp2.x
        m.row({j:float(v) for j,v in enumerate(cost2) if v},hi=float(lp2.fun)+1e-5)
    st=time.time();r2=m.solve(cost2,second_limit);info['stage2']=dict(status=int(r2.status),runtime_s=time.time()-st)
    if r2.x is not None and (lp2.x is None or r2.fun<lp2.fun+1e-7):x=r2.x
    if r2.x is not None:info['stage2'].update(primal=float(cost2@x)+(wait_const if objective=='finish' else 0),bound=float(r2.mip_dual_bound)+(wait_const if objective=='finish' else 0),gap=float(r2.mip_gap))
    info['stage2']['fixed_chain_lp_value']=None if lp2.x is None else float(lp2.fun)+(wait_const if objective=='finish' else 0)
    if 'primal' in info['stage2']:
        z=info['stage2'];z['solver_gap']=z['gap'];z['gap']=max(0.,z['primal']-z['bound'])/max(1e-12,abs(z['primal']))
    # Energy minimization LP with all integer decisions and achieved time objectives fixed.
    ml=copy.deepcopy(m)
    for j,z in enumerate(ml.integ):
        if z:ml.lo[j]=ml.hi[j]=round(x[j])
    ml.hi[C]=min(ml.hi[C],float(x[C])+.0001);ml.row({s[i]:coeff[i] for i in s},hi=sum(coeff[i]*x[s[i]] for i in s)+.0001)
    ce=np.zeros(len(ml.lo))
    for j,v in ecoeff.items():ce[j]=v
    r3=ml.solve(ce,10)
    if r3.x is not None:x=r3.x
    assignments={}
    for (g,kind),(roots,arcs,names) in chains.items():assignments[g,kind]=decode_chains(x,roots,arcs,names,s)
    ra=decode_chains(x,rroots,rarcs,['R01','R02'],rs);recs=[]
    for i,r in enumerate(records):
        row=copy.deepcopy(r);start=max(0,float(x[s[i]]));g=r['type']
        mapping={pid:f'RE{k+1:02d}' for (j,pid,k),v in choices.items() if j==i and x[v]>.5}
        row.update(sortie=f'T{i+1:03d}',start_s=start,takeoff_s=start+r['takeoff_offset_s'],return_s=start+dur[i],battery_ready_s=start+dur[i]+r['charge_s'],guarded_battery_ready_s=start+bdur[i],uav=assignments[g,'air'][i],battery=assignments[g,'battery'][i],relay_mission_ids=mapping,relay_mission_id=next(iter(mapping.values())) if len(mapping)==1 else None)
        recs.append(row)
    jw.POINTS=pts
    comp=decode_chains(x,croots,carcs,[f'BR{k+1:02d}' for k in range(6)],rs) if croots else {k:f'BR{k+1:02d}' for k in rs}
    relays=[jw.relay_record(pid,max(0,float(x[rs[k]])),max(float(x[re[k]]),max(0,float(x[rs[k]]))+ready[k]),ra[k],comp[k],k+1) for k,pid in enumerate(visits)]
    joint=max(r['return_s'] for r in recs+relays);ee=sum(r['energy_kwh'] for r in recs+relays);ws=sum((r['start_s']+dt)*tr.BOX[b]['priority'] for r in recs for b,dt in r['deliveries'].items())
    sol=dict(records=recs,relay_records=relays,joint_objective=[0.,0.,joint,ee,n,kcount],objective=[0.,0.,max(r['return_s'] for r in recs),sum(r['energy_kwh'] for r in recs),n],weighted_delivery_s=ws,weighted_mean_delivery_s=ws/weight,last_delivery_s=max(r['start_s']+dt for r in recs for dt in r['deliveries'].values()),model=dict(coordinate_crs='EPSG:32649',relay_cruise_rule='exact DEM path maximum plus 50m, hover at or below cruise',energy_closure='declared declared horizontal range + climb potential',energy_increase_guard=alpha,hard_buffer_s=hard_buffer,common_delay_guard_s=common_delay,extra_link_margin_db=.3,endpoint_buffer_s=1.),optimization=info)
    info.update(joint_finish_s=joint,weighted_mean_s=ws/weight,total_energy_kwh=ee)
    return sol,info

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',default='B');ap.add_argument('--visits',default='P1,P2,P3,P3,P4');ap.add_argument('--limit',type=float,default=60);ap.add_argument('--second-limit',type=float,default=30);ap.add_argument('--alpha',type=float,default=.03);ap.add_argument('--hard-buffer',type=float,default=300);ap.add_argument('--delay',type=float,default=120);ap.add_argument('--horizon',type=float,default=8500);ap.add_argument('--objective',default='finish');ap.add_argument('--name',default='integrated_B');ap.add_argument('--mean-cap',type=float);ap.add_argument('--energy-cap',type=float);a=ap.parse_args()
    ter,pts=geometry();records=import_seed(a.seed,ter,pts)
    dump(records,ROOT/'results/optimization'/f'{a.name}_geometry.json');print('GEOMETRY',a.seed,None if records is None else len(records),flush=True)
    sol,info=solve(records,pts,a.visits.split(','),alpha=a.alpha,hard_buffer=a.hard_buffer,common_delay=a.delay,horizon=a.horizon,limit=a.limit,second_limit=a.second_limit,objective=a.objective,mean_cap=a.mean_cap,energy_cap=a.energy_cap)
    dump(info,ROOT/'results/optimization'/f'{a.name}_solve.json')
    if sol:dump(sol,ROOT/'results/optimization'/f'{a.name}_solution.json')
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
