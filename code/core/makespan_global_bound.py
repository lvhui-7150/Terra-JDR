# 对应问题与章节：问题二、三的下界；第8节
# 功能：全路线工作量/能量外松弛、列生成和整数定价
# 重点函数或流程：column、pricing、run
# 数据路径相对本包根目录；时刻和时长用秒，距离用米，能量用kWh。
# 运行方法及输入输出见同目录《代码与图表索引.md》；默认复核不搜索新方案。

"""Full-route makespan workload relaxation with global MILP pricing.

No route pool, relay grid, energy cap or sortie cap defines the bound. Metric
closure of physical flight times and omitted energy/deadline/communications
constraints only enlarge the feasible set. Every nonempty real sortie maps to
one integer cargo-count tour column; fractional columns further relax it.

Numerical HiGHS branch-and-bound pricing bounds are recorded, not called exact
real-arithmetic certificates. A uniform per-box dual repair makes *unfinished*
pricing bounds usable. Independent finite arithmetic validation is separate.
"""
from schedule_model import Model, dump
import transport as tr
import numpy as np
from scipy.optimize import linprog
from scipy.sparse.csgraph import floyd_warshall
import argparse, json, time

ROOT=tr.ROOT
NODES=[n['id'] for n in tr.DATA['nodes']]
KEYS=sorted({(b['node'],b['category']) for b in tr.BOX.values()})
GROUPS=[[b for b in tr.BOX.values() if (b['node'],b['category'])==k] for k in KEYS]
DEM=np.array([len(g) for g in GROUPS],float)
MASS=np.array([g[0]['mass_kg'] for g in GROUPS])
VOL=np.array([round(g[0]['volume_m3']*1000) for g in GROUPS])
J=len(KEYS); NN=len(NODES)-1
FT={g:floyd_warshall(np.array([[0 if a==b else tr.flight(p,tr.ARCS[a,b]) for b in NODES] for a in NODES]),directed=True) for g,p in tr.TYPES.items()}
E0={g:floyd_warshall(np.array([[0 if a==b else tr.energy(p,tr.ARCS[a,b],0) for b in NODES] for a in NODES]),directed=True) for g,p in tr.TYPES.items()}
DIST=floyd_warshall(np.array([[0 if a==b else tr.ARCS[a,b]['distance_m'] for b in NODES] for a in NODES]),directed=True)
HEIGHT=np.array([n['elev_m']+(0 if n['id']=='O01' else 30) for n in tr.DATA['nodes']])

def tangent(g,q):
    p=tr.TYPES[g];Q=p['capacity_kg'];E=p['energy_kwh'];L=p['range_empty_m'];D=L-p['range_full_m'];d=L-D*(q/Q)**1.5
    slope=E*D*1.5*np.sqrt(q/Q)/Q/d**2
    value=E/d-E/L
    return slope,value-slope*q

def relaxed_energy(g,counts,order):
    p=tr.TYPES[g];q=float(np.dot(counts,MASS));e=0.;path=[0]+list(order)+[0]
    for a,b in zip(path,path[1:]):
        inc=max(tangent(g,t)[0]*q+tangent(g,t)[1] for t in np.linspace(0,p['capacity_kg'],11))
        e+=E0[g][a,b]+DIST[a,b]*inc+q*max(0,HEIGHT[b]-HEIGHT[a])*9.81/(p['up_eff']*3.6e6)
        if b:q-=sum(counts[j]*MASS[j] for j,k in enumerate(KEYS) if k[0]==NODES[b])
    return e

def column(g,counts,order):
    counts=np.asarray(counts,int);p=tr.TYPES[g]
    path=[0]+list(order)+[0]
    dur=p['prep_s']+(p['load_per_box_s']+p['handoff_per_box_s'])*sum(counts)+p['handoff_base_s']*len(order)+sum(FT[g][a,b] for a,b in zip(path,path[1:]))
    assert 0<sum(counts) and np.all(counts<=DEM) and counts@MASS<=p['capacity_kg']+1e-8 and counts@VOL<=1000*p['volume_m3']+1e-8
    assert set(order)=={NODES.index(KEYS[j][0]) for j,n in enumerate(counts) if n}
    return dict(type=g,counts=counts.tolist(),order=[NODES[n] for n in order],duration_s=float(dur),energy_lower_kwh=float(relaxed_energy(g,counts,order)))

# 构建全货类整数定价模型，求最小约化成本及其数值界。
def pricing(g,pi,lam,limit=12,energy_relaxation=False):
    p=tr.TYPES[g];m=Model()
    a=[m.var(f'count_{j}',0,DEM[j],True) for j in range(J)]
    y={i:m.var(f'visit_{i}',0,1,True) for i in range(1,NN+1)}
    x={(i,j):m.var(f'arc_{i}_{j}',0,1,True) for i in range(NN+1) for j in range(NN+1) if i!=j}
    u={i:m.var(f'order_{i}',1,NN) for i in y}
    m.row({a[j]:float(MASS[j]) for j in range(J)},hi=p['capacity_kg'])
    m.row({a[j]:float(VOL[j]) for j in range(J)},hi=1000*p['volume_m3'])
    m.row({a[j]:1 for j in range(J)},lo=1)
    for i in y:
        jj=[j for j,k in enumerate(KEYS) if k[0]==NODES[i]]
        for j in jj:m.row({a[j]:1,y[i]:-DEM[j]},hi=0)
        m.row({y[i]:1,**{a[j]:-1 for j in jj}},hi=0)
    for i in range(NN+1):
        d={x[i,j]:1 for j in range(NN+1) if i!=j}
        e={x[j,i]:1 for j in range(NN+1) if i!=j}
        if i:d[y[i]]=-1;e[y[i]]=-1
        m.row(d,lo=0 if i else 1,hi=0 if i else 1)
        m.row(e,lo=0 if i else 1,hi=0 if i else 1)
    for i in y:
        for j in y:
            if i!=j:m.row({u[i]:1,u[j]:-1,x[i,j]:NN},hi=NN-1)
    if energy_relaxation:
        q={ij:m.var(f'payload_{ij}',0,p['capacity_kg']) for ij in x}
        en={ij:m.var(f'energy_{ij}',0,.8*p['energy_kwh']) for ij in x}
        for ij,v in x.items():
            i,j=ij;m.row({q[ij]:1,v:-p['capacity_kg']},hi=0)
            if j==0:m.hi[q[ij]]=0.
            for at in np.linspace(0,p['capacity_kg'],11):
                slope,intercept=tangent(g,at)
                aa=DIST[i,j]*slope+max(0,HEIGHT[j]-HEIGHT[i])*9.81/(p['up_eff']*3.6e6)
                bb=E0[g][i,j]+DIST[i,j]*intercept
                # Slightly lower each facet to preserve relaxation under rounding.
                m.row({en[ij]:1,q[ij]:-aa,v:-bb},lo=-1e-10)
        for i in y:
            row={q[j,i]:1 for j in range(NN+1) if j!=i};row.update({q[i,j]:-1 for j in range(NN+1) if j!=i})
            row.update({a[j]:-MASS[j] for j,k in enumerate(KEYS) if k[0]==NODES[i]});m.row(row,lo=0,hi=0)
        m.row({v:1 for v in en.values()},hi=.8*p['energy_kwh'])
    c=np.zeros(len(m.lo));unit=p['load_per_box_s']+p['handoff_per_box_s']
    for j in range(J):c[a[j]]=lam*unit-pi[j]
    for i in y:c[y[i]]=lam*p['handoff_base_s']
    for ij,v in x.items():c[v]=lam*FT[g][ij]
    st=time.time();res=m.solve(c,limit,gap=1e-9)
    b=float(res.mip_dual_bound)+lam*p['prep_s'] if getattr(res,'mip_dual_bound',None) is not None else -float(np.maximum(pi,0)@DEM)
    info=dict(type=g,status=int(res.status),message=res.message,lower_reduced_cost=b,primal_reduced_cost=None if res.x is None else float(res.fun)+lam*p['prep_s'],elapsed_s=time.time()-st,nodes=int(getattr(res,'mip_node_count',0) or 0))
    col=None
    if res.x is not None:
        counts=np.rint(res.x[a]).astype(int);nxt={i:j for (i,j),v in x.items() if res.x[v]>.5};order=[];at=nxt[0]
        while at:assert at not in order;order.append(at);at=nxt[at]
        col=column(g,counts,order)
        assert abs(lam*col['duration_s']-pi@counts-info['primal_reduced_cost'])<1e-5
    return col,info

def run(out,iterations=150,seconds=900,pricing_seconds=10,energy_relaxation=False):
    out=ROOT/out;out.mkdir(parents=True,exist_ok=True);cols={};hist=[];best=-1;certificate=None;start=time.time()
    def add(c):
        key=(c['type'],tuple(c['counts']))
        if key not in cols or c['duration_s']<cols[key]['duration_s']-1e-7:cols[key]=c;return True
        return False
    for g in tr.TYPES:
        for j in range(J):
            c=np.zeros(J,int);c[j]=1;add(column(g,c,[NODES.index(KEYS[j][0])]))
    for f in [ROOT/'results/optimization/transport_time.json',ROOT/'results/optimization/joint_fast.json']:
        for r in json.loads(f.read_text(encoding='utf-8'))['records']:
            counts=[sum(b in r['boxes'] for b in [z['id'] for z in gg]) for gg in GROUPS]
            add(column(r['type'],counts,[NODES.index(n) for n in r['nodes']]))
    for folder in ['makespan_bound','energy_bound']:
        prior=ROOT/f'results/optimization/search/{folder}/columns.json'
        if energy_relaxation and prior.exists():
            for c in json.loads(prior.read_text(encoding='utf-8')):
                c=column(c['type'],c['counts'],[NODES.index(n) for n in c['order']])
                if c['energy_lower_kwh']<=.8*tr.TYPES[c['type']]['energy_kwh']+1e-6:add(c)
    for it in range(iterations):
        cc=list(cols.values());K=len(cc);eq=np.column_stack([c['counts'] for c in cc]+[np.zeros(J)])
        ub=np.zeros((3,K+1))
        for k,c in enumerate(cc):ub['ABC'.index(c['type']),k]=c['duration_s']
        ub[:,-1]=[-len(tr.FLEET[g]) for g in 'ABC'];cost=np.zeros(K+1);cost[-1]=1
        lp=linprog(cost,A_ub=ub,b_ub=np.zeros(3),A_eq=eq,b_eq=DEM,bounds=(0,None),method='highs')
        assert lp.success,lp.message
        pi=lp.eqlin.marginals;lam=-lp.ineqlin.marginals
        # Avoid dual feasibility drift in the C column; positive scaling safe.
        scale=max(1.,sum(lam[i]*len(tr.FLEET[g]) for i,g in enumerate('ABC')))+1e-12;pi=pi/scale;lam=lam/scale
        priced=[];added=0
        for i,g in enumerate('ABC'):
            col,inf=pricing(g,pi,lam[i],pricing_seconds,energy_relaxation);priced.append(inf)
            if col and inf['primal_reduced_cost']<-.000001:added+=add(col)
        # Each true sortie has >=1 box, total exactly 80 boxes. This shift repairs
        # ALL omitted columns using valid global pricing lower bounds, not primals.
        delta=min(0.,min(z['lower_reduced_cost'] for z in priced))-1e-5
        bound=float((pi+delta)@DEM)-1e-5
        row=dict(iteration=it,columns=K,restricted_lp_s=float(lp.fun),global_workload_bound_s=bound,delta=delta,pricing=priced,added=int(added),elapsed_s=time.time()-start)
        hist.append(row)
        if bound>best:
            best=bound;certificate=dict(bound_s=bound,classes=[dict(node=k[0],category=k[1],count=int(DEM[j])) for j,k in enumerate(KEYS)],pi=pi.tolist(),lambda_=lam.tolist(),uniform_per_box_shift=delta,pricing=priced,iteration=it,numerical_allowance_s=1e-5,energy_outer_relaxation=energy_relaxation,scope='Global for all nonempty transport routes under the declared flight model; relaxes deadlines, charging and every communication/relay restriction. Optional energy outer approximation uses empty-energy shortest paths plus tangent lower bounds for payload increments. Solver-certified pricing lower bounds, not an exact-arithmetic proof. No total energy or sortie count budget.')
            dump(certificate,out/'certificate.json')
        dump(hist,out/'history.json');dump(list(cols.values()),out/'columns.json')
        print(json.dumps(dict(iteration=it,columns=K,restricted=float(lp.fun),bound=bound,best=best,added=int(added),seconds=time.time()-start)),flush=True)
        if added==0 or time.time()-start>seconds:break
    dump(dict(global_lower_bound_s=best,iterations=len(hist),columns=len(cols),elapsed_s=time.time()-start,global_optimum_proven=False),out/'summary.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/optimization/search/makespan_bound');p.add_argument('--iterations',type=int,default=150);p.add_argument('--seconds',type=float,default=900);p.add_argument('--pricing-seconds',type=float,default=10);p.add_argument('--energy',action='store_true');a=p.parse_args();run(a.out,a.iterations,a.seconds,a.pricing_seconds,a.energy)
