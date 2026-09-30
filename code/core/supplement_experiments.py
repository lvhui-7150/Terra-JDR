# 对应问题与章节：问题一、三、四；第3、6、7节
# 功能：六种目标优先级、45项MILP交叉核验及60项分区阈值实验
# 重点函数或流程：目标枚举、资源链边界反解与分区重选
# 数据路径相对本包根目录；时刻和时长用秒，距离用米，能量用kWh。
# 运行方法及输入输出见同目录《代码与图表索引.md》；默认复核不搜索新方案。

"""Bounded supplementary experiments on objective order and frozen schedules."""
from pathlib import Path
from functools import lru_cache
from collections import Counter,defaultdict
import json,itertools,math,sys,csv
P=Path(__file__).resolve().parents[2];R=P;sys.path.insert(0,str(R/'code/core'))
import transport as tr
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
A=P/'results/optimization';A.mkdir(exist_ok=True)
def dump(v,p):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
cats=sorted({b['category'] for b in tr.BOX.values()})
orders=list(itertools.permutations(range(3)));labels=['架次数','能耗','累计作业']
solutions={order:[] for order in orders};stats=[];crosschecks=[]
for node in sorted({b['node'] for b in tr.BOX.values()}):
 bs=[b for b in tr.BOX.values() if b['node']==node];by={c:sorted([b['id'] for b in bs if b['category']==c]) for c in cats};dem=tuple(len(by[c]) for c in cats);props={b['category']:(b['mass_kg'],b['volume_m3']) for b in bs};patterns=[]
 for counts in itertools.product(*(range(n+1) for n in dem)):
  n=sum(counts)
  if not n:continue
  mass=sum(counts[i]*props[c][0] for i,c in enumerate(cats) if counts[i]);vol=sum(counts[i]*props[c][1] for i,c in enumerate(cats) if counts[i])
  for gname,g in tr.TYPES.items():
   if mass>g['capacity_kg'] or vol>g['volume_m3']+1e-10:continue
   e=tr.energy(g,tr.ARCS['O01',node],mass)+tr.energy(g,tr.ARCS[node,'O01'],0)
   if e>.8*g['energy_kwh']+1e-9:continue
   dur=g['prep_s']+n*g['load_per_box_s']+g['handoff_base_s']+n*g['handoff_per_box_s']+tr.flight(g,tr.ARCS['O01',node])+tr.flight(g,tr.ARCS[node,'O01'])
   patterns.append(dict(type=gname,node=node,counts=counts,mass_kg=mass,volume_m3=vol,energy_kwh=e,duration_s=dur))
 # Independent integer pattern formulation verifies each primary objective.
 coef=np.array([r['counts'] for r in patterns],float).T
 primary_bounds=[]
 for objective in range(3):
  c=np.array([1 if objective==0 else r['energy_kwh'] if objective==1 else r['duration_s'] for r in patterns])
  opt=milp(c,integrality=np.ones(len(c)),bounds=Bounds(np.zeros(len(c)),np.full(len(c),sum(dem))),constraints=LinearConstraint(coef,dem,dem),options={'mip_rel_gap':0,'time_limit':30})
  assert opt.status==0;primary_bounds.append(float(opt.fun));crosschecks.append(dict(node=node,objective=labels[objective],status=int(opt.status),lower_bound=float(opt.mip_dual_bound),objective_value=float(opt.fun)))
 for order in orders:
  @lru_cache(None)
  def dp(state):
   if not any(state):return (0,0.,0.),()
   first=next(i for i,n in enumerate(state) if n);best=None;bestkey=None
   for i,r in enumerate(patterns):
    if r['counts'][first]==0 or any(a>b for a,b in zip(r['counts'],state)):continue
    cost,tail=dp(tuple(b-a for a,b in zip(r['counts'],state)))
    v=(cost[0]+1,cost[1]+r['energy_kwh'],cost[2]+r['duration_s']);key=tuple(round(v[j],10) for j in order)
    if bestkey is None or key<bestkey:bestkey=key;best=(v,(i,)+tail)
   return best
  obj,chosen=dp(dem);assert abs(obj[order[0]]-primary_bounds[order[0]])<1e-5;slots={c:list(by[c]) for c in cats}
  for idx in chosen:
   rr=dict(patterns[idx]);rr['boxes']=[]
   for c,n in zip(cats,rr['counts']):rr['boxes']+=slots[c][:n];slots[c]=slots[c][n:]
   solutions[order].append(rr)
  stats.append(dict(node=node,priority_order=order,patterns=len(patterns),states=dp.cache_info().currsize))
out=[]
for order,recs in solutions.items():
 ids=[b for r in recs for b in r['boxes']];assert sorted(ids)==sorted(tr.BOX)
 # Replay exact physics through the independent record route calculator.
 for r in recs:
  q=tr.route(r['type'],tuple(sorted(r['boxes'])),(r['node'],));assert q and abs(q['energy_kwh']-r['energy_kwh'])<1e-8 and abs(q['duration_s']-r['duration_s'])<1e-7
 row=dict(order=list(order),label=' → '.join(labels[j] for j in order),sorties=len(recs),energy_kwh=sum(r['energy_kwh'] for r in recs),operation_s=sum(r['duration_s'] for r in recs),type_counts=dict(Counter(r['type'] for r in recs)),records=recs);out.append(row)
 print('Q1_PRIORITY',row['label'],row['sorties'],row['energy_kwh'],row['operation_s']/60,flush=True)
main=next(x for x in out if x['order']==[0,1,2]);assert main['sorties']==18 and abs(main['energy_kwh']-59.13166924155935)<1e-7
dump({'scope':'Exact count-state DP under declared physics; lexicographic orders; Q2/Q3 frozen','orders':out,'state_counts':stats,'integer_program_crosscheck':crosschecks},A/'objective_priority_experiment.json')

names=['q2','nominal','robust','balanced','partition_friendly','fast_partition_friendly'];m=json.loads((R/'results/optimization/consolidated_results.json').read_text(encoding='utf-8'))['plans'];NAMES={'q2': 'transport_time', 'nominal': 'joint_nominal', 'robust': 'joint_fast', 'balanced': 'joint_energy', 'partition_friendly': 'joint_two_groups', 'fast_partition_friendly': 'joint_comprehensive'};boundaries=[];beta_rows=[]
for name in names:
 s=json.loads((R/f'results/optimization/{NAMES[name]}.json').read_text(encoding='utf-8'));rs=s['records'];rr=s.get('relay_records',[])
 energy_limits=[(.8*tr.TYPES[r['type']]['energy_kwh']/r['energy_kwh']-1,r['sortie']) for r in rs]+[(.8*3.2/r['energy_kwh']-1,r['id']) for r in rr]
 charge_limits=[];groups=defaultdict(list)
 for r in rs:groups[('transport',r['battery'])].append(r)
 for r in rr:groups[('relay',r['component'])].append(r)
 for (typ,rid),jobs in groups.items():
  jobs=sorted(jobs,key=lambda r:r['start_s'])
  for a,b in zip(jobs,jobs[1:]):
   full=tr.CHG[a['type']] if typ=='transport' else 1800;cap=tr.TYPES[a['type']]['energy_kwh'] if typ=='transport' else 3.2;t=b['start_s']-a['return_s']
   if t>=full:continue
   if t<.35*full:required=1-.1*t/(.35*full)
   else:required=.9*(1-(t/full-.35)/.65)
   alpha=(1-required)*cap/a['energy_kwh']-1;charge_limits.append((alpha,rid,a.get('sortie',a.get('id')),b.get('sortie',b.get('id'))))
 ec=min(energy_limits);bc=min(charge_limits) if charge_limits else (math.inf,'none');slacks=[min(tr.HARD[b],tr.BOX[b]['expected_s'])-r['start_s']-dt for r in rs for b,dt in r['deliveries'].items()]
 limits=dict(plan=name,energy_reserve_limit=ec[0],energy_critical_sortie=ec[1],fixed_battery_chain_limit=bc[0],battery_critical=list(bc[1:]),combined_energy_limit=min(ec[0],bc[0]),common_shift_limit_s=min(slacks),certified_extra_loss_db=m[name]['verification'].get('minimum_certified_margin_db'))
 boundaries.append(limits);print('FROZEN_BOUNDARY',limits,flush=True)
 if name=='q2':continue
 pp=json.loads((R/f'results/verification/{NAMES[name]}/all_partitions.json').read_text(encoding='utf-8'))
 for K in [2,3]:
  for beta in [.05,.10,.15,.20,.25,.30]:
   eligible=[z for z in pp if z['K']==K and z['max_relative_work_deviation']<=beta+1e-9]
   z=min(eligible,key=lambda z:(z['normalized_deficit'],sum(n/i for n,i in zip(z['needs'],[4,2,2,6,4,4,2,6])),z['CV'])) if eligible else None
   beta_rows.append(dict(plan=name,K=K,beta=beta,eligible=len(eligible),deficit=z['normalized_deficit'] if z else None,actual_deviation=z['max_relative_work_deviation'] if z else None,gap=z['gap'] if z else None))
dump(boundaries,A/'frozen_boundaries.json');dump(beta_rows,A/'partition_threshold_experiment.json')
print('SUPPLEMENT_COMPLETE',flush=True)
