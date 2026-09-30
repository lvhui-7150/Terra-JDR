# 对应问题与章节：问题一至四及下界；第9节
# 功能：完整验证已交付方案并重新计算分区和下界
# 重点函数或流程：顺序执行，最终输出REPLAY_COMPLETE
# 数据路径相对本包根目录；时刻和时长用秒，距离用米，能量用kWh。
# 运行方法及输入输出见同目录《代码与图表索引.md》；默认复核不搜索新方案。

"""Validate submitted frozen schedules and numerical lower-bound certificate."""
from pathlib import Path
import json,hashlib,csv,contextlib,io
import transport as tr
from verify_transport import verify,peak
from verify_integrated import run as verify_joint
from partition_integrated import run as partition
from verify_bound import run as bound
from schedule_model import dump
R=tr.ROOT;A=R/'results/verification';A.mkdir(parents=True,exist_ok=True)
manifest=json.loads((R/'results/optimization/manifest.json').read_text(encoding='utf-8'));checks={}
for name,m in manifest.items():
 src=R/'results/optimization'/f'{name}.json';assert hashlib.sha256(src.read_bytes()).hexdigest()==m['sha256']
 with contextlib.redirect_stdout(io.StringIO()):
  v=verify(src,A/name) if name=='transport_time' else verify_joint(src,A/name)
  p=None if name=='transport_time' else partition(src,A/name)
 if p:
  assert [z['selected']['gap'] for z in p['chosen']]==m['partition_gaps']
  for z in p['chosen']:
   seen=[];needs=[0]*8
   for g in z['selected']['group_details']:
    seen+=g['nodes']
    for i,kind in enumerate(p['resource_order']):
     rows=[a for a in g['assignments'] if a['resource_type']==kind];n=peak([(a['start_s'],a['end_s']) for a in rows]) if rows else 0
     assert n==g['needs'][i];needs[i]+=n
     for rid in {a['resource'] for a in rows}:
      iv=sorted((a['start_s'],a['end_s']) for a in rows if a['resource']==rid);assert all(b[0]>=a[1]-1e-6 for a,b in zip(iv,iv[1:]))
   assert len(seen)==15 and len(set(seen))==15 and needs==z['selected']['needs']
 joint=name!='transport_time';c=v['joint_finish_s'] if joint else v['finish_return_s'];e=v['total_energy_kwh'] if joint else v['energy_kwh']
 assert abs(c-m['finish_s'])<1e-6 and abs(e-m['energy_kwh'])<1e-7
 checks[name]={'feasible':'PASS','partition':'PASS' if p else 'not_applicable'};print('REPLAY_PASS',name,flush=True)
with contextlib.redirect_stdout(io.StringIO()):verify(R/'results/optimization/transport_delivery.json',A/'transport_delivery')
checks['transport_delivery']={'feasible':'PASS'}
rows=list(csv.DictReader((R/'results/optimization/q1_results/q1_batches_assumptions.csv').open(encoding='utf-8-sig')));seen=[];total=0
for r in rows:
 bs=r['boxes'].split(';');seen+=bs;g=tr.TYPES[r['type']];q=sum(tr.BOX[b]['mass_kg'] for b in bs);vol=sum(tr.BOX[b]['volume_m3'] for b in bs);e=0.;d=g['prep_s']+len(bs)*g['load_per_box_s']+g['handoff_base_s']+len(bs)*g['handoff_per_box_s']
 assert q<=g['capacity_kg'] and vol<=g['volume_m3']+1e-9
 for aa,bb,load in [('O01',r['node'],q),(r['node'],'O01',0)]:
  ar=tr.ARCS[aa,bb];length=g['range_empty_m']-(g['range_empty_m']-g['range_full_m'])*(load/g['capacity_kg'])**1.5
  e+=g['energy_kwh']*ar['distance_m']/length+(g['empty_mass_kg']+load)*9.81*ar['climb_m']/(g['up_eff']*3600000)
  d+=ar['distance_m']/g['cruise_mps']+ar['climb_m']/g['up_mps']+ar['descent_m']/g['down_mps']
 assert abs(e-float(r['energy_kwh']))<1e-8 and abs(d-float(r['operation_s']))<1e-8 and e<=.8*g['energy_kwh']+1e-9;total+=e
assert sorted(seen)==sorted(tr.BOX);checks['single_site']={'coverage':'PASS','arithmetic':'PASS','sorties':len(rows),'energy_kwh':total}
with contextlib.redirect_stdout(io.StringIO()):bound('results/optimization/lower_bound.json','results/verification/lower_bound.json',90)
cert=json.loads((A/'lower_bound.json').read_text(encoding='utf-8'));assert cert['all_pricing_optimal'];checks['lower_bound']={'status':'PASS','bound_s':cert['verified_bound_s']}
dump({'status':'PASS','global_optimum_proven':False,'checks':checks},A/'summary.json');print('REPLAY_COMPLETE',flush=True)
