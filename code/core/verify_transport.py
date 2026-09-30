# 对应问题与章节：问题二；第9节
# 功能：逐箱覆盖、航段能耗、时限和资源冲突复核
# 重点函数或流程：verify、peak
# 数据路径相对本包根目录；时刻和时长用秒，距离用米，能量用kWh。
# 运行方法及输入输出见同目录《代码与图表索引.md》；默认复核不搜索新方案。

"""Independent arithmetic/resource audit of an exported Q2 schedule.
Does NOT call the optimizer's route, energy, charging or scheduling functions.
It shares the declared input transcription and preprocessed arc geometry; it therefore
checks conditional model feasibility, not the truth of the missing energy formula.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[2]

def readcsv(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def csvwrite(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# 半开区间扫描计算资源峰值，边界释放先于同刻重新占用。
def peak(intervals):
    # Group floating-point-equivalent boundaries (100 ns); releases precede starts.
    events=sorted([(round(s,7),1) for s,e in intervals]+[(round(e,7),-1) for s,e in intervals])
    now=0;ans=0
    for t,delta in events:now+=delta;ans=max(ans,now)
    return ans

def verify(schedule_path,out):
    out.mkdir(parents=True,exist_ok=True)
    data=json.loads((ROOT/'data/processed/transport.json').read_text(encoding='utf-8'));sol=json.loads(schedule_path.read_text(encoding='utf-8'))
    gs={x['id']:x for x in data['types']};boxes={x['id']:x for x in data['boxes']};fleet={x['id']:x['type'] for x in data['fleet']}
    inv={x['type']:x for x in data['battery_inventory']}
    arcs={(x['from'],x['to']):{k:v if k in ['from','to'] else float(v) for k,v in x.items()} for x in readcsv(ROOT/'data/processed/arcs_240.csv')}
    all_ids=[];jobs=Counter();ui=defaultdict(list);bi=defaultdict(list);type_ui=defaultdict(list);type_bi=defaultdict(list);checks=[];ds=[];energies=[];durations=[]
    for r in sol['records']:
        typ=r['type'];g=gs[typ];bidlist=r['boxes'];jobs[r['sortie']]+=1;all_ids+=bidlist
        assert r['start_s']>=0 and fleet[r['uav']]==typ
        valid_bids=[f'B{typ}{i+1:02d}' for i in range(inv[typ]['count'])];assert r['battery'] in valid_bids
        assert len(set(bidlist))==len(bidlist) and all(x in boxes for x in bidlist)
        assert len(set(r['nodes']))==len(r['nodes']) and set(r['nodes'])==set(boxes[x]['node'] for x in bidlist)
        q=sum(boxes[x]['mass_kg'] for x in bidlist);vol=sum(boxes[x]['volume_m3'] for x in bidlist)
        assert q<=g['capacity_kg']+1e-8 and vol<=g['volume_m3']+1e-8
        assert abs(q-r['mass_kg'])<1e-8 and abs(vol-r['volume_m3'])<1e-8
        t=r['start_s']+g['prep_s']+len(bidlist)*g['load_per_box_s'];assert abs(t-r['takeoff_s'])<1e-7
        e=0.;previous='O01'
        for legnum,node in enumerate(r['nodes']+['O01'],1):
            a=arcs[previous,node]
            L=g['range_empty_m']-(g['range_empty_m']-g['range_full_m'])*(q/g['capacity_kg'])**1.5
            en=g['energy_kwh']*a['distance_m']/L+(g['empty_mass_kg']+q)*9.81*a['climb_m']/(g['up_eff']*3600000)
            fly=a['climb_m']/g['up_mps']+a['distance_m']/g['cruise_mps']+a['descent_m']/g['down_mps']
            lr=r['legs'][legnum-1];assert lr['origin']==previous and lr['destination']==node
            assert abs(q-lr['payload_kg'])<1e-7 and abs(en-lr['energy_kwh'])<1e-7 and abs(t-r['start_s']-lr['start_offset_s'])<1e-7
            t+=fly;e+=en
            assert abs(t-r['start_s']-lr['end_offset_s'])<1e-7
            if node!='O01':
                delivered=[b for b in bidlist if boxes[b]['node']==node];st=next(st for st in r['stops'] if st['node']==node)
                assert abs(t-r['start_s']-st['arrival_offset_s'])<1e-7 and set(st['boxes'])==set(delivered)
                t+=g['handoff_base_s']+len(delivered)*g['handoff_per_box_s']
                for b in delivered:
                    bx=boxes[b];hard=min(([bx['expected_s']] if bx['category']=='医疗物资' else [])+([bx['first_deadline_s']] if bx['first_batch']=='是' else []),default=math.inf)
                    assert abs(t-r['start_s']-r['deliveries'][b])<1e-7
                    ds.append(dict(box=b,node=node,sortie=r['sortie'],delivery_s=t,expected_s=bx['expected_s'],hard_s=hard if math.isfinite(hard) else None,hard_late_s=max(0,t-hard),soft_late_s=max(0,t-bx['expected_s']),hard_slack_s=hard-t if math.isfinite(hard) else None,soft_slack_s=bx['expected_s']-t,priority=bx['priority']))
                q-=sum(boxes[b]['mass_kg'] for b in delivered)
            previous=node
        assert abs(q)<1e-8 and abs(t-r['return_s'])<1e-7 and abs(e-r['energy_kwh'])<1e-7
        soc=1-e/g['energy_kwh'];assert soc>=g['reserve_pct']/100-1e-9 and abs(soc-r['return_soc'])<1e-8
        full=inv[typ]['full_charge_s'];charging=(.65*(.9-soc)/.9+.35)*full if soc<.9 else .35*(1-soc)/.1*full
        assert abs(t+charging-r['battery_ready_s'])<1e-7
        ui[r['uav']].append((r['start_s'],t,r['sortie']));bi[r['battery']].append((r['start_s'],t+charging,r['sortie']))
        type_ui[typ].append((r['start_s'],t));type_bi[typ].append((r['start_s'],t+charging));energies.append(e);durations.append(t-r['start_s'])
        checks.append(dict(sortie=r['sortie'],type=typ,physical_check='PASS',energy_kwh=e,return_soc=soc,reserve_margin_percentage_points=(soc-g['reserve_pct']/100)*100))
    assert set(all_ids)==set(boxes) and len(all_ids)==len(boxes) and len(set(all_ids))==len(all_ids)
    assert all(n==1 for n in jobs.values())
    resource_rows=[]
    for kind,resources in [('uav',ui),('battery_including_recharge',bi)]:
        for resource,intervals in resources.items():
            intervals.sort()
            for a,b in zip(intervals[:-1],intervals[1:]):assert b[0]>=a[1]-1e-7,(resource,a,b)
            resource_rows.append(dict(kind=kind,resource=resource,tasks=len(intervals),occupancy_s=sum(e-s for s,e,_ in intervals),no_overlap='PASS',first_start_s=intervals[0][0],last_occupied_until_s=intervals[-1][1]))
    assert all(d['hard_late_s']<1e-7 for d in ds)
    tmax=max(r['return_s'] for r in sol['records']);total_e=sum(energies)
    objective=[sum(d['hard_late_s']*d['priority'] for d in ds),sum(d['soft_late_s']*d['priority'] for d in ds),tmax,total_e,len(sol['records'])]
    assert all(abs(a-b)<1e-7 for a,b in zip(objective,sol['objective']))
    hard=[d for d in ds if d['hard_s'] is not None]
    utilization=[]
    for typ in gs:
        n=sum(v==typ for v in fleet.values());rs=[r for r in sol['records'] if r['type']==typ];mp=peak(type_ui[typ]);bp=peak(type_bi[typ])
        assert mp<=n and bp<=inv[typ]['count']
        utilization.append(dict(type=typ,sorties=len(rs),uav_used=len(set(r['uav'] for r in rs)),uav_inventory=n,uav_peak=mp,battery_used=len(set(r['battery'] for r in rs)),battery_inventory=inv[typ]['count'],battery_peak=bp,total_mass_kg=sum(r['mass_kg'] for r in rs),energy_kwh=sum(r['energy_kwh'] for r in rs),operation_s=sum(r['duration_s'] for r in rs),utilization=sum(r['duration_s'] for r in rs)/(n*tmax)))
    report=dict(status='PASS_CONDITIONAL_MODEL',time_comparison_tolerance_s=1e-7,scope='Independent arithmetic and scheduling implementation; declared arc geometry and declared energy closure remain assumptions.',boxes_delivered_once=len(all_ids),hard_deadline_boxes=len(hard),hard_deadline_violations=sum(d['hard_late_s']>1e-7 for d in hard),expected_deadline_violations=sum(d['soft_late_s']>1e-7 for d in ds),weighted_soft_tardiness_s=objective[1],sorties=len(sol['records']),multi_node_sorties=sum(len(r['nodes'])>1 for r in sol['records']),finish_return_s=tmax,last_delivery_s=max(d['delivery_s'] for d in ds),energy_kwh=total_e,total_operation_s=sum(durations),min_return_soc=min(r['return_soc'] for r in sol['records']),min_hard_slack_s=min(d['hard_slack_s'] for d in hard),tightest_hard_box=min(hard,key=lambda d:d['hard_slack_s'])['box'],min_expected_slack_s=min(d['soft_slack_s'] for d in ds),type_resources=utilization,global_optimality_proven=False)
    (out/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    csvwrite(out/'independent_sortie_checks.csv',checks);csvwrite(out/'resource_checks.csv',resource_rows);csvwrite(out/'type_resource_summary.csv',utilization);csvwrite(out/'independent_box_checks.csv',ds)
    print(json.dumps(report,ensure_ascii=False,indent=2));return report
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--schedule',type=Path,default=ROOT/'results/optimization/transport_time.json');ap.add_argument('--out',type=Path,default=ROOT/'results/verification/transport_time');aa=ap.parse_args();verify(aa.schedule,aa.out)
