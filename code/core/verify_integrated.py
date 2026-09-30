# 对应问题与章节：问题三；第9节
# 功能：在运输复核上加入中继、保障条件及完整通信证书
# 重点函数或流程：run
# 数据路径相对本包根目录；时刻和时长用秒，距离用米，能量用kWh。
# 运行方法及输入输出见同目录《代码与图表索引.md》；默认复核不搜索新方案。

"""Independent schedule arithmetic, guarded inventory and interval RF replay.
The RF verifier uses projected upper envelopes, separately from task scheduling.
Terrain/energy assumptions are explicit; this is not a field safety certificate.
"""
from pathlib import Path
from collections import defaultdict, Counter
import argparse,json,math,csv,copy
import numpy as np
import transport as tr
import communication as cm
from robust_comm import MarginTerrain
from verify_transport import verify,peak,csvwrite
from schedule_model import dump

ROOT=tr.ROOT
def run(source,out):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sol=json.loads(source.read_text(encoding='utf-8'));tv=verify(source,out)
    config=sol['model'];alpha=config['energy_increase_guard'];delay=config['common_delay_guard_s'];margin=config['extra_link_margin_db']
    base=cm.Terrain(ROOT/'data/processed/dem.tif',ROOT/'data/processed/terrain_envelope.npz');ter=MarginTerrain(base,margin)
    cert=[];failed=[];energy=0.;busy=defaultdict(list);min_soc=1.;relay_energy=0.;min_hard=math.inf;min_expected=math.inf
    for r in sol['records']:
        g=tr.TYPES[r['type']];soc=1-(1+alpha)*r['energy_kwh']/g['energy_kwh'];assert soc>=.2-1e-8;min_soc=min(min_soc,soc)
        # Re-evaluate guarded charging, rather than trusting saved guarded release.
        full=tr.CHG[r['type']];charge=full*(.65*(.9-soc)/.9+.35) if soc<.9 else full*.35*(1-soc)/.1
        busy['battery:'+r['battery']].append((r['start_s'],r['return_s']+charge,r['sortie']))
        for b,t in r['deliveries'].items():
            c=r['start_s']+t;hs=tr.HARD[b]-c;es=tr.BOX[b]['expected_s']-c;min_hard=min(min_hard,hs);min_expected=min(min_expected,es)
            assert hs>=max(config['hard_buffer_s'],delay)-1e-5 and es>=delay-1e-5
        energy+=r['energy_kwh']
    rmap={r['id']:r for r in sol['relay_records']};relay_geo=[]
    from single_site_model import supercover
    for r in sol['relay_records']:
        assert r['uav'] in ['R01','R02'] and r['component'] in [f'BR{i:02d}' for i in range(1,7)]
        p=np.asarray(r['xyz']);O=cm.XYZ['O01'];dist=float(np.linalg.norm(p[:2]-O[:2]));ts=np.linspace(0,1,math.ceil(dist/2)+1)
        pp=O[:2]+ts[:,None]*(p[:2]-O[:2]);lo,la=cm.INV.transform(pp[:,0],pp[:,1]);cc,rr=base.native_inv*(lo,la);cells=set()
        for k in range(len(ts)-1):cells|=supercover(cc[k],rr[k],cc[k+1],rr[k+1])
        maximum=max(float(base.original[y,x]) for y,x in cells);H=maximum+50
        assert abs(H-r['cruise_m'])<1e-7 and H>=p[2]-1e-7
        ground=base.native_height(p[:2]);assert abs(p[2]-ground-r['hover_agl_m'])<1e-7 and 0<=p[2]-ground<=300+1e-7
        tout=(H-O[2])/4+dist/15+(H-p[2])/3;back=(H-p[2])/4+dist/15+(H-O[2])/3
        ready=r['start_s']+180+tout+30;end=r['service_end_s'];ret=end+back
        assert end>=ready-1e-7 and r['start_s']>=-1e-7
        ee=1.15*(2*dist/15)/3600+23.5*9.81*((H-O[2])+(H-p[2]))/(.72*3.6e6)+1.1*(end-ready+30)/3600
        for x,y in [(ready,r['service_start_s']),(ret,r['return_s']),(ee,r['energy_kwh'])]:assert abs(x-y)<1e-6
        soc=1-(1+alpha)*ee/3.2;assert soc>=.2-1e-8
        charge=1800*(.65*(.9-soc)/.9+.35) if soc<.9 else 1800*.35*(1-soc)/.1
        busy['component:'+r['component']].append((r['start_s'],ret+charge,r['id']));busy['relay:'+r['uav']].append((r['start_s'],ret+300,r['id']))
        bh=ter.point_link(cm.GW,p,'backhaul');assert bh and bh['margin_db']>=margin-1e-8
        relay_energy+=ee;relay_geo.append(dict(id=r['id'],cruise_m=H,path_dem_max_m=maximum,hover_m=float(p[2]),guarded_soc=soc,backhaul_margin_db=bh['margin_db']))
    gaps=[]
    for key,iv in busy.items():
        iv.sort()
        for a,b in zip(iv,iv[1:]):
            gap=b[0]-a[1];assert gap>=-1e-5,(key,a,b,gap);gaps.append(dict(resource=key,previous=a[2],next=b[2],gap_s=gap))
    air_time=0
    for r in sol['records']:
        rel=[rmap[k] for k in set(r['relay_mission_ids'].values())]
        cc,ff=cm.certify_solution(ter,dict(records=[r]),rel,min_dt=.01)
        cert+=cc;failed+=ff
        assert not ff,(r['sortie'],ff[:1])
        zz=sorted(cc,key=lambda c:c['start_s']);assert abs(zz[0]['start_s']-r['takeoff_s'])<1e-5 and abs(zz[-1]['end_s']-r['return_s'])<1e-5
        for a,b in zip(zz,zz[1:]):assert abs(a['end_s']-b['start_s'])<1e-5
        air_time+=r['return_s']-r['takeoff_s']
    assert abs(sum(c['end_s']-c['start_s'] for c in cert)-air_time)<1e-5
    joint=max(r['return_s'] for r in sol['records']+sol['relay_records']);assert abs(joint-sol['joint_objective'][2])<1e-5
    result=dict(status='PASS_CONDITIONAL_STRICT_MODEL',boxes=tv['boxes_delivered_once'],hard_deadline_violations=tv['hard_deadline_violations'],expected_deadline_violations=tv['expected_deadline_violations'],transport_sorties=len(sol['records']),relay_sorties=len(sol['relay_records']),joint_finish_s=joint,total_energy_kwh=energy+relay_energy,weighted_mean_delivery_s=sol['weighted_mean_delivery_s'],min_guarded_transport_soc=min_soc,energy_increase=alpha,common_delay=delay,min_hard_slack_s=min_hard,min_expected_slack_s=min_expected,minimum_guarded_resource_gap_s=min((g['gap_s'] for g in gaps),default=None),certificate_intervals=len(cert),unresolved_intervals=len(failed),minimum_certified_margin_db=min(c['margin_db'] for c in cert),relay_geometry=relay_geo,scope='Declared strict DEM+50 cruise, unchanged trajectories/DEM and energy formula. Energy multipliers and recharge, radio attenuation, and common shift may co-occur. Independent task delays and terrain uncertainty are not covered.')
    rows=[dict(sortie=c['sortie'],phase=c['phase'],leg=c['leg'],start_s=c['start_s'],end_s=c['end_s'],provider=c['provider'],mode=c['mode'],margin_db=c['margin_db'],clearance_m=c.get('clearance_lower_m')) for c in cert]
    csvwrite(out/'communication.csv',rows);csvwrite(out/'guarded_gaps.csv',gaps);dump(result,out/'verification.json');print(json.dumps(result,ensure_ascii=False,indent=2))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.source,a.out)
