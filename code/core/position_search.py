# 对应问题与章节：问题三；第5.3节
# 功能：多尺度中继位置搜索及完整轨迹重新认证
# 重点函数或流程：run
# 数据路径相对本包根目录；时刻和时长用秒，距离用米，能量用kWh。
# 运行方法及输入输出见同目录《代码与图表索引.md》；默认复核不搜索新方案。

"""Local continuous-coordinate refinement, always interval-certify accepted paths."""
from joint_search import points_from_solution,score
from joint_schedule_model import *
import argparse

def run(source,out,seconds=600,nominal=False):
    out=ROOT/out;out.mkdir(parents=True,exist_ok=True)
    best=json.loads((ROOT/source).read_text(encoding='utf-8'));ter,pts=points_from_solution(best)
    if nominal:ter=MarginTerrain(ter.base if hasattr(ter,'base') else jw.TER,0.)
    margin=0. if nominal else .3;start=time.time();history=[];seen=set();vis=[r['point_id'] for r in best['relay_records']]
    if nominal:
        rr=[obligations(r,ter,pts,spacing=75) for r in best['records']]
        ss,inf=solve(rr,pts,vis,alpha=0,common_delay=0,hard_buffer=0,horizon=best['joint_objective'][2]+.001,limit=60,second_limit=15)
        if ss:best=ss;best['model']['extra_link_margin_db']=0.;dump(best,out/'best_solution.json')
        history.append(dict(name='nominal_seed',**inf));print('NOMINAL',score(best),flush=True)
    for step in [600,300,150,75,30,15,5]:
        for pid in ['P2','P4','P1','P3']:
            anchor=np.asarray(pts[pid]['xyz']);O=cm.XYZ['O01'];direction=(O[:2]-anchor[:2]);direction/=np.linalg.norm(direction)
            offsets=[direction*step]+[step*np.array([np.cos(k*np.pi/4),np.sin(k*np.pi/4)]) for k in range(8)]
            for off in offsets:
                if time.time()-start>seconds:break
                xy=anchor[:2]+off;key=(pid,round(xy[0],4),round(xy[1],4))
                if key in seen:continue
                seen.add(key);ground=jw.TER.native_height(xy)
                if not np.isfinite(ground):continue
                # Compute strict flight height before choosing a feasible hover.
                try:gg=jw.path_geometry([*xy,ground])
                except (AssertionError,IndexError,ValueError):continue
                z=min(gg['cruise_m'],ground+300);xyz=np.array([*xy,z]);gg=jw.path_geometry(xyz)
                bh=ter.point_link(cm.GW,xyz,'backhaul')
                if not bh:history.append(dict(name=str(key),status='backhaul_uncertified'));continue
                pp=copy.deepcopy(pts);lon,lat=cm.INV.transform(*xy)
                pp[pid]=dict(id=pid,xyz=xyz.tolist(),lon=lon,lat=lat,ground_m=ground,hover_agl_m=z-ground,geometry=gg,backhaul_margin_db=bh['margin_db'])
                # Fast necessary screening on delivery endpoints before full paths.
                bad=False
                for n in {n for r in best['records'] if pid in r['needs'] for n in r['nodes']}:
                    if not ter.point_link(cm.GW,cm.XYZ[n],'direct') and not any(ter.point_link(np.asarray(p['xyz']),cm.XYZ[n],'access') for p in pp.values()):bad=True;break
                if bad:history.append(dict(name=str(key),status='delivery_endpoint_uncertified'));continue
                rr=[obligations(r,ter,pp,spacing=75) for r in best['records']]
                if any(r is None for r in rr):history.append(dict(name=str(key),status='path_uncertified'));continue
                ss,inf=solve(rr,pp,vis,alpha=0 if nominal else .03,common_delay=0 if nominal else 120,hard_buffer=0 if nominal else 300,horizon=best['joint_objective'][2]+.001,limit=4,second_limit=2,split_windows=False)
                history.append(dict(name=str(key),step_m=step,**inf))
                if ss and score(ss)<score(best):
                    best=ss;pts=pp;best['model']['extra_link_margin_db']=margin;dump(best,out/'best_solution.json');dump(pts,out/'points.json');print('SPATIAL',pid,step,score(best),xyz,flush=True)
                dump(history,out/'history.json')
            if time.time()-start>seconds:break
        if time.time()-start>seconds:break
    dump(best,out/'best_solution.json');dump(pts,out/'points.json');dump(history,out/'history.json')
    dump(dict(best=score(best),attempts=len(history),seconds=time.time()-start,nominal=nominal,global_optimum_proven=False),out/'summary.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='results/optimization/joint_fast.json');p.add_argument('--out',default='results/optimization/search/spatial');p.add_argument('--seconds',type=float,default=600);p.add_argument('--nominal',action='store_true');a=p.parse_args();run(a.source,a.out,a.seconds,a.nominal)
