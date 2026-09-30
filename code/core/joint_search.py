# 对应问题与章节：问题三；第4.5、5.3节
# 功能：从参考解启动联合路线、职责与资源调度搜索
# 重点函数或流程：points_from_solution、score、run
# 数据路径相对本包根目录；时刻和时长用秒，距离用米，能量用kWh。
# 运行方法及输入输出见同目录《代码与图表索引.md》；默认复核不搜索新方案。

"""Joint upper search: broadened duties, visits, grouping and positions.
Lexicographic zero tardiness, makespan, weighted delivery, energy is explicit.
Never labels a finite-candidate optimum as full-problem global optimality.
"""
from joint_schedule_model import *
import itertools,random,argparse

OUT=ROOT/'results/optimization/search';OUT.mkdir(parents=True,exist_ok=True)

def points_from_solution(sol):
    ter,pts=geometry()
    for r in sol['relay_records']:
        pts[r['point_id']]={k:copy.deepcopy(r[k]) for k in ['xyz','lon','lat','ground_m','hover_agl_m','geometry','backhaul_margin_db']}
        pts[r['point_id']]['id']=r['point_id']
    return ter,pts

def score(s):return (round(s['joint_objective'][2],3),round(s['weighted_mean_delivery_s'],3),s['joint_objective'][3])

def run(seconds=900):
    source=OUT/'best_solution.json'
    best=json.loads((source if source.exists() else ROOT/'results/optimization/joint_fast.json').read_text(encoding='utf-8'))
    ter,pts=points_from_solution(best);vis=[r['point_id'] for r in best['relay_records']];hist=[];rng=random.Random(20260925);begin=time.time();cache={}
    def attempt(label,recs,pp=None,vv=None,limit=6,second=2,split=True):
        nonlocal best,pts,vis
        pp=pp or pts;vv=vv or vis
        ss,inf=solve(recs,pp,vv,limit=limit,second_limit=second,horizon=best['joint_objective'][2]+.001,split_windows=split)
        hist.append(dict(name=label,**inf));dump(hist,OUT/'upper_history.json')
        if ss and score(ss)<score(best):
            best=ss;pts=copy.deepcopy(pp);vis=list(vv);dump(best,source);dump(pts,OUT/'points.json');print('IMPROVED',label,score(best),flush=True)
        return ss
    # Fair baseline: remove previous Pareto caps, then split noncontiguous duties.
    attempt('fixed_routes_free_secondary',best['records'],limit=60,second=20,split=False)
    attempt('split_communication_windows',best['records'],limit=60,second=20)
    for vv in [['P1','P2','P3','P4'],['P1','P2','P2','P3','P3','P4'],['P1','P2','P3','P3','P4','P4'],['P1','P1','P2','P2','P3','P3','P4']]:
        attempt('visits_'+'_'.join(vv),best['records'],vv=vv,limit=35,second=5)
    def make(g,bs,prior=None,order=()):
        key=(tuple((p,tuple(pts[p]['xyz'])) for p in sorted(pts)),g,tuple(sorted(bs)),prior,order)
        if key not in cache:
            rr=tr.route(g,tuple(sorted(bs)),order)
            cache[key]=None if not rr or rr['energy_kwh']*1.03>.8*tr.TYPES[g]['energy_kwh']+1e-8 else obligations(dict(rr,relay_point=prior),ter,pts,spacing=100)
        return cache[key]
    # Dynamic restarts from each improved incumbent, including cross-region
    # exchanges, changing BOTH types, and route permutation changes.
    epoch=0
    while time.time()-begin<seconds:
        snap=copy.deepcopy(best);cands=[]
        for i,a in enumerate(snap['records']):
            for g in 'ABC':
                if g!=a['type']:cands.append(('type',i,None,g,None,a['boxes'],None))
            for j,b in enumerate(snap['records']):
                if i>=j:continue
                transformations=[]
                # Main neighborhoods unrestricted by earlier relay-region labels.
                for x in a['boxes']:
                    if len(a['boxes'])>1:transformations.append(([z for z in a['boxes'] if z!=x],b['boxes']+[x]))
                for x in b['boxes']:
                    if len(b['boxes'])>1:transformations.append((a['boxes']+[x],[z for z in b['boxes'] if z!=x]))
                for x,y in itertools.product(a['boxes'],b['boxes']):
                    if (tr.BOX[x]['category'],tr.BOX[x]['node'])==(tr.BOX[y]['category'],tr.BOX[y]['node']):continue
                    transformations.append(([z for z in a['boxes'] if z!=x]+[y],[z for z in b['boxes'] if z!=y]+[x]))
                for aa,bb in transformations:
                    for ga,gb in itertools.product('ABC',repeat=2):
                        if any(sum(tr.BOX[z]['mass_kg'] for z in bs)>tr.TYPES[g]['capacity_kg'] or sum(tr.BOX[z]['volume_m3'] for z in bs)>tr.TYPES[g]['volume_m3']+1e-8 for g,bs in [(ga,aa),(gb,bb)]):continue
                        cands.append(('pair',i,j,ga,gb,aa,bb))
        rng.shuffle(cands)
        # Single-type options first; later critical-path exchanges followed by all.
        cands.sort(key=lambda c:(c[0]!='type',not(any(tr.BOX[b]['node']=='S008' for b in c[5]+(c[6] or [])))))
        trials=0
        for n,(kind,i,j,ga,gb,aa,bb) in enumerate(cands):
            if time.time()-begin>seconds:break
            a=snap['records'][i];prior=next(iter(a['needs']),None);ra=make(ga,aa,prior)
            if ra is None:continue
            rb=make(gb,bb,next(iter(snap['records'][j]['needs']),None)) if j is not None else None
            if j is not None and rb is None:continue
            rec=copy.deepcopy(snap['records']);rec[i]=ra
            if j is not None:rec[j]=rb
            attempt(f'epoch{epoch}_{kind}_{n}',rec,limit=2,second=1,split=False);trials+=1
            if score(best)[0]<score(snap)[0]-10 and trials>30:break
        print('EPOCH',epoch,'attempts',len(hist),'best',score(best),'seconds',time.time()-begin,flush=True);epoch+=1
        if not trials:break
    dump(best,source);dump(pts,OUT/'points.json')
    dump(dict(best=score(best),attempts=len(hist),elapsed_s=time.time()-begin,global_optimum_proven=False),OUT/'upper_summary.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=900);a=p.parse_args();run(a.seconds)
