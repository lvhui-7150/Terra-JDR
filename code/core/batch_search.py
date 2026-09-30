# 对应问题与章节：问题三；第5.3节
# 功能：固定参考位置下的货箱重分批邻域搜索
# 重点函数或流程：run
# 数据路径相对本包根目录；时刻和时长用秒，距离用米，能量用kWh。
# 运行方法及输入输出见同目录《代码与图表索引.md》；默认复核不搜索新方案。

"""Exhaustive small destroy/repair cargo pools plus continuous schedule MILP."""
from joint_search import points_from_solution,score
from joint_schedule_model import *
import itertools,argparse

def run(source,out,seconds=600,nominal=False,wide=False):
    out=ROOT/out;out.mkdir(parents=True,exist_ok=True);best=json.loads((ROOT/source).read_text(encoding='utf-8'));ter,pts=points_from_solution(best)
    if nominal:ter=MarginTerrain(jw.TER,0)
    alpha=0 if nominal else .03;begin=time.time();history=[];obcache={}
    def certify(r):
        key=(r['type'],tuple(r['boxes']),tuple(r['nodes']))
        if key not in obcache:obcache[key]=obligations(r,ter,pts,spacing=100)
        return obcache[key]
    if nominal:
        rr=[certify(r) for r in best['records']]
        ss,inf=solve(rr,pts,[r['point_id'] for r in best['relay_records']],alpha=0,hard_buffer=0,common_delay=0,limit=40,second_limit=10,horizon=best['joint_objective'][2]+.001)
        if ss:best=ss;best['model']['extra_link_margin_db']=0.;dump(best,out/'best_solution.json')
        history.append(dict(name='nominal_initial',**inf));print('NOMINAL',score(best),flush=True)
    for epoch in range(3):
        snap=copy.deepcopy(best);rr=snap['records'];p2=[i for i,r in enumerate(rr) if 'S008' in r['nodes']];ctasks=[i for i,r in enumerate(rr) if r['type']=='C']
        pools=[sorted(set(p2+[i])) for i in ctasks]+[sorted(set(p2+[i])) for i,r in enumerate(rr) if 'P2' in r['needs'] and i not in p2]
        if wide:
            allp2=[i for i,r in enumerate(rr) if 'P2' in r['needs']]
            pools=[allp2]+[sorted(set(allp2+[i])) for i,r in enumerate(rr) if i not in allp2 and r['type'] in 'AB']
        # Also redistribute three west/north/east routes where capacity is tight.
        for pid in ['P3','P4','P1']:
            ix=[i for i,r in enumerate(rr) if pid in r['needs']]
            pools+=list(map(list,itertools.combinations(ix,3)))
        for ni,ix in enumerate(pools):
            if time.time()-begin>seconds:break
            bs=sorted(b for i in ix for b in rr[i]['boxes'])
            if len(bs)>16 or len(ix)<2:continue
            fixed=[r for i,r in enumerate(rr) if i not in ix];cols=[]
            for mask in range(1,1<<len(bs)):
                ids=tuple(b for j,b in enumerate(bs) if mask>>j&1)
                # Count and cargo are fully enumerated in this bounded repair pool.
                for g in 'ABC':
                    if sum(tr.BOX[b]['mass_kg'] for b in ids)>tr.TYPES[g]['capacity_kg'] or sum(tr.BOX[b]['volume_m3'] for b in ids)>tr.TYPES[g]['volume_m3']+1e-8:continue
                    r=tr.route(g,ids)
                    if not r or r['energy_kwh']*(1+alpha)>.8*tr.TYPES[g]['energy_kwh']+1e-8:continue
                    if any(dt>min(tr.HARD[b]-(0 if nominal else 300),tr.BOX[b]['expected_s']-(0 if nominal else 120)) for b,dt in r['deliveries'].items()):continue
                    cols.append(r)
            if not cols:continue
            m=Model();yy=[m.var(f'pattern_{j}',0,1,True) for j in range(len(cols))];C=m.var('workload_C',0,best['joint_objective'][2])
            for b in bs:m.row({yy[j]:1 for j,c in enumerate(cols) if b in c['boxes']},lo=1,hi=1)
            for g in 'ABC':m.row({C:-len(tr.FLEET[g]),**{yy[j]:c['duration_s'] for j,c in enumerate(cols) if c['type']==g}},hi=-sum(r['duration_s'] for r in fixed if r['type']==g))
            cost=np.zeros(len(m.lo));cost[C]=1
            for j,c in enumerate(cols):cost[yy[j]]=.00001*c['duration_s']+.000001*c['energy_kwh']
            successes=0
            for attempt in range(80 if wide else 25):
                if time.time()-begin>seconds:break
                res=m.solve(cost,5)
                if res.x is None:break
                selected=[j for j,v in enumerate(yy) if res.x[v]>.5];m.row({yy[j]:1 for j in selected},hi=len(selected)-1)
                rec=fixed+[certify(cols[j]) for j in selected]
                if any(r is None for r in rec):continue
                ss,inf=solve(rec,pts,[r['point_id'] for r in best['relay_records']],alpha=alpha,hard_buffer=0 if nominal else 300,common_delay=0 if nominal else 120,horizon=best['joint_objective'][2]+.001,limit=4,second_limit=2,split_windows=False)
                history.append(dict(name=f'epoch{epoch}_pool{ni}_cover{attempt}',pool=ix,pool_boxes=len(bs),columns=len(cols),workload_bound=float(res.fun),**inf))
                dump(history,out/'history.json')
                if ss and score(ss)<score(best):
                    best=ss;best['model']['extra_link_margin_db']=0 if nominal else .3;dump(best,out/'best_solution.json');print('REBATCH',epoch,ni,attempt,score(best),flush=True);successes+=1
            print('POOL',epoch,ni,'size',len(bs),'patterns',len(cols),'best',score(best),'elapsed',time.time()-begin,flush=True)
            if successes and score(best)[0]<score(snap)[0]-10:break
        if score(best)==score(snap) or time.time()-begin>seconds:break
    dump(best,out/'best_solution.json');dump(dict(best=score(best),attempts=len(history),elapsed_s=time.time()-begin,nominal=nominal,global_optimum_proven=False),out/'summary.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='results/optimization/joint_comprehensive.json');p.add_argument('--out',default='results/optimization/search/rebatch');p.add_argument('--seconds',type=float,default=600);p.add_argument('--nominal',action='store_true');p.add_argument('--wide',action='store_true');a=p.parse_args();run(a.source,a.out,a.seconds,a.nominal,a.wide)
