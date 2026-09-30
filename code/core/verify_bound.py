# 对应问题与章节：问题二、三的下界；第8节
# 功能：重新求解各机型定价并核验对偶修复和映射
# 重点函数或流程：run
# 数据路径相对本包根目录；时刻和时长用秒，距离用米，能量用kWh。
# 运行方法及输入输出见同目录《代码与图表索引.md》；默认复核不搜索新方案。

"""Replay unrestricted pricing and dual repair, plus feasible-plan mappings.
Numerical MILP optimality is distinguished from formal exact arithmetic.
"""
from makespan_global_bound import *

def run(source,out,limit=90):
    cert=json.loads((ROOT/source).read_text(encoding='utf-8'));pi=np.array(cert['pi']);lam=np.array(cert['lambda_']);delta=cert['uniform_per_box_shift'];logs=[]
    assert len(pi)==J and np.min(lam)>=-1e-12 and sum(lam[i]*len(tr.FLEET[g]) for i,g in enumerate('ABC'))<=1+1e-12
    energy=cert.get('energy_outer_relaxation',False)
    for i,g in enumerate('ABC'):
        col,log=pricing(g,pi,lam[i],limit,energy);logs.append(log)
    repaired=min(delta,min(z['lower_reduced_cost'] for z in logs)-1e-5,0.)
    lb=float((pi+repaired)@DEM)-1e-5
    mapped=[]
    for name in ['results/optimization/transport_time.json','results/optimization/joint_nominal.json','results/optimization/joint_fast.json','results/optimization/joint_energy.json','results/optimization/joint_two_groups.json','results/optimization/joint_comprehensive.json']:
        f=ROOT/name
        if not f.exists():continue
        sol=json.loads(f.read_text(encoding='utf-8'));work=0.;minimum=1e20;worstenergy=-1e20
        for r in sol['records']:
            counts=np.array([sum(b['id'] in r['boxes'] for b in gg) for gg in GROUPS]);c=column(r['type'],counts,[NODES.index(n) for n in r['nodes']]);assert c['duration_s']<=r['duration_s']+1e-7
            if energy:assert c['energy_lower_kwh']<=r['energy_kwh']+1e-7
            value=lam['ABC'.index(r['type'])]*c['duration_s']-(pi+repaired)@counts;minimum=min(minimum,value);assert value>=-1e-6
            work+=lam['ABC'.index(r['type'])]*r['duration_s'];worstenergy=max(worstenergy,c['energy_lower_kwh']-r['energy_kwh'])
        assert lb<=work+1e-6 and work<=sol['joint_objective'][2]+1e-6
        mapped.append(dict(source=name,minimum_column_dual_slack=minimum,weighted_resource_work_s=work,feasible_makespan_s=sol['joint_objective'][2],max_energy_lower_minus_actual=worstenergy))
    report=dict(status='PASS_NUMERICAL_GLOBAL_PRICING_REPLAY',verified_bound_s=lb,original_bound_s=cert['bound_s'],pricing=logs,all_pricing_optimal=all(z['status']==0 for z in logs),mapping=mapped,scope='All cargo counts and metric-closure visiting orders covered by numerical global MILP pricing. Energy tangents optional outer relaxation. Does not certify whole-problem optimality or provide formal real-arithmetic branch certificates.')
    dump(report,ROOT/out);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--out',required=True);p.add_argument('--limit',type=float,default=90);a=p.parse_args();run(a.source,a.out,a.limit)
