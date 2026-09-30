# 对应问题与章节：问题二；第4节
# 功能：固定路线按机型分解并精修开始时刻，保存子模型日志
# 重点函数或流程：脚本顺序执行；不改变货箱路线
# 数据路径相对本包根目录；时刻和时长用秒，距离用米，能量用kWh。
# 运行方法及输入输出见同目录《代码与图表索引.md》；默认复核不搜索新方案。

"""Q2 fixed-route scheduling decomposes exactly by transport type.
With communication absent, no aircraft/battery resource is shared across types.
First minimize each type's C, then W for each under their common feasible C cap.
"""
from joint_schedule_model import *
from joint_search import score

out=ROOT/'results/optimization/search/q2_decomposed';out.mkdir(parents=True,exist_ok=True)
base=json.loads((ROOT/'results/optimization/transport_time.json').read_text(encoding='utf-8'));parts={};logs={}
for g in 'ABC':
    rec=[r for r in base['records'] if r['type']==g]
    ss,inf=solve(rec,{},[],alpha=0,hard_buffer=0,common_delay=0,horizon=base['joint_objective'][2]+.001,limit=90,second_limit=10)
    assert ss is not None
    parts[g]=ss;logs[g]=dict(stage1=inf);print('TYPE_C',g,score(ss),inf.get('lower_bound'),flush=True)
    dump(ss,out/f'{g}_stage1.json')
cap=max(s['joint_objective'][2] for s in parts.values())+.0001
for g in 'ABC':
    ss,inf=solve(parts[g]['records'],{},[],alpha=0,hard_buffer=0,common_delay=0,horizon=cap,objective='mean',limit=90,second_limit=10)
    if ss:parts[g]=ss
    logs[g]['stage2']=inf;print('TYPE_W',g,score(parts[g]),flush=True)
    dump(parts[g],out/f'{g}_stage2.json')
rec=[r for g in 'ABC' for r in parts[g]['records']]
for i,r in enumerate(rec,1):r['sortie']=f'T{i:03d}'
s=copy.deepcopy(base);s['records']=rec;C=max(r['return_s'] for r in rec);ee=sum(r['energy_kwh'] for r in rec);ws=sum((r['start_s']+dt)*tr.BOX[b]['priority'] for r in rec for b,dt in r['deliveries'].items())
s.update(joint_objective=[0.,0.,C,ee,len(rec),0],objective=[0.,0.,C,ee,len(rec)],weighted_delivery_s=ws,weighted_mean_delivery_s=ws/sum(b['priority'] for b in tr.BOX.values()),last_delivery_s=max(r['start_s']+dt for r in rec for dt in r['deliveries'].values()),optimization=dict(type_decomposition=logs,scope='Only fixed routes: type independence is exact in Q2; overall routing/grouping optimum not certified.'))
dump(s,out/'best_solution.json');dump(logs,out/'logs.json');print('DECOMPOSED',score(s),flush=True)
