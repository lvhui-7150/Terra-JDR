# 对应问题与章节：问题四；第7节
# 功能：原子组合并、有效分区穷举、区间着色及专用资源编号
# 重点函数或流程：labelings、color、run
# 数据路径相对本包根目录；时刻和时长用秒，距离用米，能量用kWh。
# 运行方法及输入输出见同目录《代码与图表索引.md》；默认复核不搜索新方案。

"""Exhaustive frozen-plan task partition and guarded interval coloring."""
from collections import defaultdict
from pathlib import Path
import json,csv,argparse,math
import transport as tr
from schedule_model import dump
ROOT=tr.ROOT;STOCK=[4,2,2,6,4,4,2,6];KEYS=['A_air','B_air','C_air','A_battery','B_battery','C_battery','relay_air','relay_component']
# 枚举非空无标签分区的规范标号，避免仅因组号互换而重复计数。
def labelings(n,k):
    def rec(a):
        if len(a)==n:
            if max(a)+1==k:yield a
            return
        for v in range(min(k-1,max(a)+1)+1):yield from rec(a+[v])
    yield from rec([0])
# 按开始时刻贪心复用已释放设备；区间图所需颜色数等于最大重叠数。
def color(rows,prefix):
    ends=[];out=[]
    for s,e,sid in sorted(rows):
        free=[i for i,t in enumerate(ends) if t<=s+1e-6]
        if free:i=min(free);ends[i]=e
        else:i=len(ends);ends.append(e)
        out.append(dict(resource=prefix+f'{i+1:02d}',sortie=sid,start_s=s,end_s=e))
    return len(ends),out
def run(path,audit,output=True,beta=.2):
    sol=json.loads(Path(path).read_text(encoding='utf-8'));records=sol['records'];relays=sol['relay_records']
    cert=list(csv.DictReader((Path(audit)/'communication.csv').open(encoding='utf-8-sig')))
    rs={r['sortie']:r for r in records};support={r['id']:{c['sortie'] for c in cert if c['provider']==r['id']} for r in relays}
    nodes=sorted({b['node'] for b in tr.BOX.values()});parent={s:s for s in nodes}
    def root(x):
        while x!=parent[x]:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def merge(ss):
        ss=list(ss)
        for s in ss[1:]:parent[root(s)]=root(ss[0])
    for r in records:merge(r['nodes'])
    for rid,ss in support.items():merge(s for sid in ss for s in rs[sid]['nodes'])
    at=defaultdict(list)
    for s in nodes:at[root(s)].append(s)
    atoms=sorted(at.values());alpha=sol['model']['energy_increase_guard'];allrows=[]
    def group(ss,idx,guard):
        ss=set(ss);rr=[r for r in records if r['nodes'][0] in ss];ids={r['sortie'] for r in rr};relay=[r for r in relays if support[r['id']]&ids]
        assert all(set(r['nodes'])<=ss for r in rr)
        assert all(support[r['id']]<=ids for r in relay)
        ints=defaultdict(list)
        for r in rr:
            g=r['type'];gi='ABC'.index(g);end=r['return_s'];soc=1-(1+guard)*r['energy_kwh']/tr.TYPES[g]['energy_kwh'];ready=end+tr.charge(soc,tr.CHG[g])
            ints[gi].append((r['start_s'],end,r['sortie']));ints[gi+3].append((r['start_s'],ready,r['sortie']))
        for r in relay:
            end=r['return_s'];soc=1-(1+guard)*r['energy_kwh']/3.2;ready=end+tr.charge(soc,1800)
            ints[6].append((r['start_s'],end+300,r['id']));ints[7].append((r['start_s'],ready,r['id']))
        counts=[];assign=[]
        for k in range(8):
            n,ar=color(ints[k],f'G{idx}_{KEYS[k]}_');counts.append(n);assign.extend(dict(group=idx,resource_type=KEYS[k],**a) for a in ar)
        return dict(nodes=sorted(ss),needs=counts,work_s=sum(r['return_s']-r['start_s'] for r in rr+relay),transport_work_s=sum(r['return_s']-r['start_s'] for r in rr),boxes=sum(len(r['boxes']) for r in rr),mass_kg=sum(r['mass_kg'] for r in rr),assignments=assign)
    base=group(nodes,0,alpha)['needs'];chosen=[]
    for K in [2,3]:
        for labels in labelings(len(atoms),K):
            gs=[group([s for a,l in zip(atoms,labels) if l==k for s in a],k+1,alpha) for k in range(K)]
            totals=[sum(g['needs'][j] for g in gs) for j in range(8)];w=[g['work_s'] for g in gs];wm=sum(w)/K
            d=max(abs(v/wm-1) for v in w);cv=math.sqrt(sum((v/wm-1)**2 for v in w)/K)
            gap=[max(0,t-s) for t,s in zip(totals,STOCK)]
            allrows.append(dict(K=K,groups=[g['nodes'] for g in gs],work_s=w,boxes=[g['boxes'] for g in gs],mass_kg=[g['mass_kg'] for g in gs],max_relative_work_deviation=d,CV=cv,needs=totals,gap=gap,redundancy=[t-b for t,b in zip(totals,base)],normalized_deficit=sum(g/s for g,s in zip(gap,STOCK)),group_details=gs))
        candidates=[r for r in allrows if r['K']==K];eligible=[r for r in candidates if r['max_relative_work_deviation']<=beta+1e-9]
        select=min(eligible,key=lambda r:(r['normalized_deficit'],sum(n/s for n,s in zip(r['needs'],STOCK)),r['CV'])) if eligible else min(candidates,key=lambda r:(r['max_relative_work_deviation'],r['normalized_deficit'])) if candidates else None
        chosen.append(dict(K=K,partitions=len(candidates),eligible_count=len(eligible),minimum_deviation=min((r['max_relative_work_deviation'] for r in candidates),default=None),selected=select))
    result=dict(atoms=atoms,stock=STOCK,resource_order=KEYS,unpartitioned_guarded_need=base,energy_guard=alpha,balance_beta=beta,definition='Prep-to-return operation including relays; guarded resource intervals include charging and relay turnaround; frozen certified responsibilities with no mission copying.',chosen=chosen)
    if output:
        target=Path(audit);dump(result,target/'partition.json');dump(allrows,target/'all_partitions.json')
        for c in chosen:
            if c['selected']:
                rows=[dict(K=c['K'],**a) for g in c['selected']['group_details'] for a in g['assignments']]
                tr.write_csv(target/f"dedicated_resources_{c['K']}.csv",rows)
    for c in chosen:print(dict(K=c['K'],atoms=len(atoms),partitions=c['partitions'],minimum_deviation=c['minimum_deviation'],eligible=c['eligible_count'],selected_gap=c['selected']['gap'] if c['selected'] else None),flush=True)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--audit',type=Path,required=True);a=p.parse_args();run(a.source,a.audit)
