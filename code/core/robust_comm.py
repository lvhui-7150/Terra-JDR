# 对应问题与章节：问题三及附加保障；第5—6节
# 功能：在通信模型上增加明确的额外损耗保障
# 重点函数或流程：MarginTerrain
# 数据路径相对本包根目录；时刻和时长用秒，距离用米，能量用kWh。
# 运行方法及输入输出见同目录《代码与图表索引.md》；默认复核不搜索新方案。

"""Additional-margin certificates on the unchanged, supplied trajectory model.
A failed certificate is NOT a demonstrated outage.  DEM uplift affects radio
occlusion only; it does not certify obstacle clearance of the flight path.
"""
from __future__ import annotations
import math, numpy as np
from communication import Terrain, FSPL_BASE, BUDGET

class MarginTerrain(Terrain):
    def __init__(self, base: Terrain, extra_db: float=0., dem_uplift_m: float=0., best_point_margin: bool=False):
        self.__dict__.update(base.__dict__)
        self.extra_db=float(extra_db)
        self.dem_uplift_m=float(dem_uplift_m)
        self.best_point_margin=bool(best_point_margin)
    def link_certificate(self,q,p0,p1,kind):
        q,p0,p1=map(np.asarray,(q,p0,p1))
        dist=max(float(np.linalg.norm(q-p0)),float(np.linalg.norm(q-p1)))
        fspl=FSPL_BASE+20*math.log10(max(dist/1000,1e-12))
        los_margin=BUDGET[kind]-fspl;obs_margin=los_margin-10.
        if los_margin<self.extra_db-1e-9:return None
        if obs_margin>=self.extra_db+1e-9 and not self.best_point_margin:
            return dict(mode='worst_case_obstruction',margin_db=obs_margin,distance_upper_m=dist,clearance_lower_m=None)
        clear=self.los_sweep(q,p0,p1)-self.dem_uplift_m
        if clear>.05:
            return dict(mode='LOS_swept_region',margin_db=los_margin,distance_upper_m=dist,clearance_lower_m=clear)
        if obs_margin>=self.extra_db+1e-9:
            return dict(mode='worst_case_obstruction',margin_db=obs_margin,distance_upper_m=dist,clearance_lower_m=None)
        return None
