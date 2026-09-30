# Code Index

| Component | Main files |
|---|---|
| Terrain geometry and safe payload | `code/core/single_site_model.py` |
| Segment flight time and energy | `code/core/transport.py` |
| Candidate sortie generation | `code/core/transport.py`, `code/core/transport_decomposition.py` |
| Transport and battery scheduling | `code/core/schedule_model.py` |
| Joint delivery--relay scheduling | `code/core/joint_schedule_model.py`, `code/core/joint_search.py` |
| Communication and LOS audits | `code/core/communication.py`, `code/core/robust_comm.py` |
| Atomic task blocks and resource peaks | `code/core/partition_integrated.py` |
| Route-workload lower bound | `code/core/makespan_global_bound.py`, `code/core/verify_bound.py` |
| Independent replay | `code/core/replay.py` |
| Problem-specific variants | `code/variants/` |
| Independent verification | `code/verification/` |
| Figure generation | `code/visualization/make_figures.py` |
