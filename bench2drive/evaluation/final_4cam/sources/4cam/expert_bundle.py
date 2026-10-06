"""One immutable expert record and one lookup within a training loss call."""
import os
from contextlib import contextmanager
from expert_cache import cached_expert_rows


@contextmanager
def expert_scope(agent):
    previous=getattr(agent,'_expert_bundle_scope',None)
    agent._expert_bundle_scope={}
    try:yield
    finally:
        if previous is None:del agent._expert_bundle_scope
        else:agent._expert_bundle_scope=previous


def expert_rows(agent,targets):
    if ('_scorer' not in targets or not getattr(agent,'_fast_final_only',False)
            or os.environ.get('B2D_EXPERT_CACHE')=='off'
            or os.environ.get('B2D_EXPERT_BUNDLE')=='off'):
        return None
    scope=getattr(agent,'_expert_bundle_scope',None)
    if scope is not None and scope.get('targets') is targets:return scope['rows']

    def compute(one):
        from corrected_score import _expert_checks_uncached,score_batch
        checks=_expert_checks_uncached(agent,one)[0]
        result=score_batch(agent,one,one['trajectory'][:,None],test=False)
        base=(float(result[2][0,0,-1]),bool(agent.base_score_valid[0,0]))
        traffic=agent.traffic_labels.score(
            one['trajectory'][:,None].detach().float().cpu().numpy(),one['_safety'])
        return {'checks':checks,'base':base,
                'traffic':{k:traffic[k] for k in ('aux_value','aux_valid')}}

    rows=cached_expert_rows(agent,targets,'bundle_v1',compute)
    if scope is not None:scope.update(targets=targets,rows=rows)
    return rows
