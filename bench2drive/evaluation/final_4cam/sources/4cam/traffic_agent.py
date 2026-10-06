import torch
import numpy as np
from posttraj_agent import PostTrajAgent
from safety_labels import Labels
from expert_gate import masked_final_bce
from expert_distance import distance_targets
from corrected_score import score_batch,expert_base_scores
from expert_bundle import expert_rows,expert_scope

class TrafficAgent(PostTrajAgent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.traffic_labels=Labels(traffic_only=True)
        self.road_metric_worker=self.traffic_labels.road
        self._fast_final_only=True
    def compute_loss(self,features,targets,pred):
        # The actual recorded trajectory, not the nearest generated proposal.
        gt=targets['trajectory'][:,None].detach()
        model=self._pad_model;scorer=model.scorer
        scorer_context=pred['scorer_context_latent']
        pose,_=scorer._external_candidate_packet(gt,pred['projected_map'],scorer_context)
        future=model._future_predictor.predict_candidate_tokens(pred['future_context_latent'],gt,pose)
        self.expert_logit=scorer(gt,pose,pred['projected_map'],future,scorer_context)[0][...,0]
        with expert_scope(self):
            result=super().compute_loss(features,targets,pred)
        result['expert_override_fraction']=self.expert_override.float().mean()
        return result
    def compute_score(self,targets,proposals,test=True):
        original=score_batch(self,targets,proposals,test=False)
        base_known=self.base_score_valid.clone()
        array=proposals.detach().float().cpu().numpy()
        if test:
            labels=self.traffic_labels.score(array,targets['_safety'])
        else:
            # The confirmed fallback makes traffic scoring unnecessary for
            # samples whose expert already fails the original safety rules.
            needed=(~self.expert_reasons[:,:3].any(-1)).nonzero().flatten().tolist()
            labels={'aux_value':np.ones((*array.shape[:2],3),np.float32),'aux_valid':np.zeros((*array.shape[:2],3),bool)}
            if needed:
                partial=self.traffic_labels.score(array[needed],[targets['_safety'][i] for i in needed])
                for key in labels:labels[key][needed]=partial[key]
        values=torch.as_tensor(labels['aux_value'][...,1:3],device=proposals.device)
        component_known=torch.as_tensor(labels['aux_valid'][...,1:3],device=proposals.device)
        zero_known=(component_known&(values==0)).any(-1)|(base_known&(original[2][...,-1]==0))
        # Missing direction/signal evidence is omitted rather than predicted.
        # A known traffic failure still forces zero; otherwise a known base
        # rule score remains a valid partial target with unknown factors neutral.
        traffic_factor=torch.where(component_known,values,torch.ones_like(values)).prod(-1)
        valid=zero_known|base_known
        scores=original[2].clone();scores[...,-1]*=traffic_factor
        if test:
            final=torch.where(valid,scores[...,-1],torch.full_like(scores[...,-1],float('nan')))
            return final[:,0].mean(),final.amax(-1).mean(),final,(proposals[:,0]-targets['trajectory']).norm(dim=-1).mean(),scores[:,0]
        from expert_cache import cached_expert_rows
        def expert_traffic(one):
            return self.traffic_labels.score(one['trajectory'][:,None].detach().float().cpu().numpy(),one['_safety'])
        bundle=expert_rows(self,targets)
        rows=([r['traffic'] for r in bundle] if bundle is not None
              else cached_expert_rows(self,targets,'traffic',expert_traffic))
        expert=expert_traffic(targets) if rows is None else {k:np.concatenate([r[k] for r in rows],axis=0) for k in ('aux_value','aux_valid')}
        ev=torch.as_tensor(expert['aux_value'][:,0,1:3],device=proposals.device)
        expert_valid_components=torch.as_tensor(expert['aux_valid'][:,0,1:3],device=proposals.device)
        ek=expert_valid_components.all(-1)
        direction_fail=(ev[:,0]<1)&expert_valid_components[:,0]
        traffic_fail=(ev[:,1]<1)&expert_valid_components[:,1]
        unknown=~ek|~valid.any(-1)
        self.candidate_traffic_valid=valid
        self.expert_override=self.expert_reasons[:,:3].any(-1)|direction_fail|traffic_fail
        finite=torch.isfinite(targets['trajectory']).flatten(1).all(-1)&~self.expert_reasons[:,3]
        if '_scorer' in targets:
            gt_scores,expert_base_known=expert_base_scores(self,targets)
        else:
            gt_scores=score_batch(self,targets,targets['trajectory'][:,None],test=False)[2][:,0,-1]
            expert_base_known=self.base_score_valid[:,0]
        unknown |= ~expert_base_known
        expert_factor=torch.where(expert_valid_components,ev,torch.ones_like(ev)).prod(-1)
        self.expert_target=torch.where(self.expert_override,torch.ones_like(gt_scores),gt_scores*expert_factor)[:,None]
        self.expert_target_valid=finite&(expert_base_known|self.expert_override)
        fallback,_=distance_targets(proposals,targets['trajectory'])
        apply_fallback=self.expert_override[:,None]&finite[:,None]
        scores[...,-1]=torch.where(apply_fallback,fallback,scores[...,-1])
        self.candidate_traffic_valid|=apply_fallback.expand_as(self.candidate_traffic_valid)
        # A failed expert no longer suppresses candidate supervision.
        self.expert_keep=self.candidate_traffic_valid.any(-1)|self.expert_target_valid
        self.expert_reasons=torch.cat((self.expert_reasons,direction_fail[:,None],traffic_fail[:,None],unknown[:,None]),-1)
        out=list(original);out[0]=scores[...,-1];out[1]=out[0].amax(-1);out[2]=scores
        return tuple(out)
    def score_loss(self,pred_logit,pred_logit2,agents_state,pred_area_logits,target_scores,gt_states,gt_valid,gt_ego_areas):
        assert pred_logit.shape[-1]==1 and pred_logit2 is None and agents_state is None and pred_area_logits is None
        zero=pred_logit.sum()*0
        keep=torch.cat((self.candidate_traffic_valid,self.expert_target_valid[:,None]),1)
        logits=torch.cat((pred_logit[...,0],self.expert_logit),1)
        targets=torch.cat((target_scores[...,-1],self.expert_target),1)
        return zero,masked_final_bce(logits,targets,keep),zero,zero,zero
