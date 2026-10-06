"""Masked tail supervision without inventing complete-horizon rule labels."""
import torch
import torch.nn.functional as F
from traffic_agent import TrafficAgent
from expert_gate import masked_final_bce
from navsim.agents.drive_jepa_perception_based.ema_jqtf import build_future_target_camera_pair
from navsim.agents.drive_jepa_perception_based.ema_jqtf.loss_schedule import scheduled_loss_weight


def select_batch(value,indices,batch):
    if torch.is_tensor(value) and value.ndim and len(value)==batch:return value[indices]
    if isinstance(value,dict):return {k:select_batch(v,indices,batch) for k,v in value.items()}
    if isinstance(value,list):
        if len(value)==batch and (not value or not torch.is_tensor(value[0])):return [value[i] for i in indices.tolist()]
        return [select_batch(v,indices,batch) for v in value]
    return value


def masked_distances(proposals,target,mask):
    error=(proposals-target[:,None]).abs().sum(-1)
    return torch.where(mask[:,None],error,0.).sum(-1)/mask.sum(-1)[:,None].clamp_min(1)


def masked_trajectory_loss(agent,pred,target,mask):
    valid=mask.any(-1);zero=pred['proposals'].sum()*0
    if not valid.any():return zero
    total=zero
    for proposals in pred['proposal_list']:
        selected=proposals[valid]
        match=masked_distances(selected,target[valid],mask[valid]).amin(1).mean()
        # Apply the inherited diversity formula on observed points only as well.
        distances=(selected[:,:,None]-selected[:,None]).abs().sum(-1)
        observed=mask[valid]
        distances=torch.where(observed[:,None,None],distances,0.).sum(-1)/observed.sum(-1)[:,None,None]
        distances=distances+(distances==0)
        diversity=-distances.amin(1).amin(1).mean()
        total=agent._config.prev_weight*total+match+agent._config.inter_weight*diversity
    return total


def global_valid_mean(mean,count):
    """All ranks participate, including ranks with no valid target for a task."""
    mean=mean.float()
    if not torch.distributed.is_initialized():return mean
    total=mean*count
    global_sum=total.detach().clone();global_count=total.new_tensor(float(count))
    torch.distributed.all_reduce(global_count);torch.distributed.all_reduce(global_sum)
    grad=total*torch.distributed.get_world_size()/global_count.clamp_min(1)
    return grad+(global_sum/global_count.clamp_min(1)-grad.detach())


def future_weight(agent):
    config=agent._config
    return scheduled_loss_weight(getattr(config,'future_prediction_weight_schedule',()),
                                 getattr(agent,'_training_epoch',0),
                                 getattr(config,'future_prediction_weight',.1))


def normalize_tasks(result,ntraj,nfuture,weight=None):
    result['trajectory_loss']=global_valid_mean(result['trajectory_loss'],ntraj)
    # Keep disabled/absent auxiliary values on-device for DDP collectives.
    # A zero scheduled weight does not disable reconstruction in this recipe.
    if not torch.is_tensor(result['future_loss']):
        result['future_loss']=result['trajectory_loss'].new_tensor(result['future_loss'])
    result['future_loss']=global_valid_mean(result['future_loss'],nfuture)
    if weight is None:weight=float(result['scheduled_future_weight'])
    result['scheduled_future_weight']=result['future_loss'].detach().new_tensor(weight)
    result['weighted_future_loss']=weight*result['future_loss']
    result['loss']=result['trajectory_loss']+result['final_score_loss']+result['weighted_future_loss']
    return result


class TailTrafficAgent(TrafficAgent):
    def compute_loss(self,features,targets,pred):
        from collision_reuse import collision_scope
        with collision_scope() as state:
            result=self._compute_loss_with_aux(features,targets,pred)
            result['collision_reuse_hits']=pred['proposals'].new_tensor(state['hits'])
            result['collision_reuse_misses']=pred['proposals'].new_tensor(state['misses'])
            return result

    def _compute_loss_with_aux(self,features,targets,pred):
        result=self._compute_planning_loss(features,targets,pred)
        if 'collision_object_prediction' in pred:
            from collision_aux import auxiliary_loss
            ce,regression,known,positive=auxiliary_loss(pred['collision_object_prediction'],pred['proposals'],targets)
            result.update(collision_object_bce=ce,collision_object_l1=regression,
                          collision_object_known=known,collision_object_positive=positive)
            result['loss']=result['loss']+self._config.pred_ce_weight*ce+self._config.pred_l1_weight*regression
        return result

    def _compute_planning_loss(self,features,targets,pred):
        weight=future_weight(self)
        if 'rule_score_valid' not in targets or bool(targets['rule_score_valid'].all()):
            result=super().compute_loss(features,targets,pred)
            return normalize_tasks(result,len(targets['trajectory']),len(targets['trajectory']),weight)
        batch=len(targets['trajectory']);full=targets['rule_score_valid'].nonzero().flatten()
        tail=(~targets['rule_score_valid']).nonzero().flatten()
        zero=pred['pred_logit'].sum()*0+pred['proposals'].sum()*0
        full_result=None;attributes={}
        names=('expert_keep','expert_reasons','expert_override','expert_target','expert_target_valid','candidate_traffic_valid')
        if len(full):
            full_result=super().compute_loss(select_batch(features,full,batch),select_batch(targets,full,batch),select_batch(pred,full,batch))
            attributes={k:getattr(self,k).clone() for k in names}
        else:
            # The original BCE performs two collectives. Skipping it on an
            # all-tail rank would mismatch another rank's supervised BCE.
            logits=pred['pred_logit'][...,0]
            empty_score=masked_final_bce(logits,torch.zeros_like(logits),torch.zeros_like(logits,dtype=torch.bool))
        t=select_batch(targets,tail,batch);p=select_batch(pred,tail,batch)
        trajectory=masked_trajectory_loss(self,p,t['trajectory'],t['trajectory_valid'])
        count=int(t['trajectory_valid'].any(-1).sum());nfull=len(full)
        trajectory=(trajectory*count+(full_result['trajectory_loss']*nfull if nfull else zero))/max(count+nfull,1)
        future_indices=t['future_camera_valid'][:,0].nonzero().flatten()
        future=zero
        if len(future_indices):
            pp=select_batch(p,future_indices,len(tail));tt=select_batch(t,future_indices,len(tail))
            match=masked_distances(pp['proposals'],tt['trajectory'],tt['trajectory_valid']).argmin(1).detach()
            model=self._pad_model
            selected=model._future_predictor._gather_candidates(pp['candidate_future_tokens'],match)
            predicted=model._future_predictor.decode_full_latent(pp['future_context_latent'],selected)
            with torch.no_grad():
                frame=model.transform(model._camera_to_float(tt['future_camera_features'][:,0]))
                pair=build_future_target_camera_pair(pp['future_target_current_frame'],frame,self._config.future_target_pair_mode)
                target=model.encode_future_target(pair)
                if self._config.future_target_layernorm:
                    target=F.layer_norm(target.permute(0,2,3,1),(target.shape[1],)).permute(0,3,1,2)
            if self._config.future_target_layernorm:
                predicted=F.layer_norm(predicted.permute(0,2,3,1),(predicted.shape[1],)).permute(0,3,1,2)
            if self._config.future_prediction_type!='l1':raise ValueError('Tail recipe currently requires the configured L1 future loss')
            future=F.l1_loss(predicted,target)
        nf=len(future_indices)
        future=(future*nf+(full_result['future_loss']*nfull if nfull else zero))/max(nf+nfull,1)
        score=full_result['final_score_loss'] if nfull else empty_score
        result=dict(full_result or {})
        result.update(trajectory_loss=trajectory,final_score_loss=score,future_loss=future,
                      sub_score_loss=zero,scheduled_future_weight=zero.detach()+weight,
                      loss=trajectory+score+weight*future)
        # Tail rows never trigger the unsafe-expert distance fallback.
        device=pred['proposals'].device;proposals=pred['proposals'].shape[1]
        self.expert_keep=torch.zeros(batch,dtype=torch.bool,device=device)
        self.expert_reasons=torch.zeros(batch,7,dtype=torch.bool,device=device);self.expert_reasons[tail,-1]=True
        self.expert_override=torch.zeros_like(self.expert_keep)
        self.expert_target=torch.zeros(batch,1,device=device)
        self.expert_target_valid=torch.zeros_like(self.expert_keep)
        self.candidate_traffic_valid=torch.zeros(batch,proposals,dtype=torch.bool,device=device)
        for k,v in attributes.items():getattr(self,k)[full]=v
        result['expert_score_keep_fraction']=self.expert_keep.float().mean()
        result['expert_override_fraction']=self.expert_override.float().mean()
        result['tail_fraction']=zero.detach()+len(tail)/batch
        return normalize_tasks(result,count+nfull,nf+nfull,weight)

    def compute_score(self,targets,proposals,test=True):
        mask=targets.get('rule_score_valid')
        if mask is None or bool(mask.all()):return super().compute_score(targets,proposals,test)
        if not test:raise RuntimeError('Tail rows must be removed before rule loss')
        full=mask.nonzero().flatten();batch,count=proposals.shape[:2]
        # Rule labels are FP32 even when model proposals use BF16 autocast.
        scores=torch.full((batch,count),float('nan'),device=proposals.device,dtype=torch.float32)
        components=torch.full((batch,6),float('nan'),device=proposals.device,dtype=torch.float32)
        if len(full):
            result=super().compute_score(select_batch(targets,full,batch),proposals[full],test=True)
            scores[full]=result[2];components[full]=result[4]
        return scores[:,0].nanmean(),scores.amax(-1).nanmean(),scores,proposals.sum()*0,components
