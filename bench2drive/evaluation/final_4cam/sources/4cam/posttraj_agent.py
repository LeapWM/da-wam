import torch
from agent import DenseAgent
from posttraj_model import PostTrajModel
from posttraj_modules import PostTrajectoryFutureHead
from expert_gate import masked_final_bce
from corrected_score import expert_checks

class PostTrajAgent(DenseAgent):
    def __init__(self,config,lr,checkpoint_path=None,cache_data=False):
        if cache_data:raise ValueError('Use repaired dense dataset')
        assert config.ema_jqtf_scorer_head_mode=='direct_final'
        assert config.final_score_weight==1 and config.sub_score_weight==0
        assert not config.scorer_pdm_formula_inference
        super().__init__(config,lr);self._checkpoint_path=checkpoint_path
        model=self._pad_model
        old_hist=model.hist_encoding
        if old_hist.in_features != 7+config.command_num:
            raise ValueError(f'Unexpected inherited ego-status width: {old_hist.in_features}')
        # New B2D training consumes one scalar speed.  Remove the legacy
        # lateral-velocity column (index 4), which was identically zero.
        model.hist_encoding=torch.nn.Linear(
            old_hist.in_features-1,old_hist.out_features,bias=old_hist.bias is not None
        )
        with torch.no_grad():
            keep=[i for i in range(old_hist.in_features) if i!=4]
            model.hist_encoding.weight.copy_(old_hist.weight[:,keep])
            if old_hist.bias is not None:model.hist_encoding.bias.copy_(old_hist.bias)
        # Match the original NAVSIM PostTraj branch: predict and supervise
        # future features directly in the 256-D projected-map space.  The
        # learned scorer still receives the separate 1024-D V-JEPA context.
        model._future_predictor=PostTrajectoryFutureHead(
            dim=config.tf_d_model,
            num_poses=config.trajectory_sampling.num_poses,
            num_heads=config.tf_num_head,
            ffn_dim=config.tf_d_ffn,
            candidate_layers=config.ema_jqtf_candidate_layers,
            full_decoder_layers=config.ema_jqtf_full_decoder_layers,
            dropout=config.tf_dropout,
        )
        model.route_target_encoder=torch.nn.Sequential(
            torch.nn.Linear(2,config.tf_d_model),torch.nn.ReLU(),
            torch.nn.Linear(config.tf_d_model,config.tf_d_model),
        )
        from multiview import FrontGridFusion
        model.multiview_fusion=FrontGridFusion(
            config.tf_d_model, config.tf_num_head, config.tf_d_ffn,
            dropout=config.tf_dropout)
        model.__class__=PostTrajModel
        from collision_aux import CollisionObjectHead
        model.collision_object_head=CollisionObjectHead(config.tf_d_model,config.future_predictor_dim)
    def compute_loss(self,features,targets,pred):
        decisions=expert_checks(self,targets)
        self.expert_keep=torch.tensor([x[0] for x in decisions],device=pred['pred_logit'].device,dtype=torch.bool)
        self.expert_reasons=torch.tensor([x[1] for x in decisions],device=self.expert_keep.device,dtype=torch.bool)
        result=super().compute_loss(features,targets,pred)
        if (self._config.predict_future and not self._config.scorer_only_training
                and float(result['scheduled_future_weight'])==0.
                and 'future_camera_features' in targets):
            from future_reconstruction import complete_future_reconstruction
            result['future_loss']=complete_future_reconstruction(self,targets,pred)
            result['future_matched_distance']=pred['future_matched_distance'].mean()
            result['loss']=result['loss']+0.*result['future_loss']
        result['expert_score_keep_fraction']=self.expert_keep.float().mean()
        return result
    def score_loss(self,pred_logit,pred_logit2,agents_state,pred_area_logits,target_scores,gt_states,gt_valid,gt_ego_areas):
        assert pred_logit.shape[-1]==1 and pred_logit2 is None
        assert agents_state is None and pred_area_logits is None
        zero=pred_logit.sum()*0
        final=masked_final_bce(pred_logit[...,0],target_scores[...,-1],self.expert_keep)
        return zero,final,zero,zero,zero
    def get_optimizers(self,total_steps,warmup_steps=500,weight_decay=.001,min_lr=1e-6):
        # Native256 deliberately gives the generator representation and scorer
        # the same peak LR; only the low-weight future auxiliary head uses 2x.
        groups={'generator_representation':[],'scorer':[],'future_predictor':[]}
        for n,p in self._pad_model.named_parameters():
            if p.requires_grad:
                key='scorer' if n.startswith(('scorer.','collision_object_head.')) else 'future_predictor' if n.startswith('_future_predictor.') else 'generator_representation'
                groups[key].append((n,p))
        cfg=self._config
        scales={'generator_representation':1.,'scorer':cfg.ema_jqtf_formula_scorer_lr_scale,'future_predictor':cfg.ema_jqtf_formula_future_lr_scale}
        from optimization import grouped_adamw
        recipe=[{'named_params':ps,'lr':self._lr*scales[k],'name':k} for k,ps in groups.items()]
        return grouped_adamw(recipe,total_steps,warmup_steps,weight_decay,min_lr)
