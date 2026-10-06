"""Original EMA-JQTF Formula model with batch-local measured geometry."""
import gzip,pickle
import torch
from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
from navsim.agents.drive_jepa_perception_based.score_module.compute_b2d_score import b2d_before_score,get_scores

class DenseAgent(DriveJEPAAgent):
    def __init__(self,config,lr):
        # None skips the legacy all-dataset scorer cache and Ray initialization.
        super().__init__(config,lr,checkpoint_path=None)
        self.bce_logit_loss=torch.nn.BCEWithLogitsLoss();self.b2d=True;self.ray=False
        with gzip.open(config.b2d_map_cache_path,'rb') as f:self.b2d_map_infos=pickle.load(f)
        self.b2d_before_score=b2d_before_score;self.get_scores=get_scores
    def compute_score(self,targets,proposals,test=True):
        local={sample['token']:sample for sample in targets['_scorer']}
        self.train_metric_cache_paths=local;self.test_metric_cache_paths=local
        return super().compute_score(targets,proposals,test)
    def trajectory_loss_anchors(self,proposal_list,target_trajectory,config,scores_index):
        # Legacy zip(proposal_list, scores_index) truncated stages at small batch
        # sizes. B2D has no anchor pseudo labels: supervise every stage with GT.
        if any(len(x) for x in scores_index):raise RuntimeError('Unexpected trajectory teacher/anchor targets')
        total=0;mins=[];divs=[]
        for proposals in proposal_list:
            matched=(proposals-target_trajectory[:,None]).abs().sum(-1).mean(-1).amin(1).mean()
            diversity=self.diversity_loss(proposals)
            total=config.prev_weight*total+matched+config.inter_weight*diversity
            mins.append(matched);divs.append(diversity)
        return total,mins[-1],divs[-1],mins,divs
