import argparse,hashlib,json,os
from pathlib import Path
import hydra,numpy as np,torch,pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from torch.utils.data import DataLoader
from multiview_data import DenseDataset,collate,expand_index
from camera_contract import CAMERA_ORDER,FUTURE_CAMERA
from data import ValidationSampler
from tail_agent import TailTrafficAgent as DenseAgent
from prepare import prepare
from evaluation_metrics import fixed_horizon_sums

class DelayedValidation(pl.Callback):
    """Skip full validation before ``start_epoch``, then validate every epoch."""
    def __init__(self,start_epoch,total_epochs):
        super().__init__();self.start_epoch=int(start_epoch);self.skip_interval=int(total_epochs)+1
        if self.start_epoch<0:raise ValueError('validation start epoch must be non-negative')
    def on_train_epoch_start(self,trainer,pl_module):
        trainer.check_val_every_n_epoch=1 if trainer.current_epoch>=self.start_epoch else self.skip_interval
        if trainer.is_global_zero:
            action='enabled' if trainer.current_epoch>=self.start_epoch else 'skipped'
            print(f'VALIDATION_SCHEDULE epoch={trainer.current_epoch} validation={action} start_epoch={self.start_epoch}',flush=True)

class Module(pl.LightningModule):
    TRAIN_EPOCH_KEYS=('loss','trajectory_loss','final_score_loss','future_loss','weighted_future_loss',
                      'scheduled_future_weight',
                      'collision_object_bce','collision_object_l1',
                      'expert_score_keep_fraction','expert_override_fraction','tail_fraction')
    def __init__(self,config,lr):
        super().__init__();self.agent=DenseAgent(config,lr)
    def transfer_batch_to_device(self,batch,device,dataloader_idx):
        features,targets=batch;geometry={k:targets[k] for k in ('_scorer','_safety')}
        features,targets=super().transfer_batch_to_device((features,{k:v for k,v in targets.items() if k not in geometry}),device,dataloader_idx)
        targets.update(geometry);return features,targets
    def on_train_epoch_start(self):
        self.gate_counts=torch.zeros(9,device=self.device,dtype=torch.float64)
        # Per-key sum and contributing sample count. Some auxiliary metrics do
        # not exist on every batch, so each key needs its own denominator.
        self.train_epoch_stats=torch.zeros(len(self.TRAIN_EPOCH_KEYS),2,device=self.device,dtype=torch.float64)
    def on_train_epoch_end(self):
        counts=self.gate_counts.clone()
        if torch.distributed.is_initialized():torch.distributed.all_reduce(counts)
        stats=self.train_epoch_stats.clone()
        if torch.distributed.is_initialized():torch.distributed.all_reduce(stats)
        if self.trainer.is_global_zero:
            labels=['samples','kept','collision','road_or_route','ttc','nonfinite','direction','traffic_control','unknown_traffic_label']
            report=dict(zip(labels,counts.tolist()));report['epoch']=int(self.current_epoch)
            report['filtered_fraction']=1-float(counts[1]/counts[0].clamp_min(1))
            (Path(self.trainer.default_root_dir)/f'expert_gate_epoch{self.current_epoch:02d}.json').write_text(json.dumps(report,indent=2))
            train={key:float(stats[i,0]/stats[i,1].clamp_min(1)) if stats[i,1]>0 else None
                   for i,key in enumerate(self.TRAIN_EPOCH_KEYS)}
            lrs={}
            for group in self.optimizers().param_groups:
                name=group.get('source_group','group')
                value=float(group['lr'])
                if name in lrs and abs(lrs[name]-value)>1e-15:raise RuntimeError(f'Inconsistent LR for {name}')
                lrs[name]=value
            payload={'stage':'train','epoch':int(self.current_epoch),'global_step':int(self.global_step),
                     'metrics':train,'learning_rates':lrs,'expert_gate':report}
            self._append_epoch_metrics(payload)
            print('TRAIN_EPOCH_METRICS '+json.dumps(payload,sort_keys=True),flush=True)
    def training_step(self,batch,index):
        f,t=batch;self.agent.set_training_epoch(int(self.current_epoch))
        self.trajectory_supervision_active=bool(t['trajectory_valid'].any())
        self.future_supervision_active=bool(t['future_camera_valid'].any())
        pred=self.agent(f)
        assert pred['pred_logit'].shape[-1]==1
        result=self.agent.compute_loss(f,t,pred)
        self.gate_counts+=torch.cat((torch.tensor([len(t['token'])],device=self.device),self.agent.expert_keep.sum()[None],self.agent.expert_reasons.sum(0))).double()
        keep_count=self.agent.expert_keep.sum().detach().clone()
        if torch.distributed.is_initialized():torch.distributed.all_reduce(keep_count)
        self.score_supervision_active=bool(keep_count>0)
        if self.global_step==0:
            expected=result['trajectory_loss']+result['final_score_loss']+result['scheduled_future_weight']*result['future_loss']
            expected+=self.agent._config.pred_ce_weight*result.get('collision_object_bce',0)+self.agent._config.pred_l1_weight*result.get('collision_object_l1',0)
            torch.testing.assert_close(result['loss'],expected)
            assert float(result['sub_score_loss'])==0
            if self.trainer.is_global_zero:
                evidence={k:float(result[k].detach()) for k in ('loss','trajectory_loss','final_score_loss','sub_score_loss','future_loss','scheduled_future_weight')}
                evidence['weights']={'trajectory':1.,'final_score':1.,'future':.1,'subscore':0.}
                evidence['collision_aux']={k:float(result[k].detach()) for k in ('collision_object_bce','collision_object_l1','collision_object_known','collision_object_positive') if k in result}
                evidence['collision_reuse']={k:float(result[k].detach()) for k in ('collision_reuse_hits','collision_reuse_misses') if k in result}
                evidence['weights'].update(collision_object_bce=self.agent._config.pred_ce_weight,collision_object_l1=self.agent._config.pred_l1_weight)
                evidence['expert_keep']=self.agent.expert_keep.tolist()
                evidence['expert_reasons']=self.agent.expert_reasons.tolist()
                evidence['expert_override']=self.agent.expert_override.tolist()
                evidence['expert_target']=self.agent.expert_target.tolist()
                evidence['expert_target_valid']=self.agent.expert_target_valid.tolist()
                evidence['candidate_rule_valid_count']=self.agent.candidate_traffic_valid.sum(-1).tolist()
                evidence['trajectory_valid']=t['trajectory_valid'].tolist()
                evidence['future_camera_valid']=t['future_camera_valid'].tolist()
                evidence['rule_score_valid']=t['rule_score_valid'].tolist()
                (Path(self.trainer.default_root_dir)/'loss_audit.json').write_text(json.dumps(evidence,indent=2))
        if not torch.isfinite(result['loss']):raise FloatingPointError(str(result))
        self.log_dict({'train/'+k:(v if v.is_floating_point() else v.float())
                       for k,v in result.items() if isinstance(v,torch.Tensor) and v.numel()==1},
                      sync_dist=False,batch_size=len(t['token']))
        batch_size=len(t['token'])
        for i,key in enumerate(self.TRAIN_EPOCH_KEYS):
            value=result.get(key)
            if isinstance(value,torch.Tensor) and value.numel()==1:
                self.train_epoch_stats[i,0]+=value.detach().double()*batch_size
                self.train_epoch_stats[i,1]+=batch_size
        return result['loss']
    def _append_epoch_metrics(self,payload,root=None):
        root=Path(self.trainer.default_root_dir if root is None else root);path=root/'epoch_metrics.jsonl'
        with path.open('a') as stream:stream.write(json.dumps(payload,sort_keys=True)+'\n')
        folder=root/'epoch_metrics';folder.mkdir(exist_ok=True)
        (folder/f"{payload['stage']}_epoch{payload['epoch']:03d}.json").write_text(json.dumps(payload,indent=2,sort_keys=True))
    def on_after_backward(self):
        if self.global_step==0 and self.trainer.is_global_zero:
            groups={}
            for key in ('scorer','_future_predictor','refiner','lora','route_target_encoder','multiview_fusion'):
                ps=[p for n,p in self.agent.named_parameters() if (key in n if key!='lora' else 'linear_a_' in n or 'linear_b_' in n) and p.requires_grad]
                groups[key]={'parameters':sum(p.numel() for p in ps),'nonzero':sum(p.grad is not None and bool(p.grad.count_nonzero()) for p in ps),'finite':all(p.grad is None or bool(p.grad.isfinite().all()) for p in ps)}
            (Path(self.trainer.default_root_dir)/'gradient_audit.json').write_text(json.dumps(groups,indent=2))
            active={'multiview_fusion':self.trajectory_supervision_active or self.future_supervision_active or self.score_supervision_active,'scorer':self.score_supervision_active,'_future_predictor':self.future_supervision_active or self.score_supervision_active,
                    'refiner':self.trajectory_supervision_active or self.future_supervision_active,'lora':self.trajectory_supervision_active or self.future_supervision_active or self.score_supervision_active,
                    'route_target_encoder':self.trajectory_supervision_active or self.future_supervision_active or self.score_supervision_active}
            if any(not g['finite'] or (not g['nonzero'] and active[k]) for k,g in groups.items()):raise FloatingPointError(groups)
    def on_validation_epoch_start(self):
        self.stats=torch.zeros(7,dtype=torch.float64,device=self.device)
        self.fixed_stats=torch.zeros(6,dtype=torch.float64,device=self.device)
    def validation_step(self,batch,index):
        f,t=batch;p=self.agent(f);selected=p['trajectory']
        _,_,scores,_,_=self.agent.compute_score(t,selected[:,None],test=True)
        d=(selected[...,:2]-t['trajectory'][...,:2]).float().norm(dim=-1)
        w=t['sample_weight'];known=torch.isfinite(scores[:,0])
        mask=t['trajectory_valid'];ade_valid=mask.any(-1);fde_valid=mask[:,-1]
        self.fixed_stats+=fixed_horizon_sums(selected,t['trajectory'],mask,w)
        ade=torch.where(mask,d,0.).sum(-1)/mask.sum(-1).clamp_min(1)
        self.stats+=torch.stack([(torch.nan_to_num(scores[:,0])*w).sum(),(ade*w*ade_valid).sum(),
                                (d[:,-1]*w*fde_valid).sum(),w.sum(),(known*w).sum(),
                                (ade_valid*w).sum(),(fde_valid*w).sum()]).double()
    def on_validation_epoch_end(self):
        if torch.distributed.is_initialized():torch.distributed.all_reduce(self.stats)
        if torch.distributed.is_initialized():torch.distributed.all_reduce(self.fixed_stats)
        l2,n2,a3,f3,n3,ipad2=self.fixed_stats
        ade3=a3/n3.clamp_min(1);fde3=f3/n3.clamp_min(1)
        self.log_dict({'val/l2_xy_2s':l2/n2.clamp_min(1),'val/ade_xy_3s':ade3,
                       'val/fde_xy_3s':fde3,'val/ade_fde':ade3+.5*fde3},sync_dist=True)
        if self.trainer.is_global_zero:
            metrics=dict(planning_horizon_s=3.,planning_dt_s=.5,l2_xy_2s=float(l2/n2) if n2>0 else None,
                         count_2s=float(n2),ade_xy_3s=float(a3/n3) if n3>0 else None,
                         fde_xy_3s=float(f3/n3) if n3>0 else None,count_3s=float(n3),
                         ipad_code_pose_l2_2s=float(ipad2/n2) if n2>0 else None)
            (Path(self.trainer.default_root_dir)/'validation_fixed_horizon.json').write_text(json.dumps(metrics,indent=2))
        s,a,f,n,known,ade_n,fde_n=self.stats
        if known==0 and not self.allow_empty_rule_validation:
            raise RuntimeError('No known traffic scores in full validation; cannot select best checkpoint')
        # ``stats`` was manually all-reduced above, so every rank logs the
        # same global value. Lightning's distributed mean is therefore an
        # identity operation and prevents misleading epoch-level warnings.
        self.log_dict({'val/score_epoch':s/known.clamp_min(1),'val/score_coverage':known/n.clamp_min(1),'val/ade':a/ade_n.clamp_min(1),'val/fde':f/fde_n.clamp_min(1)},sync_dist=True)
        if self.trainer.is_global_zero:
            legacy=dict(count=float(n),score_count=float(known),score=float(s/known) if known>0 else None,ade_count=float(ade_n),fde_count=float(fde_n),ade=float(a/ade_n.clamp_min(1)) if ade_n>0 else None,fde=float(f/fde_n.clamp_min(1)) if fde_n>0 else None)
            (Path(self.trainer.default_root_dir)/'validation.json').write_text(json.dumps(legacy,indent=2))
            payload={'stage':'validation','epoch':int(self.current_epoch),'global_step':int(self.global_step),
                     'metrics':dict(legacy,l2_xy_2s=float(l2/n2) if n2>0 else None,
                                    ade_xy_3s=float(ade3) if n3>0 else None,
                                    fde_xy_3s=float(fde3) if n3>0 else None,
                                    ade_fde=float(ade3+.5*fde3) if n3>0 else None,
                                    score_coverage=float(known/n.clamp_min(1)))}
            self._append_epoch_metrics(payload)
            print('VAL_EPOCH_METRICS '+json.dumps(payload,sort_keys=True),flush=True)
    def configure_optimizers(self):
        return self.agent.get_optimizers(int(self.trainer.estimated_stepping_batches),
            self.hparams.warmup_steps,self.hparams.weight_decay,self.hparams.min_lr)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--smoke-tail',action='store_true');ap.add_argument('--devices',type=int,default=8);ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--prefetch-factor',type=int,default=4);ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--validation-start-epoch',type=int,default=10);ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--warmup-steps',type=int,default=500);ap.add_argument('--min-lr',type=float,default=1e-6);ap.add_argument('--weight-decay',type=float,default=.001);ap.add_argument('--gradient-clip-val',type=float,default=1.0);ap.add_argument('--smoke-train-token',default=None);args=ap.parse_args()
    if args.smoke_tail and not args.smoke:raise ValueError('--smoke-tail requires --smoke')
    if args.smoke_train_token and not args.smoke:raise ValueError('Diagnostic sample selection requires --smoke')
    root=Path(os.environ['B2D_NEW_CACHE']);runroot=Path(os.environ['JinnTrainResult'])
    if runroot.resolve().is_relative_to(Path('/dahuafs/userdata/2639639/Code')):
        runroot=Path(os.environ['B2D_DENSE_INDEX'])/'runs'/runroot.name
    out=runroot/os.environ.get('EXPERIMENT_NAME','PostTraj_TrafficScore_10Hz')
    out.mkdir(parents=True,exist_ok=True)
    if any((out/'checkpoints').glob('*.ckpt')):raise RuntimeError('Existing checkpoints; use a new experiment name')
    from audit_ready import verify_acceptance
    verify_acceptance(root)
    index=json.loads((Path(os.environ['B2D_DENSE_INDEX'])/'index.json').read_text())
    if index['contract']['source']!=json.loads((root/'DATA_READY.json').read_text()):raise RuntimeError('Stale dense index')
    index=expand_index(index)
    pl.seed_everything(20260913,workers=True)
    cfg=OmegaConf.load(Path(__file__).resolve().parents[1]/'agent_config.yaml');config=hydra.utils.instantiate(cfg.config)
    module=Module(config,args.lr);module.save_hyperparameters(dict(warmup_steps=args.warmup_steps,weight_decay=args.weight_decay,min_lr=args.min_lr));module.allow_empty_rule_validation=args.smoke
    train=DenseDataset(root,index,'train');val=DenseDataset(root,index,'val')
    if args.smoke:
        if args.smoke_train_token:
            clip,frame=args.smoke_train_token.rsplit('/',1)
            entry=next(e for e in train.entries if e[0]==clip and e[1]==int(frame))
            # Explicit diagnostic only; repeated sample is not performance evidence.
            train.entries=[entry]*max(args.batch_size*args.devices*2,2)
        elif args.smoke_tail:
            c=index['train'][0];last=c['frames'][-1];first=c['first'];clip=c['folder']
            cases=[(clip,first,first),(clip,last-25,first),(clip,last-5,first),(clip,last,first)]
            n=max(args.batch_size*args.devices*2,8);train.entries=(cases*((n+3)//4))[:n]
        else:train.entries=train.entries[:max(args.batch_size*args.devices*2,2)]
        if args.smoke_tail:
            c=index['val'][0];val.entries=[(c['folder'],i,c['first']) for i in (c['first'],c['frames'][-1]-25,c['frames'][-1]-5,c['frames'][-1])]
        else:val.entries=val.entries[:max(args.batch_size*args.devices+1,3)]
    if args.workers < 0 or args.prefetch_factor < 1:raise ValueError('Invalid DataLoader settings')
    loader_workers=dict(num_workers=args.workers)
    if args.workers:
        loader_workers.update(persistent_workers=True,prefetch_factor=args.prefetch_factor)
    dl=lambda ds,training:DataLoader(ds,batch_size=args.batch_size,shuffle=training,collate_fn=collate,pin_memory=True,sampler=None if training else ValidationSampler(ds,int(os.environ.get('LOCAL_RANK',0)),args.devices),**loader_workers)
    plan_ckpt=ModelCheckpoint(dirpath=out/'checkpoints',monitor='val/ade_fde',mode='min',save_top_k=1,save_last=False,filename='best-plan-{epoch:02d}',auto_insert_metric_name=False,save_on_train_epoch_end=False)
    score_ckpt=ModelCheckpoint(dirpath=out/'checkpoints',monitor='val/score_epoch',mode='max',save_top_k=1,save_last=False,filename='best-score-{epoch:02d}',auto_insert_metric_name=False,save_on_train_epoch_end=False)
    # Keep a recoverable last.ckpt even during the ten epochs with no validation.
    last_ckpt=ModelCheckpoint(dirpath=out/'checkpoints',save_top_k=0,save_last=True,save_on_train_epoch_end=True,every_n_epochs=1)
    validation_schedule=DelayedValidation(args.validation_start_epoch,args.epochs)
    trainer=pl.Trainer(inference_mode=False,accelerator='gpu',devices=args.devices,strategy='ddp_find_unused_parameters_true' if args.devices>1 else 'auto',precision='bf16-mixed',max_epochs=args.epochs,max_steps=2 if args.smoke else -1,limit_train_batches=2 if args.smoke else 1.,limit_val_batches=1.,num_sanity_val_steps=0,check_val_every_n_epoch=args.epochs+1,callbacks=[validation_schedule,plan_ckpt,score_ckpt,last_ckpt,LearningRateMonitor(logging_interval='step')]+module.agent.get_training_callbacks(),default_root_dir=out,gradient_clip_val=args.gradient_clip_val,enable_progress_bar=False)
    if trainer.is_global_zero:
        (out/'data_contract.json').write_text(json.dumps(index['contract'],indent=2))
        (out/'contract.json').write_text(json.dumps({'recipe':'PostTraj TrafficScore native NAVSIM 256D future space','smoke_train_token':args.smoke_train_token,'input_hz':10,'input_projection':'crop_depth_corrected_v1','future_steps':6,'future_dt':.5,'future_representation':'projected_map_256','scorer_context':'lora_vit_map_1024','cameras':4,'camera_order':list(CAMERA_ORDER),'future_camera':FUTURE_CAMERA,'future_target_offset_seconds':.5,'fusion':'front-grid cross-attention','ego_status':'pose_zero3 + scalar_speed + acceleration2 + command6','navigation_target':'recorded RoutePlanner near_xy in x-forward ego frame','train':len(train),'val':len(val),'batch_per_gpu':args.batch_size,'devices':args.devices,'lr':args.lr,'epochs':args.epochs,'precision':'bf16-mixed','validation_schedule':{'skip_epochs':list(range(args.validation_start_epoch)),'start_epoch':args.validation_start_epoch,'frequency_epochs':1},'future_loss_weight_schedule':list(config.future_prediction_weight_schedule),'future_reconstruction_at_zero_weight':True,'optimizer':{'name':'AdamW','weight_decay':args.weight_decay,'warmup_steps':args.warmup_steps,'warmup_start_ratio':1/3,'schedule':'cosine','min_lr':args.min_lr,'gradient_clip_val':args.gradient_clip_val,'group_lr_scales':{'generator_representation':1.,'scorer':config.ema_jqtf_formula_scorer_lr_scale,'future_predictor':config.ema_jqtf_formula_future_lr_scale}},'checkpoint_selection':{'best_plan':'min val/ade_fde','best_score':'max val/score_epoch','last':'every train epoch'},'dataloader':{'workers':args.workers,'persistent_workers':bool(args.workers),'prefetch_factor':args.prefetch_factor if args.workers else None,'pin_memory':True},'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}},indent=2))
        OmegaConf.save(cfg,out/'agent_config.yaml')
        OmegaConf.save(OmegaConf.create({'agent':OmegaConf.to_container(cfg,resolve=True)}),out/'hydra_config.yaml')
        bridge=json.loads(Path('/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/Drive-JEPA-V2-EMA-JQTF/bench2drive/configs/ema_jqtf_b2d_aligned_progress_e22.json').read_text())
        # Training examples are dense 10 Hz frames.  Make deployment cadence
        # explicit instead of inheriting the legacy 0.5 s (2 Hz) bridge value.
        bridge['bridge']['inference_interval_seconds']=.1
        bridge.setdefault('ego_status',{})['velocity_mode']='scalar_speed'
        bridge.setdefault('features',{})['target_point']=True
        bridge['controller'].update(
            lookahead_mode='speed_distance',
            lookahead_speed_scale=.5,
            lookahead_distance_offset=2.5,
            lookahead_distance_min=4.,
            lookahead_distance_max=8.,
            speed_estimator='mean_interpoint',
            speed_points=5,
        )
        bridge['features'].update(camera_order=list(CAMERA_ORDER),future_camera=FUTURE_CAMERA)
        bridge['agent_entrypoint']=str(Path(__file__).with_name('multiview_agent.py'))
        bridge['model_launcher']=str(Path(__file__).with_name('run_model.sh'))
        bridge['model'].update(name='PostTraj Native256 four-camera front-future 10Hz',hydra_config=str(out/'hydra_config.yaml'),checkpoint=str(out/'checkpoints/last.ckpt'))
        (out/'inference_last.json').write_text(json.dumps(bridge,indent=2))
    trainer.fit(module,dl(train,True),dl(val,False))
    if args.smoke:
        trainer.validate(module,dl(val,False));trainer.save_checkpoint(out/'smoke.ckpt')
        if trainer.is_global_zero:
            state=torch.load(out/'smoke.ckpt',map_location='cpu')['state_dict'];module.load_state_dict(state,strict=True)
            (out/'SMOKE_PASSED.json').write_text(json.dumps({'optimizer_steps':int(trainer.global_step),'strict_reload':True}))

if __name__=='__main__':main()
