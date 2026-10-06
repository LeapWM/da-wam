"""Time-aligned recorded-box safety. Unknown is distinct from a violation."""
import os
import warnings
import numpy as np
import torch
from shapely.geometry import Polygon,Point,LineString
from navsim.agents.drive_jepa_perception_based.score_module import compute_b2d_score as legacy
from navsim.agents.drive_jepa_perception_based.ema_jqtf.formula_selection import formula_progress_ratio

VERSION='aligned_recorded_safety_3s_ipad_comfort_static_2hz_v5'
warnings.filterwarnings('ignore', message='invalid value encountered in line_locate_point', category=RuntimeWarning)
from metric_geometry import footprints,expert_relative_comfort
from static_collision import static_collision,merge_collision
from types import SimpleNamespace

def aligned_safety(point,actor_valid,actor_z=None,ego_z=None,dt=.5,return_witness=False):
    from collision_reuse import cached_safety
    return cached_safety(_aligned_safety_uncached,point,actor_valid,actor_z,ego_z,dt,return_witness)

def _aligned_safety_uncached(point,actor_valid,actor_z=None,ego_z=None,dt=.5,return_witness=False):
    """Same GEOS predicates and unknown policy, batched over candidates/actors."""
    import shapely
    boxes=np.asarray(point['fut_box_corners']);ego=np.asarray(point['_ego_coords'])
    actors,times=boxes.shape[:2];count=len(ego)
    annotated=np.asarray(actor_valid,bool)
    finite=np.isfinite(boxes).all(axis=(-1,-2));present=annotated&finite
    centers=boxes.mean(-2)
    jumps=(np.linalg.norm(np.diff(centers,axis=1),axis=-1)>50.*dt+5.)&present[:,1:]&present[:,:-1]
    bad_edge=jumps|(present[:,1:]!=present[:,:-1])
    hit=np.zeros((count,2),bool);unknown=np.zeros((count,2),bool)
    witness=np.full((count,2),-1,dtype=np.int64)
    # Invalid/nonfinite shapes never reach intersects; retain their unknown flags.
    ego_finite=np.isfinite(ego).all(axis=(-1,-2))
    ep=shapely.polygons(np.where(np.isfinite(ego),ego,0.))
    ap=shapely.polygons(np.where(np.isfinite(boxes),boxes,0.))
    ev=ego_finite&shapely.is_valid(ep)&(shapely.area(ep)>0)
    av=finite&shapely.is_valid(ap)&(shapely.area(ap)>0)
    for t in range(times):
        for shift in (0,1,2):
            if shift and t>times-3:continue
            future=t+shift;channel=int(shift>0)
            unknown[:,channel]|=~ev[:,t,shift]
            discontinuity=bad_edge[:,t:future].any(-1) if shift else np.zeros(actors,bool)
            bad_actor=discontinuity|(annotated[:,future]&~finite[:,future])|(present[:,future]&~av[:,future])
            if bad_actor.any():unknown[:,channel]=True
            ai=np.flatnonzero(~discontinuity&present[:,future]&av[:,future])
            pi=np.flatnonzero(ev[:,t,shift])
            if not len(ai) or not len(pi):continue
            intersects=shapely.intersects(ep[pi,t,shift,None],ap[ai,future][None,:])
            if actor_z is not None and ego_z is not None:
                az=np.asarray(actor_z)[ai,future];ez=np.asarray(ego_z)[future]
                zknown=np.isfinite(az).all(-1)&np.isfinite(ez).all()
                unknown[pi,channel]|=(intersects&~zknown[None,:]).any(-1)
                separated=np.minimum(az[:,1],ez[1])<np.maximum(az[:,0],ez[0])
                intersects&=(zknown&~separated)[None,:]
            collided=intersects.any(-1)
            selected=collided&(witness[pi,channel]<0)
            witness[pi[selected],channel]=ai[intersects[selected].argmax(-1)]
            hit[pi,channel]|=collided
    result=((~hit).astype(np.float32),hit|~unknown,{'jump_edges':int(jumps.sum()),'missing_edges':int((present[:,1:]!=present[:,:-1]).sum())})
    return (*result,witness) if return_witness else result

def path_progress(trajectories,expert):
    points=np.vstack((np.zeros((1,2)),expert[:,:2]));line=LineString(points)
    # Projection gives progress along the reference, not Euclidean distance
    # sideways beyond its endpoint. Only forward tangent overshoot counts.
    delta=np.diff(points,axis=0);nonzero=np.linalg.norm(delta,axis=-1)>1e-6
    tangent=delta[nonzero][-1] if nonzero.any() else np.zeros(2)
    tangent=tangent/max(np.linalg.norm(tangent),1e-6)
    if (not np.isfinite(points).all() or line.is_empty or not line.is_valid
            or not np.isfinite(line.length) or line.length<=1e-6):
        return np.zeros(len(trajectories)),0.
    result=[]
    for trajectory in trajectories:
        end=trajectory[-1,:2]
        if not np.isfinite(end).all():
            result.append(float('nan'));continue
        point=Point(end)
        if point.is_empty or not point.is_valid:
            result.append(float('nan'));continue
        s=line.project(point)
        if line.length>1e-6 and abs(s-line.length)<1e-6:s+=max(float(np.dot(end-points[-1],tangent)),0.)
        result.append(s)
    return np.asarray(result),line.length

def prepare_geometry(point,prepared):
    ego=prepared['current_ego'];w2e=np.asarray(prepared['world2ego'])
    mirror=np.diag([1.,-1.,1.,1.])
    vertices=np.asarray(ego['world_cord'])
    local=np.c_[vertices,np.ones(8)]@(w2e@mirror).T
    offset=local[:,:2].mean(0)
    cfg=SimpleNamespace(half_length=float(np.ptp(local[:,0])/2),half_width=float(np.ptp(local[:,1])/2),rear_axle_to_center=offset[0],center_offset_y=offset[1])
    poses=np.concatenate([point['proposal'],point['target_traj'][None]])
    velocity=np.diff(poses[...,:2],axis=1,prepend=np.zeros((len(poses),1,2)))/.5
    projected=np.repeat(poses[:,:,None,:],3,axis=2)
    projected[...,:2]+=velocity[:,:,None,:]*np.array([0.,.5,1.])[None,None,:,None]
    point['_ego_coords']=footprints(projected,cfg)
    point['fut_box_corners']=prepared['scoring_box_corners']

def official_road(agent,point,prepared):
    from safety_labels import RoadWorker
    worker=getattr(agent,'road_metric_worker',None)
    if worker is None:
        worker=RoadWorker();agent.road_metric_worker=worker
    poses=np.concatenate([point['proposal'],point['target_traj'][None]])
    mirror=np.diag([1.,-1.,1.,1.]);w2e=np.asarray(prepared['world2ego'])
    future=prepared['future_egos'][4::5]
    raw=np.asarray([e['location'] for e in future])
    local=np.c_[raw,np.ones(len(raw))]@(w2e@mirror).T
    if not np.allclose(local[:,:2],point['target_traj'][:,:2],atol=.002,rtol=0):raise ValueError('Road label expert/cache mismatch')
    xyz=np.concatenate([poses[...,:2],np.broadcast_to(local[None,:,2:3],(*poses.shape[:2],1)),np.ones((*poses.shape[:2],1))],axis=-1)
    world=xyz@(mirror@np.linalg.inv(w2e)).T
    valid=np.isfinite(world).all(-1)
    r=worker.request({'town':prepared['town'],'road_only':True,'world_positions':np.nan_to_num(world[...,:3]).tolist(),'available':valid.tolist()})
    road=np.asarray(r['road'],bool);known=np.asarray(r['valid'],bool)
    fail=(known&~road).any(-1)
    return (~fail).astype(float),fail|known.all(-1)

def score_points(agent,targets,proposals):
    """Skip superseded legacy geometry only for the direct-final consumer.

    ego_areas is an unused, false placeholder in this path. The active road
    label and its known mask are still computed by official_road below.
    """
    local={s['token']:s for s in targets['_scorer']}
    if not getattr(agent,'_fast_final_only',False) or os.environ.get('B2D_MINIMAL_PACKET')=='off':
        return legacy.b2d_before_score(agent.b2d_map_infos,proposals.detach(),targets,local,agent._config)
    array=proposals.detach().float().cpu().numpy()
    experts=targets['trajectory'].detach().float().cpu().numpy()
    points=[]
    for token,proposal,expert in zip(targets['token'],array,experts):
        sample=local[str(token)]
        if proposal.shape[-2:]!=expert.shape[-2:]:raise ValueError('proposal/target trajectory mismatch')
        if sample['future_box_corners'].shape[1]!=proposal.shape[1]:raise ValueError('future-box horizon mismatch')
        points.append({'proposal':proposal,'target_traj':expert,
                       'ego_areas':np.zeros((len(proposal)+1,len(expert),3),bool)})
    return points

def score_batch(agent,targets,proposals,test=False):
    points=score_points(agent,targets,proposals)
    if getattr(agent,'_fast_final_only',False):
        # Direct-final loss asserts that agent/area auxiliary heads are absent.
        # Keep the return packet shape, with unused agent supervision masked out.
        raw=[]
        for point in points:
            count,times=point['proposal'].shape[:2]
            raw.append((np.zeros((count,6)),np.zeros((count,times,2,4,2)),
                        np.zeros((count,times,2),bool),point['ego_areas'][:-1,:,1:3],
                        None,None,np.empty(0,dtype=np.int64)))
    else:
        raw=legacy.get_scores(points)
    scored=[];valids=[];diagnostics=[]
    for p,sample,prepared,old in zip(points,targets['_scorer'],targets['_safety'],raw):
        prepare_geometry(p,prepared)
        road,road_known=official_road(agent,p,prepared)
        comfort,comfort_known=expert_relative_comfort(p['proposal'],p['target_traj'])
        values,valid,diag=aligned_safety(p,sample['future_box_valid'],prepared['actor_world_z'],prepared['ego_world_z'])
        static_value,static_valid,static_first=static_collision(np.concatenate([p['proposal'],p['target_traj'][None]]),prepared)
        values[:,0],valid[:,0]=merge_collision(values[:,0],valid[:,0],static_value,static_valid)
        diag.update(static_failed=int((static_valid[:-1]&(static_value[:-1]==0)).sum()),
                    static_known=int(static_valid[:-1].sum()),expert_static_failed=bool(static_valid[-1] and static_value[-1]==0))
        scores=old[0].copy();scores[:,0]=values[:-1,0];scores[:,1]=road[:-1];scores[:,3]=values[:-1,1];scores[:,4]=comfort
        progress,reference=path_progress(p['proposal'],p['target_traj'])
        ratio=formula_progress_ratio(progress,reference)
        scores[:,2]=scores[:,0]*scores[:,1]*ratio
        scores[:,-1]=scores[:,0]*scores[:,1]*(5*scores[:,3]+5*scores[:,2]+2*scores[:,4])/12
        # Both safety components must be known; don't label unknown as clear.
        zero_known=(valid[:-1,0]&(values[:-1,0]==0))|(road_known[:-1]&(scores[:,1]==0))
        valids.append((zero_known|(valid[:-1].all(-1)&road_known[:-1]&comfort_known))&np.isfinite(scores).all(-1))
        diag.update(road_known=road_known[:-1].tolist(),comfort_known=comfort_known.tolist(),actor_repairs=int(prepared['actor_footprint_repaired'].sum()))
        diagnostics.append(diag);old=list(old);old[0]=scores;old[4]=progress.astype(np.float32);old[5]=np.float32(reference)
        scored.append(old)
    device=proposals.device
    convert=lambda x,dtype=torch.float32:torch.as_tensor(np.stack(x),device=device,dtype=dtype)
    scores=convert([x[0] for x in scored]);known=convert(valids,torch.bool)
    agent.base_score_valid=known;agent.base_score_diagnostics=diagnostics
    return (scores[...,-1],scores[...,-1].amax(-1),scores,convert([x[1] for x in scored]),convert([x[2] for x in scored],torch.bool),convert([x[3] for x in scored],torch.bool),[x[-1] for x in scored],convert([x[4] for x in scored]),convert([x[5] for x in scored]))

def _expert_checks_uncached(agent,targets):
    points=score_points(agent,targets,targets['trajectory'][:,None])
    decisions=[]
    for point,sample,prepared in zip(points,targets['_scorer'],targets['_safety']):
        prepare_geometry(point,prepared)
        road,road_known=official_road(agent,point,prepared)
        values,valid,_=aligned_safety(point,sample['future_box_valid'],prepared['actor_world_z'],prepared['ego_world_z'])
        static_value,static_valid,_=static_collision(point['target_traj'][None],prepared)
        values[-1:,0],valid[-1:,0]=merge_collision(values[-1:,0],valid[-1:,0],static_value,static_valid)
        reasons=[bool(valid[-1,0] and values[-1,0]==0),bool(road_known[-1] and road[-1]==0),bool(valid[-1,1] and values[-1,1]==0),not bool(np.isfinite(point['target_traj']).all())]
        decisions.append((not any(reasons),reasons))
    return decisions


def expert_checks(agent,targets):
    from expert_bundle import expert_rows
    bundle=expert_rows(agent,targets)
    if bundle is not None:return [r['checks'] for r in bundle]
    from expert_cache import cached_expert_rows
    rows=cached_expert_rows(agent,targets,'checks',lambda one:_expert_checks_uncached(agent,one)[0])
    return _expert_checks_uncached(agent,targets) if rows is None else rows


def expert_base_scores(agent,targets):
    from expert_bundle import expert_rows
    bundle=expert_rows(agent,targets)
    if bundle is not None:
        return (torch.tensor([r['base'][0] for r in bundle],device=targets['trajectory'].device),
                torch.tensor([r['base'][1] for r in bundle],device=targets['trajectory'].device,dtype=torch.bool))
    from expert_cache import cached_expert_rows
    def compute(one):
        scores=score_batch(agent,one,one['trajectory'][:,None],test=False)[2][:,0,-1]
        return float(scores[0]),bool(agent.base_score_valid[0,0])
    rows=cached_expert_rows(agent,targets,'base',compute)
    if rows is None:
        scores=score_batch(agent,targets,targets['trajectory'][:,None],test=False)[2][:,0,-1]
        return scores,agent.base_score_valid[:,0]
    return (torch.tensor([r[0] for r in rows],device=targets['trajectory'].device),
            torch.tensor([r[1] for r in rows],device=targets['trajectory'].device,dtype=torch.bool))
