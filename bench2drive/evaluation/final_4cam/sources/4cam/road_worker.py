"""CARLA-map-only labels. No simulator, inferred light states or actor forecasts."""
import hashlib,json,math,sys
from pathlib import Path
import carla,numpy as np
from shapely.geometry import LineString
from traffic_rules import direction_score,stop_sequence,known_and

ROOT=Path('/mnt/c2-worldmodel/2639639/Bench2Drive/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps')
MAPS={};STOPS={}
class RequestMap:
    """Exact CARLA-location memoization, bounded to one scoring request."""
    def __init__(self,source):self.source=source;self.cache={}
    def get_waypoint(self,loc,project_to_road=True,lane_type=carla.LaneType.Driving):
        key=(loc.x,loc.y,loc.z,bool(project_to_road),int(lane_type))
        if key not in self.cache:
            self.cache[key]=self.source.get_waypoint(loc,project_to_road=project_to_road,lane_type=lane_type)
        return self.cache[key]

def location(p):return carla.Location(x=float(p[0]),y=float(p[1]),z=float(p[2]))
def getmap(town):
    if town not in MAPS:
        paths=[ROOT/town/'OpenDrive'/(town+'.xodr'),ROOT/'OpenDrive'/(town+'.xodr')]
        path=next(p for p in paths if p.is_file());raw=path.read_bytes()
        MAPS[town]=(carla.Map(town,raw.decode()),hashlib.sha256(raw).hexdigest())
    return MAPS[town]
def road_point(m,p):
    loc=location(p)
    w=[m.get_waypoint(loc,project_to_road=True,lane_type=t) for t in (carla.LaneType.Driving,carla.LaneType.Parking)]
    return any(x is not None and loc.distance(x.transform.location)-x.lane_width/2<=.15 for x in w)
def stop_lines(m,town,b):
    key=(town,str(b['id']),str(b['trigger_volume_location']),str(b['rotation']),str(b['trigger_volume_extent']))
    if key in STOPS:return STOPS[key]
    center=np.asarray(b['trigger_volume_location'],float);extent=np.asarray(b['trigger_volume_extent'],float)
    yaw=math.radians(float(b['rotation'][2]));wps=[]
    if center.shape!=(3,) or extent.shape!=(3,) or not np.isfinite(center).all() or not (extent>0).all():raise ValueError('Missing light geometry')
    for x in np.arange(-.9*extent[0],.9*extent[0],1.):
        p=center+np.array([math.cos(yaw)*x,math.sin(yaw)*x,0.]);w=m.get_waypoint(location(p))
        if w is None:raise ValueError('No light waypoint')
        if not wps or (wps[-1].road_id,wps[-1].lane_id)!=(w.road_id,w.lane_id):wps.append(w)
    result=[]
    for w in wps:
        for _ in range(400):
            if w.is_intersection:break
            nxt=w.next(.5)
            if len(nxt)!=1:raise ValueError('Missing or ambiguous light successor')
            if nxt[0].is_intersection:break
            w=nxt[0]
        else:raise ValueError('No unambiguous junction stop line')
        v=w.transform.get_forward_vector();p=w.transform.location
        side=np.array([-v.y,v.x])*.6*w.lane_width
        result.append((w.road_id,w.lane_id,np.array([v.x,v.y]),np.array([p.x,p.y]),side))
    if not result:raise ValueError('Empty signal lane mapping')
    STOPS[key]=result;return result

def evaluate(q):
    m,digest=getmap(q['town'])
    if q.get('road_only'):
        poses=np.asarray(q['world_positions'],float);available=np.asarray(q['available'],bool)
        onroad=np.zeros(available.shape,bool)
        for index in np.ndindex(available.shape):
            if not available[index]:continue
            loc=location(poses[index]);choices=[m.get_waypoint(loc,lane_type=t) for t in (carla.LaneType.Driving,carla.LaneType.Parking)]
            choices=[w for w in choices if w is not None]
            if choices:
                w=min(choices,key=lambda w:loc.distance(w.transform.location))
                onroad[index]=loc.distance(w.transform.location)<=w.lane_width/2+.5
        return {'ok':True,'road':onroad.tolist(),'valid':available.tolist(),'map_sha256':digest}
    m=RequestMap(m)
    corners=np.asarray(q['corners'],float);poses=np.asarray(q['world_positions'],float);forward=np.asarray(q['world_forward'],float)
    available=np.asarray(q['available'],bool);p,t=poses.shape[:2]
    traffic_only=q.get('traffic_only',False)
    road=np.ones((p,t),bool);road_valid=available.copy();direction=np.ones((p,t),bool);direction_valid=np.zeros((p,t),bool)
    signal=np.ones((p,t),bool);signal_valid=np.zeros((p,t),bool)
    for ti in range(t):
        lights=q['lights'][ti];mapped=[];evidence=bool(lights)
        if lights:
            for b in lights:
                if b['state'] not in (0,1,2):evidence=False;continue
                try:mapped.extend((b['state'],line) for line in stop_lines(m,q['town'],b))
                except (ValueError,TypeError,IndexError):evidence=False
        for pi in range(p):
            if not available[pi,ti]:continue
            pos=poses[pi,ti]
            # Center + actual recorded-condition corners; this is sampled
            # footprint coverage, not a polygon proof of complete map legality.
            if not traffic_only:
                road[pi,ti]=all(road_point(m,x) for x in np.vstack((pos,corners[pi,ti])))
            w=m.get_waypoint(location(pos),project_to_road=False,lane_type=carla.LaneType.Driving)
            if w is not None and not w.is_junction:
                v=w.transform.get_forward_vector();direction_valid[pi,ti]=True
                direction[pi,ti]=np.dot(forward[pi,ti,:2],[v.x,v.y])>=0
            elif w is not None and w.is_junction:
                direction_valid[pi,ti]=True  # EPDMS excludes intersections.
            # Official red-only state and rear-tail/stop-line intersection.
            # Recorded nearby light list limits scope; not all world signals.
            signal_valid[pi,ti]=evidence
            half=float(q['half_length']);fwd=forward[pi,ti,:2]
            near=pos[:2]-.8*half*fwd;far=pos[:2]-(half+1)*fwd
            wp=m.get_waypoint(location([far[0],far[1],pos[2]]))
            if wp is None:signal_valid[pi,ti]=False;continue
            for state,(rid,lid,v,center,side) in mapped:
                if state!=0 or (wp.road_id,wp.lane_id)!=(rid,lid) or np.dot(fwd,v)<=0:continue
                if LineString([near,far]).intersects(LineString([center+side,center-side])):
                    signal[pi,ti]=False;signal_valid[pi,ti]=True;break
    def aggregate(value,valid):
        fail=(valid&~value).any(-1);clear=(valid&value).all(-1)
        return np.where(fail,0.,np.where(clear,1.,np.nan)),fail|clear
    r,rv=aggregate(road,road_valid)
    if traffic_only:r[:]=np.nan;rv[:]=False
    history=q['history']
    # Only the contiguous, finite, non-teleporting observed prefix can carry
    # a fulfilled stop obligation into the planning horizon.
    start=0
    for i,h in enumerate(history):
        if not np.isfinite(h['position']).all() or not np.isfinite(h['speed']):
            start=i+1;continue
        if i and (h['frame']-history[i-1]['frame']!=1 or np.linalg.norm(np.asarray(h['position'])-history[i-1]['position'])>10.):start=i
    history=history[start:]
    if not history:raise ValueError('No valid current ego history')
    initial=np.asarray(history[-1]['position'],float)
    d,dd=direction_score(np.concatenate((np.broadcast_to(initial,(p,1,3)),poses),1),np.concatenate((np.zeros((p,1),bool),~direction&direction_valid),1),np.concatenate((np.ones((p,1),bool),direction_valid),1))
    dv=dd;s,sv=aggregate(signal,signal_valid)
    stops={str(b['id']):b for frame in q['stops'] if frame is not None for b in frame}
    # An absent catalog is unknown, never fabricated as stop-compliant.
    stop_value=np.ones(p);stop_valid=np.full(p,bool(stops) and all(x is not None for x in q['stops']))
    stop_failed=np.zeros(p,bool)
    eligible_cache={}
    for b in stops.values():
        try:
            center=np.asarray(b['trigger_volume_location'],float);extent=np.asarray(b['trigger_volume_extent'],float)
            if center.shape!=(3,) or extent.shape!=(3,) or not np.isfinite(center).all() or not (extent>0).all():raise ValueError()
        except (KeyError,TypeError,ValueError):stop_valid[:]=False;continue
        def affected(pos):
            w=m.get_waypoint(location(pos))
            if w is None:return None
            if location(center).distance(w.transform.location)>4.:return False
            for _ in range(9):
                v=w.transform.location
                if abs(v.x-center[0])<1.2*extent[0] and abs(v.y-center[1])<1.2*extent[1]:return True
                nxt=w.next(.5)
                if not nxt:break
                w=nxt[0]
            return False
        hist=[affected(x['position']) for x in history]
        for pi in range(p):
            current=[affected(pos) if available[pi,ti] else None for ti,pos in enumerate(poses[pi])]
            if any(x is None for x in hist+current):stop_valid[pi]=False;continue
            velocities=np.linalg.norm(np.diff(np.vstack((initial,poses[pi])),axis=0),axis=-1)/.1
            if pi not in eligible_cache:
                positions=[x['position'] for x in history]+poses[pi].tolist()
                forwards=[x['forward'] for x in history]+forward[pi].tolist()
                moves=np.diff(np.asarray(positions),axis=0,prepend=np.asarray(positions[:1]))/.1
                eligible=[]
                for pos,fwd,move in zip(positions,forwards,moves):
                    wp=m.get_waypoint(location(pos));lane=wp.transform.get_forward_vector()
                    eligible.append(np.dot(fwd,[lane.x,lane.y,lane.z])>=-.17 and np.dot(move,fwd)>=-.17)
                eligible_cache[pi]=eligible
            eligible=eligible_cache[pi]
            v,known=stop_sequence(hist+current,[x['speed'] for x in history]+velocities.tolist(),initial_unknown=bool(hist[0]),evaluation_start=len(hist),eligible=eligible)
            if known and v==0:stop_value[pi]=0;stop_failed[pi]=True
            stop_valid[pi]&=known
    stop_valid |= stop_failed
    # Both red and stop must be supported; any known violation is sufficient.
    traffic,traffic_valid=known_and([s,stop_value],[sv,stop_valid])
    # JSON has no NaN: unknown entries get neutral placeholders plus masks.
    return {'ok':True,'version':'sampled_road_and_recorded_red_v1','map_sha256':digest,
            'value':np.nan_to_num(np.stack([r,d,traffic],-1)).tolist(),'valid':np.stack([rv,dv,traffic_valid],-1).tolist()}

if __name__=='__main__':
    for line in sys.stdin:
        try:result=evaluate(json.loads(line))
        except Exception as e:result={'ok':False,'error':type(e).__name__+': '+str(e)}
        print(json.dumps(result,allow_nan=False),flush=True)
