"""Recover actor identity and vertical overlap lost by the legacy 2D cache.

Reproduce its actor ordering/XY projection for a strict cache consistency check.
Height is evaluated against the reference road level, as for the road worker.
This is a conservative vertical broad phase, not a CARLA collision sensor.
"""
import numpy as np

GEOMETRY_VERSION = 'actors_xy_full_box_fallback_v1'


def scoring_footprint(legacy_xy, raw_corners, world2ego):
    """Keep normal cached footprints; recover collapsed faces from all 8 corners.

    A fallen actor's local XY face can project to a line although its full 3D
    box has a finite footprint. The enclosing rectangle preserves the [4,2]
    cache interface and conservatively includes the entire projected box.
    No invented minimum size and no removal of the actor is used.
    """
    xy = np.asarray(legacy_xy, dtype=np.float64)
    shifted = xy - xy[0]
    twice_area = abs(np.sum(shifted[:, 0] * np.roll(shifted[:, 1], -1)
                            - shifted[:, 1] * np.roll(shifted[:, 0], -1)))
    if twice_area > 2e-8:
        return xy, False
    from shapely.geometry import MultiPoint
    points = transform_points(raw_corners, world2ego)[:, :2]
    hull = MultiPoint(points).convex_hull
    if hull.geom_type != 'Polygon' or not hull.is_valid or hull.area <= 1e-8:
        raise ValueError('Full raw 3D actor box also has a degenerate XY footprint')
    repaired = np.asarray(hull.minimum_rotated_rectangle.exterior.coords)[:4]
    return repaired, True


def legacy_world_corners(box):
    key = 'world2vehicle' if 'world2vehicle' in box else 'world2ped'
    if key in box and 'center' in box and 'extent' in box:
        matrix = np.asarray(box[key], dtype=np.float64)
        center = np.asarray(box['center'], dtype=np.float64)
        extent = np.asarray(box['extent'], dtype=np.float64)
        if (matrix.shape == (4, 4) and center.shape == (3,) and extent.shape == (3,)
                and all(np.isfinite(x).all() for x in (matrix, center, extent))):
            rotation = np.linalg.inv(matrix)[:2, :2]
            x, y = extent[:2]
            xy = np.array([[x, y], [-x, y], [-x, -y], [x, -y]]) @ rotation.T + center[:2]
            return np.c_[xy, np.full(4, center[2] - extent[2])]
    corners = np.asarray(box.get('world_cord', []), dtype=np.float64)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        return None
    xy = corners[[0, 2, 4, 6], :2]
    delta = xy - xy.mean(axis=0)
    xy = xy[np.argsort(np.arctan2(delta[:, 1], delta[:, 0]))]
    return np.c_[xy, np.full(4, corners[:, 2].min())]


def transform_points(points, world2ego):
    points = np.asarray(points).copy()
    points[..., 1] *= -1
    return (np.c_[points, np.ones(len(points))] @ np.asarray(world2ego).T)[:, :3]


def reconstruct_actor_geometry(future_frames, world2ego, max_agents):
    by_time = []
    first_seen = {}
    for t, frame in enumerate(future_frames):
        actors = {}
        for box in frame['actors']:
            if box['class'] not in ('vehicle', 'walker'):
                continue
            corners = legacy_world_corners(box)
            if corners is None:
                continue
            key = box['class'] + ':' + str(box['id'])
            actors[key] = (box, transform_points(corners, world2ego))
        for key, (_, local) in actors.items():
            first_seen.setdefault(key, (t, float(np.linalg.norm(local.mean(axis=0)[:2]))))
        by_time.append(actors)
    ids = sorted(first_seen, key=first_seen.get)[:max_agents]
    horizon = len(future_frames)
    boxes = np.zeros((max_agents, horizon, 4, 2), dtype=np.float32)
    scoring_boxes = np.zeros_like(boxes)
    repaired = np.zeros((max_agents, horizon), dtype=bool)
    valid = np.zeros((max_agents, horizon), dtype=bool)
    height_overlap = np.zeros_like(valid)
    z_bounds = np.zeros((max_agents, horizon, 2), dtype=np.float64)
    ego_z_bounds = []
    for t, (frame, actors) in enumerate(zip(future_frames, by_time)):
        ego = np.asarray(frame['world_cord'], dtype=np.float64)
        if ego.shape != (8, 3) or not np.isfinite(ego).all():
            raise ValueError('Missing finite raw ego 3D bounding box')
        ego_min, ego_max = ego[:, 2].min(), ego[:, 2].max()
        ego_z_bounds.append([ego_min, ego_max])
        for a, key in enumerate(ids):
            if key not in actors:
                continue
            box, local = actors[key]
            raw = np.asarray(box.get('world_cord', []), dtype=np.float64)
            if raw.shape != (8, 3) or not np.isfinite(raw).all():
                raise ValueError('Missing finite raw actor 3D bounding box: ' + key)
            lower, upper = raw[:, 2].min(), raw[:, 2].max()
            boxes[a, t] = local[:, :2]
            try:
                scoring_boxes[a, t], repaired[a, t] = scoring_footprint(boxes[a, t], raw, world2ego)
            except ValueError as error:
                raise ValueError(f'{key} horizon={t}: {error}') from error
            valid[a, t] = True
            z_bounds[a, t] = [lower, upper]
            # No world-z threshold or scenario whitelist: use actual box overlap.
            height_overlap[a, t] = lower <= ego_max and ego_min <= upper
    return dict(actor_ids=ids, reconstructed_box_corners=boxes.tolist(),
                scoring_box_corners=scoring_boxes.tolist(), actor_footprint_repaired=repaired.tolist(),
                actor_geometry_version=GEOMETRY_VERSION,
                reconstructed_box_valid=valid.tolist(), actor_height_overlap=height_overlap.tolist(),
                actor_z_bounds=z_bounds.tolist(), ego_z_bounds=ego_z_bounds)
