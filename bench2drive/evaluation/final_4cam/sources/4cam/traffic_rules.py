"""Pure trajectory-level aggregation; no predicted actors or signal states."""
import numpy as np

def direction_score(positions,oncoming,valid,dt=.1):
    distance=np.linalg.norm(np.diff(positions,axis=1,prepend=positions[:,:1]),axis=-1)
    wrong=distance*np.asarray(oncoming)
    window=int(1./dt)
    maximum=np.stack([wrong[:,max(0,i-window+1):i+1].sum(-1) for i in range(wrong.shape[1])],-1).max(-1)
    value=np.where(maximum<2.,1.,np.where(maximum<6.,.5,0.))
    # A definite >=6m violation is known even with another unknown segment.
    known=np.asarray(valid).all(-1)|(maximum>=6.)
    return value,known

def stop_sequence(affected,speeds,initial_unknown=False,evaluation_start=0,eligible=None):
    active=False;stopped=False;unknown=initial_unknown
    for i,(inside,speed) in enumerate(zip(affected,speeds)):
        if not active:
            if inside and (eligible is None or eligible[i]):active=True;stopped=False
            continue  # Official criterion only checks speed on subsequent ticks.
        if speed<.1:stopped=True;unknown=False
        if not inside:
            if not stopped and i>=evaluation_start:return 0.,not unknown
            active=False;stopped=False;unknown=False
    if active and not stopped:return 0.,False  # obligation unresolved at horizon
    return 1.,not unknown


def known_and(values, valid):
    """Conjunction over metrics: one proven failure overrides unknown peers."""
    values=np.asarray(values);valid=np.asarray(valid,bool)
    failed=(valid & (values==0)).any(axis=0)
    return np.where(failed,0.,1.), failed | valid.all(axis=0)
