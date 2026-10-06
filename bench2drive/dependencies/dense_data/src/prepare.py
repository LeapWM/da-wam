import concurrent.futures,json,os
from pathlib import Path
from data import index_clip

def prepare(root,out):
    root=Path(root);out=Path(out)
    from audit_ready import verify_acceptance
    verify_acceptance(root)
    contract=Path('/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/data_support/data_contract')
    val={Path(n).name for n in json.loads((contract/'official_split.json').read_text())['val']}
    base=sorted(n[:-7] for n in json.loads((contract/'official_base.json').read_text()))
    assert len(base)==1000 and len(val)==50 and val<=set(base)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:clips=list(pool.map(index_clip,[root/'v1'/c for c in base]))
    report={'train':[c for c in clips if c['folder'] not in val],'val':[c for c in clips if c['folder'] in val]}
    report['contract']={'input_sampling_hz':10,'history_seconds':.5,'future_steps':8,'future_dt':.5,'cameras':1,
                        'missing_future':'exclude explicitly; never extrapolate','source':json.loads((root/'DATA_READY.json').read_text())}
    out.mkdir(parents=True,exist_ok=True);tmp=out/f'index.{os.getpid()}.tmp';tmp.write_text(json.dumps(report));tmp.replace(out/'index.json')
    for split in ('train','val'):
        print(split,'clips',len(report[split]),'eligible',sum(len(c['frames']) for c in report[split]),'raw',sum(c['raw'] for c in report[split]),flush=True)
    return report

if __name__=='__main__':prepare(os.environ['B2D_NEW_CACHE'],os.environ['B2D_DENSE_INDEX'])
