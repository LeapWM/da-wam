"""Accept all official frames and splits; save the exact accepted file identities."""
import argparse,hashlib,json,pickle,time
from pathlib import Path

def audit(root):
    root=Path(root)
    state=json.loads((root/'PROGRESS.json').read_text())
    print('Data preparation:',state['status'],'completed current phase:',len(state['completed']),flush=True)
    if state['status']!='complete':raise RuntimeError('Full training blocked until all official 950/50 clips are converted')
    contract=Path(__file__).resolve().parents[1]/'data_contract'
    base=json.loads((contract/'official_base.json').read_text())
    val={Path(x).name for x in json.loads((contract/'official_split.json').read_text())['val']}
    expected={'val':val,'train':{n[:-7] for n in base}-val}
    sets={};report=dict(status='accepted',time=time.strftime('%Y-%m-%d %H:%M:%S'),splits={})
    for split,count in [('train',950),('val',50)]:
        path=root/'infos'/f'b2d_infos_{split}.pkl'
        with path.open('rb') as f:infos=pickle.load(f)
        tokens=[r['token'] for r in infos];assert len(tokens)==len(set(tokens))
        clips={r['folder'] for r in infos};assert len(clips)==count
        assert {Path(c).name for c in clips}==expected[split], 'Official split identity mismatch'
        sets[split]=clips;frames={c:[] for c in clips}
        for row in infos:frames[row['folder']].append(row['frame_idx'])
        for clip,ids in frames.items():
            raw=sorted(int(p.name.split('.')[0]) for p in (root/clip/'anno').glob('*.json.gz'))
            assert ids==raw, 'Raw frame coverage mismatch: '+clip
        for prev,cur in zip(infos,infos[1:]):
            if prev['folder']==cur['folder']:
                assert cur['frame_idx']==prev['frame_idx']+1
                assert abs(cur['timestamp']-prev['timestamp']-100000)<1
        stat=path.stat()
        report['splits'][split]=dict(clips=len(clips),frames=len(infos),empty_object_frames=sum(len(r['gt_boxes'])==0 for r in infos),path=str(path.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns)
        print(split,report['splits'][split],flush=True)
        del infos,tokens,frames
    assert not sets['train']&sets['val']
    report['progress_sha256']=hashlib.sha256((root/'PROGRESS.json').read_bytes()).hexdigest()
    temp=root/'DATA_READY.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(root/'DATA_READY.json')
    print('DATA READY',flush=True)
    return report

def verify_acceptance(root):
    root=Path(root);report=json.loads((root/'DATA_READY.json').read_text())
    assert report['status']=='accepted'
    assert report['progress_sha256']==hashlib.sha256((root/'PROGRESS.json').read_bytes()).hexdigest(), 'Preparation changed after audit'
    for split,info in report['splits'].items():
        path=root/'infos'/f'b2d_infos_{split}.pkl';stat=path.stat()
        assert (str(path.resolve()),stat.st_size,stat.st_mtime_ns)==(info['path'],info['size'],info['mtime_ns']), 'Accepted annotations changed'

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/mnt/c2-worldmodel/2639639/Bench2Drive/sparsedrivev2_base_10hz'))
    audit(ap.parse_args().root)
