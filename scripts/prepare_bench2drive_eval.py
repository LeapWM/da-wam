"""Validate or prepare a fresh single-worker final-model Bench2Drive evaluation."""
import argparse,json,os,shlex
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[1]; frozen=root/'bench2drive/evaluation/final_4cam'
config=json.loads((frozen/'4cam_score11/config.json').read_text())
for name in ['checkpoint','hydra_config','model_root']:
 path=Path(config['model'][name]);assert path.exists(),(name,path)
assert Path(config['model_launcher']).is_file()
if a.check:print(json.dumps({'status':'paths_ok','checkpoint':config['model']['checkpoint'],'evaluation_started':False},indent=2))
if a.output:
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 config['model']['checkpoint']=str(root/'checkpoints/bench2drive_4cam_score11.ckpt')
 (out/'config.json').write_text(json.dumps(config,indent=2))
 def q(x):return shlex.quote(str(x))
 script='#!/usr/bin/env bash\nset -euo pipefail\n'
 script+='export DRIVE_JEPA_B2D_CONFIG='+q(out/'config.json')+'\nexport DRIVE_JEPA_B2D_RESULT_DIR='+q(out/'results')+'\n'
 script+='export CARLA_GPU="${CARLA_GPU:-0}" DRIVE_JEPA_GPU="${DRIVE_JEPA_GPU:-1}"\n'
 script+='export CARLA_PORT="${CARLA_PORT:-2600}" TM_PORT="${TM_PORT:-8600}" DRIVE_JEPA_BRIDGE_PORT="${DRIVE_JEPA_BRIDGE_PORT:-50800}"\n'
 script+='export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n'
 script+='exec bash '+q(frozen/'resume.sh')+'\n'
 (out/'run.sh').write_text(script);(out/'run.sh').chmod(0o755)
 print('Prepared '+str(out/'run.sh')+'; no evaluation started.')
if not a.check and not a.output:p.error('provide --check or --output')
