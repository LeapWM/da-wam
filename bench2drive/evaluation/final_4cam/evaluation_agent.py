import json, os
from pathlib import Path
from multiview_agent import MultiViewBench2DriveAgent

def get_entry_point(): return 'RecordedMultiViewAgent'
class RecordedMultiViewAgent(MultiViewBench2DriveAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        self.metrics = {}
        self.calls = 0
        self.output = Path(os.environ['SAVE_PATH']) / path_to_conf_file.split('+')[-1]
        self.output.mkdir(parents=True, exist_ok=True)
    def _request_inference(self, jpeg, timestamp, speed):
        previous = self._last_inference_time
        super()._request_inference(jpeg, timestamp, speed)
        self.calls += 1
        if self.calls <= 3 or self._step % 40 == 0:
            print('REPLAN_AUDIT cameras={} step={} calls={} delta={} shape={}x2'.format(len(jpeg),self._step,self.calls,None if previous is None else round(timestamp-previous,6),len(self._world_trajectory)),flush=True)
    def run_step(self, input_data, timestamp):
        control = super().run_step(input_data, timestamp)
        if self._step % 2 == 0: self.metrics[str(self._step)] = self.get_metric_info()
        if self._step % 200 == 0: self.flush_metrics()
        return control
    def flush_metrics(self):
        p = self.output / 'metric_info.json'
        t = p.with_suffix('.tmp'); t.write_text(json.dumps(self.metrics)); t.replace(p)
    def destroy(self):
        if hasattr(self,'output'): self.flush_metrics()
        super().destroy()
