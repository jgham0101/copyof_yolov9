from pathlib import Path
import yaml
Y=Path('models/yolo.py')
SRC=Path('models/detect/yolov9-s-v10o2m-equivalent.yaml')
DST=Path('models/detect/yolov9-s-v10dual-dormant.yaml')
t=Y.read_text()
mark='# WEEK26_DORMANT_O2O'
if mark not in t:
    if 'from copy import deepcopy' not in t:
        t='from copy import deepcopy\n'+t
    anchor='class DualDDetect(nn.Module):'
    if anchor not in t: raise RuntimeError('DualDDetect anchor not found')
    code=(
      '# WEEK26_DORMANT_O2O\n'
      'class V10DualDormantDetect(DDetect):\n'
      '    def __init__(self, nc=80, ch=()):\n'
      '        super().__init__(nc, ch)\n'
      '        self.one2one_cv2 = deepcopy(self.cv2)\n'
      '        self.one2one_cv3 = deepcopy(self.cv3)\n\n'
      '    def forward(self, x):\n'
      '        # Week26: O2O modules deliberately not executed.\n'
      '        return super().forward(x)\n\n\n')
    t=t.replace(anchor,code+anchor,1)
    old=('elif m in {Detect, DualDetect, TripleDetect, DDetect, '
         'V10O2MDetect, DualDDetect, TripleDDetect, Segment, '
         'DSegment, DualDSegment, Panoptic}:')
    new=('elif m in {Detect, DualDetect, TripleDetect, DDetect, '
         'V10O2MDetect, V10DualDormantDetect, DualDDetect, TripleDDetect, '
         'Segment, DSegment, DualDSegment, Panoptic}:')
    if old not in t: raise RuntimeError('Week25 parse_model head set not found')
    t=t.replace(old,new,1); Y.write_text(t)
cfg=yaml.safe_load(SRC.read_text())
cfg['head'][-1]=[[15,18,21],1,'V10DualDormantDetect',['nc']]
DST.write_text(yaml.safe_dump(cfg,sort_keys=False))
print('patched',Y); print('created',DST)
