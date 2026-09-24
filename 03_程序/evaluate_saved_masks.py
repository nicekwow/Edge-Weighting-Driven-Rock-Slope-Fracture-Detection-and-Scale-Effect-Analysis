"""Recompute saved 21-condition and 65-image results without model inference."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_评价结果'
DATA=ROOT/'02_实验数据'

def metrics(pred,ref):
    assert pred.shape==ref.shape
    tp=int((pred&ref).sum()); fp=int((pred&~ref).sum())
    fn=int((~pred&ref).sum()); tn=int((~pred&~ref).sum())
    fracture=100*tp/(tp+fp+fn)
    background=100*tn/(tn+fp+fn)
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,fracture_iou_pct=fracture,
                background_iou_pct=background,miou_pct=(fracture+background)/2)

def read(p): return np.asarray(Image.open(p))!=0

def main():
    rows=json.loads((OUT/'图13至16'/'per_image_metrics.json').read_text(encoding='utf-8'))
    assert len(rows)==21
    seen=set()
    max_difference=0.
    for r in rows:
        key=(r['sid'],r['strategy'],r['size_px'])
        assert key not in seen
        seen.add(key)
        name=r['strategy'].lower().replace(' ','_').replace('-','_')
        size=r['size_px'] or 'full'
        pred=read(OUT/'图13至16'/'predictions'/f"{r['sid']}__{name}__{size}.png")
        ref=read(DATA/'三幅重建图像与标注'/f"{r['sid']}_mask.png")
        scores=metrics(pred,ref)
        for k,v in scores.items():
            difference=abs(v-r[k]); max_difference=max(max_difference,difference)
            assert difference<1e-9,(key,k,v,r[k])
    old=list(csv.DictReader((OUT/'尺度分组65图'/'per_image_metrics.csv').open(encoding='utf-8-sig')))
    assert len(old)==65
    aggregates={group:np.zeros((2,2),dtype=np.int64) for group in ('small','large')}
    for r in old:
        group=r['group']; name=Path(r['image']).stem+'.png'
        ref=read(DATA/group/'ann'/name)
        pred=read(OUT/'尺度分组65图'/group/'predictions'/name)
        scores=metrics(pred,ref)
        cm=np.array([[scores['tn'],scores['fp']],[scores['fn'],scores['tp']]])
        aggregates[group]+=cm
        assert abs(scores['miou_pct']-float(r['mIoU']))<1e-9
    aggregates['combined']=aggregates['small']+aggregates['large']
    summaries=[]
    for group,cm in aggregates.items():
        diag=np.diag(cm); iou=100*diag/(cm.sum(0)+cm.sum(1)-diag)
        summaries.append(dict(group=group,mIoU=float(iou.mean()),IoU_background=float(iou[0]),IoU_fracture=float(iou[1]),confusion_matrix=cm.tolist()))
    report=dict(window_conditions=21,window_max_absolute_difference=max_difference,
                scale_images=65,scale_summaries=summaries,status='passed')
    (OUT/'已保存掩码复算核对.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
