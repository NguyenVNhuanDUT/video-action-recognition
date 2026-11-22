"""Compare the two deployed pipelines on exactly the same labeled videos."""
import argparse
import gc
from pathlib import Path
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from server_v1 import DEFAULT_CK, predictor, replay
from demo_server import camera_predictor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset",type=Path,help="Directory containing normal/ and fight/")
    p.add_argument("--checkpoint",type=Path,default=DEFAULT_CK)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if not args.checkpoint.is_file(): p.error(f"Missing V1 best.pt: {args.checkpoint}")
    rows=[]
    for label,folder in enumerate(["normal","fight"]):
        paths=sorted(x for x in (args.dataset/folder).rglob("*") if x.is_file() and x.suffix.lower() in {".avi",".mp4",".mov",".mkv"})
        if not paths: p.error(f"No labeled videos in {args.dataset/folder}")
        rows.extend(dict(path=str(path),truth=label) for path in paths)
    for name in ("v1","v2"):
        predict=predictor(args.checkpoint) if name=="v1" else camera_predictor(require_pair=True)
        for row in rows:
            windows=list(replay(row["path"],predict))
            row[name+"_alert"]=int(any(w["decision"] in {"suspected","fight"} for w in windows))
            row[name+"_peak"]=max(w["p_action"] for w in windows)
        del predict
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    df=pd.DataFrame(rows)
    for name in ("v1","v2"):
        pred=df[name+"_alert"]
        tn,fp,fn,tp=confusion_matrix(df.truth,pred,labels=[0,1]).ravel()
        pr,rc,f1,_=precision_recall_fscore_support(df.truth,pred,average="binary",zero_division=0)
        print(f"{name}: TP={tp} FP={fp} FN={fn} TN={tn} precision={pr:.3f} recall={rc:.3f} F1={f1:.3f}")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.output,index=False,encoding="utf-8-sig")


if __name__=="__main__":
    main()

