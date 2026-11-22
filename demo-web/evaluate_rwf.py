from pathlib import Path
import time
import cv2
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from demo_server import ROOT, camera_predictor

def main():
    predict = camera_predictor()
    rows = []
    for path in sorted((ROOT/'dataset/RWF-2000/val').glob('*/*.avi')):
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok: break
            frames.append(frame)
        cap.release()
        if not frames: raise RuntimeError(str(path))
        tick = time.perf_counter()
        kind, score = predict(frames)
        rows.append((path.relative_to(ROOT),int(path.parent.name=='Fight'),int(kind!='normal'),score,time.perf_counter()-tick))
    df=pd.DataFrame(rows,columns=['path','truth','alert','score','inference_seconds'])
    df.to_csv(ROOT/'demo-web/rwf_eval.csv',index=False)
    print(classification_report(df.truth,df.alert,digits=4))
    print(confusion_matrix(df.truth,df.alert))
    print('Median inference seconds',df.inference_seconds.median())
if __name__=='__main__':
    main()
