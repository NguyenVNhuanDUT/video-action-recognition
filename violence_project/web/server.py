"""V1 webcam dashboard. Use --help for checkpoint and replay options."""
import argparse
import errno
import threading
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

if __package__:
    from . import monitor
else:
    import monitor

ROOT = monitor.WEB.parent
DEFAULT_CK = ROOT / "checkpoints/best.pt"
THRESHOLD = 0.61


def predictor(checkpoint):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing V1 best.pt: {checkpoint}. Supply --checkpoint PATH.")
    config = VideoMAEConfig.from_pretrained(monitor.WEB,num_labels=2)
    model = VideoMAEForVideoClassification(config)
    state = torch.load(checkpoint,map_location="cpu",weights_only=True,mmap=True)
    # This Transformers version registers query/value bias directly on self-attention.
    names = model.state_dict()
    for key in list(state):
        for old,new in ((".attention.query.bias",".attention.q_bias"),
                        (".attention.value.bias",".attention.v_bias")):
            target = key.removesuffix(old) + new
            if key.endswith(old) and target in names and target not in state:
                state[target] = state.pop(key)
    model.load_state_dict(state,strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    mean = np.array([.485,.456,.406],np.float32)
    std = np.array([.229,.224,.225],np.float32)

    @torch.inference_mode()
    def predict(frames):
        ids = ((np.arange(16)+.5)*len(frames)/16).astype(int)
        rgb = np.stack([cv2.resize(frames[i],(256,256))[16:240,16:240,::-1] for i in ids])
        x = torch.from_numpy((rgb.astype(np.float32)/255-mean)/std).permute(0,3,1,2)[None].to(device)
        with torch.autocast("cuda",enabled=device.type=="cuda",dtype=torch.float16):
            logits = model(pixel_values=x).logits
        score = logits.float().softmax(-1)[0,1].item()
        return ("suspected" if score>=THRESHOLD else "normal"), score
    return predict


def replay(path,predict):
    """Sample at the dashboard rate, evaluate every half-second, include the tail."""
    from collections import deque
    path=Path(path).resolve()
    cap=cv2.VideoCapture(str(path))
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not np.isfinite(fps) or fps<=0:
        cap.release()
        raise ValueError(f"Cannot read {path}")
    ring=deque(maxlen=16)
    tick=index=0
    def result(end):
        decision,score=predict(list(ring))
        return dict(path=str(path),start_s=max(0,end-len(ring)/16),end_s=end,p_action=score,decision=decision)
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            index+=1
            while tick/16 < index/fps:
                ring.append(cv2.resize(frame,(640,360)))
                tick+=1
                if tick%8==0: yield result(min(tick/16,index/fps))
        if not ring: raise ValueError(f"Empty video: {path}")
        if tick%8: yield result(index/fps)
    finally:
        cap.release()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=DEFAULT_CK)
    parser.add_argument("--video",type=Path,action="append",default=[],help="Optional MP4 file for replay; repeatable")
    args=parser.parse_args()
    url = "http://127.0.0.1:8878/"
    try:
        with urlopen(url, timeout=1) as response:
            already_running = b"<title>Sentinel " in (page := response.read(4096)) and b"V1</title>" in page
    except OSError:
        already_running = False
    if already_running:
        print(f"V1 is already running: {url}")
        if args.video or args.checkpoint != DEFAULT_CK:
            parser.exit(1, "Stop the existing V1 server before changing videos or checkpoint.\n")
        return
    if not args.checkpoint.is_file(): parser.error(f"Missing V1 best.pt: {args.checkpoint}")
    predict=predictor(args.checkpoint)
    rows=[r for path in args.video for r in replay(path,predict)]
    windows=pd.DataFrame(rows,columns=["path","start_s","end_s","p_action","decision"])
    try:
        server,camera=monitor.launch(windows,predict,port=8878)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE or getattr(exc,"winerror",None) == 10048:
            parser.exit(1, "Port 8878 is in use. Check the existing server before restarting V1.\n")
        raise
    print("V1: http://127.0.0.1:8878/",flush=True)
    try: threading.Event().wait()
    except KeyboardInterrupt: camera.stop.set()
    finally:
        if camera.worker: camera.worker.join()
        server.shutdown()
        server.server_close()


if __name__=="__main__":
    main()
