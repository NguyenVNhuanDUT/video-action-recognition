"""Run the local V2 camera dashboard and SQLite event archive."""
import json
import threading

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

from monitor import STORE, WEB, launch

ROOT = WEB.parent
RUN = ROOT / "train_v2/runs/local_gpu_01"


def camera_predictor(require_pair=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = VideoMAEConfig.from_pretrained(ROOT / "train_v2/cache/hf", num_labels=2)
    model = VideoMAEForVideoClassification(config)
    checkpoint = torch.load(RUN/"best.pt", weights_only=True, map_location="cpu", mmap=True)
    model.load_state_dict({k.removeprefix("video."): v for k,v in checkpoint["model"].items()
                           if k.startswith("video.")})
    model.eval().to(device)
    threshold = json.loads((RUN/"policy.json").read_text())["action_threshold"]
    if require_pair:
        from torchvision.models.detection import keypointrcnn_resnet50_fpn
        pose = keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                         min_size=320, max_size=576, box_score_thresh=.8,
                                         box_nms_thresh=.4)
        weights = ROOT/"train_v2/cache/torch/hub/checkpoints/keypointrcnn_resnet50_fpn_coco-fc266e95.pth"
        pose.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        pose.eval().to(device)
    mean = np.array([.485,.456,.406],dtype=np.float32)
    std = np.array([.229,.224,.225],dtype=np.float32)

    @torch.inference_mode()
    def predict(frames):
        images = []
        for idx in ((np.arange(16)+.5)*len(frames)/16).astype(int):
            rgb = cv2.cvtColor(frames[idx],cv2.COLOR_BGR2RGB)
            h,w = rgb.shape[:2]
            width,height = max(1,round(w*224/max(w,h))),max(1,round(h*224/max(w,h)))
            canvas = np.full((224,224,3),114,np.uint8)
            x,y = (224-width)//2,(224-height)//2
            canvas[y:y+height,x:x+width] = cv2.resize(rgb,(width,height))
            images.append(canvas)
        inputs = (np.stack(images).astype(np.float32)/255-mean)/std
        tensor = torch.from_numpy(inputs).permute(0,3,1,2)[None].to(device)
        with torch.autocast("cuda",enabled=device.type=="cuda",dtype=torch.float16):
            logits = model(pixel_values=tensor).logits
        score = logits.float().softmax(-1)[0,1].item()
        if require_pair and score >= threshold:
            for i in (len(frames)//2, len(frames)-1):
                frame = cv2.cvtColor(frames[i],cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(frame).permute(2,0,1).to(device).float()/255
                detection = pose([tensor])[0]
                if (detection["labels"] == 1).sum().item() >= 2:
                    break
            else:
                return "no_pair",score
        return ("suspected" if score>=threshold else "normal"),score
    return predict


if __name__ == "__main__":
    predict = camera_predictor(require_pair=True)
    capture = cv2.VideoCapture(str(ROOT/"KO_1s.mp4"))
    frames = []
    while len(frames)<16:
        ok, frame = capture.read()
        if not ok: break
        frames.append(frame)
    capture.release()
    print("Model check:", predict(frames), flush=True)
    server,camera = launch(pd.read_csv(STORE/"windows.csv"),predict,port=8877)
    print("http://127.0.0.1:8877/",flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        camera.stop.set()
        if camera.worker: camera.worker.join()
    finally:
        server.shutdown()
        server.server_close()

