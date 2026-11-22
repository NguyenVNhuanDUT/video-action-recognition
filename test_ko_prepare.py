import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

out = Path('ko_test_results')
out.mkdir(exist_ok=True)
rows = []
for path in sorted(Path('.').glob('KO*.mp4')):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    edges = np.linspace(0, n, 17)
    indices = [min(int((a+b)/2), n-1) for a,b in zip(edges[:-1], edges[1:])]
    sheet = Image.new('RGB', (4*224, 4*250), '#111827')
    draw = ImageDraw.Draw(sheet)
    for k, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Cannot decode {path}: {idx}')
        frame = cv2.cvtColor(cv2.resize(frame, (256,256)), cv2.COLOR_BGR2RGB)[16:240,16:240]
        x,y = (k%4)*224,(k//4)*250
        sheet.paste(Image.fromarray(frame),(x,y))
        draw.text((x+6,y+228),f'frame {idx} | {idx/fps:.3f}s',fill='white')
    cap.release()
    sheet.save(out / f'{path.stem}_model_input.jpg')
    rows.append(dict(file=path.name,frames=n,fps=fps,duration_s=n/fps,sampled_indices=indices))
(out/'video_info.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

cells=[]
def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
def code(s): cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True)))
md('''# Test 3 video KO với checkpoint hiện tại

Chạy trên Colab GPU. Chọn đúng `best.pt` và ba file KO khi được yêu cầu tải lên.
Notebook so sánh cách lấy 16 frame trên cả clip với cửa sổ 0,5 và 1 giây.
Ngưỡng 0,61 chỉ là ngưỡng cũ để tham khảo, chưa được chọn lại cho cửa sổ ngắn.
Điểm cửa sổ ngắn chỉ phục vụ chẩn đoán; không chứng minh model đã cải thiện.
Không chỉnh ngưỡng dựa vào ba clip này rồi báo cáo chúng là test độc lập.
''')
code('%pip install -q transformers opencv-python-headless pandas matplotlib\n')
code('''from google.colab import files
uploaded = files.upload()  # best.pt + KO_1s.mp4 + KO3_2s.mp4 + KO4_dai.mp4
''')
code('''import cv2, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from transformers import VideoMAEForVideoClassification
assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > GPU'
CHECKPOINT = Path('best.pt')  # or change to your mounted Drive path
VIDEOS = [Path(n) for n in ['KO_1s.mp4','KO3_2s.mp4','KO4_dai.mp4']]
assert CHECKPOINT.exists(), CHECKPOINT
assert all(p.exists() for p in VIDEOS), 'Missing video files'
model = VideoMAEForVideoClassification.from_pretrained(
    'MCG-NJU/videomae-base-finetuned-kinetics',
    num_labels=2, ignore_mismatched_sizes=True)
state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
model.load_state_dict(state['model'] if 'model' in state else state)
model.cuda().eval()
MEAN=np.array([.485,.456,.406],dtype=np.float32)
STD=np.array([.229,.224,.225],dtype=np.float32)
THR=.61

def read_video(p):
    cap=cv2.VideoCapture(str(p)); fps=cap.get(cv2.CAP_PROP_FPS); frames=[]
    assert fps>0, f'Invalid FPS: {p}'
    while True:
        ok,f=cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(cv2.resize(f,(256,256)),cv2.COLOR_BGR2RGB))
    cap.release()
    assert frames, f'Cannot read {p}'
    return np.stack(frames),fps

def make_input(frames):
    edges=np.linspace(0,len(frames),17)
    idx=[min(int((a+b)/2),len(frames)-1) for a,b in zip(edges[:-1],edges[1:])]
    x=frames[idx,16:240,16:240]
    x=(x.astype(np.float32)/255-MEAN)/STD
    return torch.from_numpy(np.ascontiguousarray(x)).permute(0,3,1,2),idx

@torch.inference_mode()
def probability(frames):
    x,_=make_input(frames)
    with torch.autocast('cuda',dtype=torch.float16):
        logits=model(pixel_values=x.unsqueeze(0).cuda()).logits
    return torch.softmax(logits.float(),-1)[0,1].item()
''')
code('''results=[]
for path in VIDEOS:
    frames,fps=read_video(path); n=len(frames)
    p=probability(frames)
    results.append(dict(video=path.name,mode='whole_clip',start_s=0,end_s=n/fps,prob=p,pred=int(p>=THR)))
    _,idx=make_input(frames)
    fig,axes=plt.subplots(4,4,figsize=(10,10))
    for ax,i in zip(axes.flat,idx):
        ax.imshow(frames[i,16:240,16:240]);ax.set_title(f'{i/fps:.3f}s');ax.axis('off')
    fig.suptitle(f'{path.name}: actual model input | P(fight)={p:.4f}')
    plt.tight_layout();plt.savefig(path.stem+'_input.jpg');plt.show()
    fig,ax=plt.subplots(figsize=(10,3))
    for seconds in [.5,1.0]:
        win=min(n,round(seconds*fps)); step=max(1,round(.1*fps))
        starts=list(range(0,n-win+1,step))
        if starts[-1]!=n-win: starts.append(n-win)
        probs=[]
        for start in starts:
            p=probability(frames[start:start+win]);probs.append(p)
            results.append(dict(video=path.name,mode=f'window_{seconds}s',start_s=start/fps,end_s=(start+win)/fps,prob=p,pred=int(p>=THR)))
        ax.plot([(s+win/2)/fps for s in starts],probs,'o-',label=f'{seconds}s window')
    ax.axhline(THR,ls='--',color='gray',label='old threshold .61')
    ax.set(xlabel='Window center (seconds)',ylabel='Model score',ylim=(0,1),title=path.name)
    ax.legend();plt.tight_layout();plt.savefig(path.stem+'_scores.png');plt.show()
df=pd.DataFrame(results)
df.to_csv('ko_predictions.csv',index=False)
display(df[df['mode']=='whole_clip'])
display(df.groupby(['video','mode']).agg(max_prob=('prob','max'),flagged_windows=('pred','sum'),windows=('pred','size')))
print('Check whether the strike is visible in sampled frames. A high window score alone does not prove correct localization.')
''')
code('''import zipfile
with zipfile.ZipFile('ko_test_results.zip','w') as z:
    z.write('ko_predictions.csv')
    for pattern in ['KO*_input.jpg','KO*_scores.png']:
        for p in Path('.').glob(pattern): z.write(p)
files.download('ko_test_results.zip')
''')
Path('test_KO_colab.ipynb').write_text(json.dumps(dict(cells=cells,metadata=dict(accelerator='GPU',kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared input contact sheets, metadata and test_KO_colab.ipynb')
