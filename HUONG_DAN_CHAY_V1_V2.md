# Hướng dẫn chạy V1, V2 và đánh giá phát hiện đánh nhau

Project có hai web server chạy độc lập trên Windows. **V1** là bản đầu trong `violence_project`. **V2** là bản huấn luyện trong `train_v2`, triển khai bằng `demo-web`. Cả hai phục vụ bài toán phát hiện đánh nhau trong video. Bạn có thể chạy web bằng checkpoint đã có, không cần train lại.

## 1. Chạy nhanh trong VS Code

Mở thư mục `D:\TaiLieu\2026-2027\ATTT` làm workspace. Mở **Terminal → New Terminal**, chọn PowerShell. Các lệnh bên dưới dùng trực tiếp Python của project, nên không cần kích hoạt môi trường trước.

| Thành phần | V1 | V2 |
|---|---|---|
| Lệnh khởi động | `.\.venv\Scripts\python.exe .\violence_project\web\server.py` | `.\.venv\Scripts\python.exe .\demo-web\demo_server.py` |
| Địa chỉ web | [V1 trên cổng 8878](http://127.0.0.1:8878/) | [V2 trên cổng 8877](http://127.0.0.1:8877/) |
| Checkpoint | `violence_project/checkpoints/best.pt` | `train_v2/runs/local_gpu_01/best.pt` |
| Web và database | `violence_project/web/recordings/events.sqlite` | `demo-web/recordings/events.sqlite` |
| Notebook | [Train V1](trainV1.ipynb) · [Demo V1 gốc trên Colab](demo1.ipynb) | [Train V2 trên VS Code](train_v2/train_v2_vscode.ipynb) · [Đánh giá V2](demo.ipynb) |

Để mở cả hai bản, chạy mỗi lệnh trong **một terminal riêng**. Đợi terminal in URL rồi mở bằng Edge hoặc Chrome. Nếu V1 báo `V1 is already running`, mở địa chỉ đang chạy. Để dừng, bấm **Dừng ghi** trên web cho clip đang thu được đóng, rồi nhấn **Ctrl+C** ở terminal tương ứng.

**Chạy V1 với bốn video KO để phát lại**. Dừng phiên V1 cũ trước khi đổi danh sách video.

```powershell
.\.venv\Scripts\python.exe .\violence_project\web\server.py --video KO.mp4 --video KO_1s.mp4 --video KO3_2s.mp4 --video KO4_dai.mp4
```

V1 quét những file được truyền qua `--video` khi khởi động. V2 đọc dự đoán phát lại từ `demo-web/recordings/windows.csv`; webcam V2 chạy suy luận mới. Khi chuyển project sang máy khác, kiểm tra cột `path` của CSV vì đường dẫn video có thể là đường dẫn tuyệt đối của máy cũ. V2 cũng đọc `KO_1s.mp4` để kiểm tra model lúc khởi động.

### Chuẩn bị trên máy Windows mới

Cần Python 3.11, FFmpeg trong PATH và hai bộ checkpoint ở đúng đường dẫn trên. Các lệnh sau khớp với môi trường Windows/NVIDIA của project:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install transformers==4.53.3 numpy pandas scikit-learn opencv-python-headless matplotlib pillow tqdm ipykernel matplotlib-inline
ffmpeg -version
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

V1 dùng cấu hình tại `violence_project/web/config.json`. V2 cần `train_v2/cache/hf/config.json`, `policy.json` cùng thư mục với `best.pt` và trọng số bộ phát hiện người `train_v2/cache/torch/hub/checkpoints/keypointrcnn_resnet50_fpn_coco-fc266e95.pth`. Train từ đầu còn cần trọng số nền VideoMAE trong cache hoặc mạng để tải. Server có thể chọn CPU khi thiếu CUDA; notebook train local yêu cầu GPU CUDA. Nếu thiếu VRAM trên RTX 3050 4 GB, chạy lần lượt từng server.

## 2. Dùng webcam và xem lại cảnh báo

1. Mở URL của server, chọn **Camera máy tính → Bật webcam** và cấp quyền camera. Hình xem trực tiếp lấy từ trình duyệt; các khung hình gửi tới server local để phân tích. Micro không được yêu cầu.
2. Chọn video KO trong danh sách nguồn để phát lại và xem điểm theo thời gian. Khi kiểm tra webcam, xem nhật ký của đúng nguồn camera. Log đã lưu nằm trong database riêng của từng phiên bản.
3. Mở **Xem đoạn đã cắt** để kiểm tra bằng chứng. **Lưu log** giữ sự kiện trong kho. **Không đánh nhau · Xóa đoạn** xóa MP4 cảnh báo và ẩn sự kiện khỏi danh sách; video nguồn vẫn còn.
4. Mỗi server có thư mục `recordings/` chứa MP4 và SQLite chứa thời gian, nguồn, khoảng thời gian trong nguồn, điểm, loại cảnh báo và trạng thái duyệt. Video bằng chứng được lưu không âm thanh.

Web lấy mẫu luồng ở nhịp mục tiêu 16 frame/giây, xét tối đa 16 frame gần nhất mỗi 0,5 giây. Khi có cảnh báo, bộ đệm cho phép giữ khoảng 3 giây trước cửa sổ phát hiện. Cảnh báo liên tiếp kéo dài cùng một clip. Sau khoảng 1 giây không có cảnh báo mới, hoặc khi dừng ghi, server đóng clip. Nhịp lấy mẫu là cấu hình; FPS thực tế còn phụ thuộc máy, webcam và thời gian suy luận.

## 3. V2 học gì từ dữ liệu?

V2 học có giám sát từ chuỗi ảnh RGB 2D và nhãn video. Chuỗi trước–trong–sau hành động cung cấp thông tin chuyển động. Với các sự kiện được chú thích chi tiết, V2 còn học đặc trưng khớp của hai người và nhãn tiếp xúc. Camera đơn vẫn có thể thấy hai người chồng lên nhau dù họ đứng ở hai độ sâu khác nhau; đây là lý do cần tách nhãn hành động khỏi nhãn tiếp xúc.

### Chia dữ liệu trước khi cắt cửa sổ

| Tập RLVS | Video gốc | Sự kiện chú thích chi tiết | Cửa sổ sự kiện bổ sung |
|---|---:|---:|---:|
| Train | 1.414 | 7 | 17 |
| Validation | 310 | 5 | 13 |
| Test | 287 | 5 | 12 |

17 sự kiện được đánh dấu trên các video RLVS đã có. 42 cửa sổ là các đoạn dẫn xuất nên không được tính thành 42 nguồn độc lập. Notebook đọc ba CSV trong [splits](train_v2/splits), gộp nguồn đã biết qua [source_groups.csv](train_v2/source_groups.csv), tính SHA-256 và kiểm tra đường dẫn, nguồn hoặc nội dung trùng giữa các tập. Những bản dựng khác nhau của cùng cảnh vẫn có thể lọt nếu chưa được nhận diện là cùng nguồn.

### Gán nhãn theo sự kiện

[events.csv](train_v2/events.csv) là đầu vào cho các mẫu có chú thích thời gian.

| Cột | Ý nghĩa |
|---|---|
| `path, source_id, split` | Video gốc, nhóm nguồn, tập train/val/test. Các đoạn cùng nguồn phải ở cùng tập. |
| `scenario` | Tình huống như `fast_strike`, `arm_overlap`, `friendly_touch`, `hand_gesture`, `grappling`. |
| `action` | 0 = bình thường, 1 = đánh nhau. |
| `contact` | `touch`, `no_touch`, `unclear`. Ô trống là thiếu nhãn và được bỏ qua khi tính loss tiếp xúc. |
| `start_s, end_s` | Khoảng thời gian mà chú thích còn đúng, tính bằng giây trong video gốc. |
| `event_s, min_window_s` | Mốc tương tác và thời lượng ngắn nhất được phép lấy quanh mốc đó. |

**Ví dụ vung tay che người:** nếu ngữ cảnh cho thấy người phía trước đang làm việc bình thường nhưng bàn tay khuất đúng lúc đi ngang người sau, gán `action=0, contact=unclear`. Chỉ gán `no_touch` khi quan sát được tay đi hụt. Một cái chạm thân thiện có thể mang `action=0, contact=touch`. Với cú đánh nhanh, đặt `event_s` ở mốc ra đòn hoặc chạm quan sát được. Cách gán này cho phép model học cả “chạm nhưng bình thường” và “đánh nhưng điểm chạm bị che”.

### Những video đã được đánh dấu

**RLVS — 17 sự kiện có nhãn chi tiết.** Tên dưới đây là tên file gốc (đuôi `.mp4`). Chỉ 7 sự kiện ở cột Train được dùng để cập nhật trọng số. Hai cột còn lại phục vụ chọn mô hình và đánh giá. Mốc bắt đầu, kết thúc, mốc tương tác và nhãn tiếp xúc từng video có trong [events.csv](train_v2/events.csv); có thể xem nhanh các frame quanh mốc ở [ảnh đối chiếu](train_v2/review_events.jpg).

| Nhãn tình huống | Nhãn hành động | Train | Validation | Test |
|---|---|---|---|---|
| `fast_strike` — cú đánh nhanh | 1 | V_301, V_241, V_741 | V_521 | V_101 |
| `grappling` — giằng co | 1 | — | V_821 | V_161 |
| `arm_overlap` — tay che người trong ảnh | 0 | NV_46 | NV_826 | NV_575 |
| `body_overlap` — hai thân người che nhau | 0 | — | — | NV_571 |
| `friendly_touch` — chạm thân thiện | 0 | NV_1, NV_104 | NV_796 | — |
| `hand_gesture` — cử chỉ tay bình thường | 0 | NV_241 | NV_830 | NV_706 |

Ví dụ, `V_301` có nhãn `fast_strike / touch` ở **1,94 giây**, còn `V_241` có nhãn `fast_strike / unclear` ở **1,11 giây**. Với `NV_575`, hành động được gán bình thường nhưng điểm tay chạm người bị che ở **2,36 giây**, nên nhãn tiếp xúc là `unclear`. Các ví dụ này giải thích vì sao V2 học hai đầu ra riêng.

**RWF-2000 — 6 đoạn đã đánh dấu để duyệt cho lần fine-tune sau.** Cả sáu lấy từ `train/NonFight`, có nhãn hành động bình thường. Các mốc dưới đây tính bằng giây từ đầu **video gốc**. Năm đoạn đầu là ứng viên bổ sung; checkpoint V2 hiện tại chưa được huấn luyện lại với chúng.

| Video gốc | Đoạn đã chọn (s) | Mốc (s) | Nhãn quan sát | Nhãn tiếp xúc |
|---|---:|---:|---|---|
| [5CwYc67y_0.avi](dataset/RWF-2000/train/NonFight/5CwYc67y_0.avi) | 0,35–3,55 | 1,75 | `arm_overlap` — người phía trước giơ tay ngang mặt người phía sau tại quầy | `unclear` |
| [cqP8OyUh_0.avi](dataset/RWF-2000/train/NonFight/cqP8OyUh_0.avi) | 1,55–3,95 | 3,00 | `object_overlap` — nhân viên nâng khay, tay và khay che người bên cạnh | `unclear` |
| [DjLWv5mr_0.avi](dataset/RWF-2000/train/NonFight/DjLWv5mr_0.avi) | 1,65–3,90 | 2,35 | `hand_gesture` — đưa tay và vật qua quầy | `unclear` |
| [zZHTBiER_0.avi](dataset/RWF-2000/train/NonFight/zZHTBiER_0.avi) | 0,70–3,95 | 2,50 | `hand_gesture` — hai nhân viên đứng gần nhau, người trước vươn tay làm việc | `unclear` |
| [tPpXT9VI_0.avi](dataset/RWF-2000/train/NonFight/tPpXT9VI_0.avi) | 1,15–3,90 | 2,35 | `hand_gesture` — người đứng xoay và giơ tay trước bàn làm việc | Để trống |
| [ev8va5uR_0.avi](dataset/RWF-2000/train/NonFight/ev8va5uR_0.avi) | 0,35–3,90 | 2,10 | `sports_background` — góc rộng thấy sân và đường chạy | Để trống |

Đoạn sân thể thao được giữ riêng vì người trong hình quá nhỏ để gán nhãn động tác. Khi bổ sung dữ liệu nhằm giảm báo nhầm trên Hockey, vẫn cần các cảnh vận động viên thực sự tranh bóng hoặc va chạm bình thường. Các liên kết trong bảng mở video nguồn vì thư mục clip cắt `train_v2/rwf_review` của lần duyệt trước hiện không có trong workspace.

### Từ video đến tensor

- Mẫu nền `base` lấy 16 frame trải trên toàn clip và dùng nhãn hành động cấp video. Mẫu sự kiện tạo các cửa sổ quanh mốc ở nhiều độ dài **0,5 / 1 / 3 giây**, giới hạn theo khoảng chú thích và `min_window_s`. Cửa sổ ngắn tăng cơ hội thấy cú đánh chỉ diễn ra trong vài frame.
- Chia cửa sổ thành 16 khoảng. Khi train, lấy ngẫu nhiên một frame trong mỗi khoảng và dịch vị trí sự kiện trong cửa sổ. Khi đánh giá, lấy giữa mỗi khoảng để kết quả lặp lại được.
- Đổi BGR sang RGB, resize giữ tỉ lệ, thêm viền để đủ **224 × 224**, chuẩn hóa theo ImageNet. Khi lật ngang, notebook cập nhật cả tọa độ và thứ tự khớp trái/phải. Ngoài ra còn hoán đổi thứ tự hai người khi train.
- Keypoint R-CNN trích khớp trên frame gốc và lưu cache theo mã video, chỉ số frame. Mỗi người dùng mũi, hai vai, hai khuỷu, hai cổ tay, hai hông. Mỗi khớp có tọa độ chuẩn hóa và cờ quan sát được. Dãy pose của một cửa sổ có kích thước **16 × 54**.
- Dùng vai để ghép danh tính giữa các frame. Frame thiếu cặp người đáng tin được biểu diễn bằng số 0. Các clip nền cũng dùng pose bằng 0. Loss tiếp xúc bỏ qua mẫu nếu 16 frame đã lấy không có frame đủ gần `event_s` (sai khác quá khoảng 1,1 frame).

## 4. Kiến trúc, kỹ thuật train và cách chọn best.pt

| Thành phần | Cách V2 sử dụng |
|---|---|
| Nhánh video | VideoMAE-base khởi tạo từ Kinetics; Transformer xử lý các vùng ảnh qua trục thời gian. Đầu phân loại hành động cho hai logits. |
| Nhánh tư thế | GRU hai chiều, hidden size 64 mỗi chiều, mã hóa chuỗi khớp của cặp người. Model xét cả hai thứ tự người và lấy trung bình để giảm phụ thuộc thứ tự. |
| Kết hợp | Pose đủ chất lượng mới hiệu chỉnh logits hành động. Nhánh tiếp xúc ghép embedding video và pose để dự đoán `no_touch / touch / unclear`. |
| Hàm mất mát | Tổng cross-entropy của hai đầu ra, mỗi đầu chỉ dùng nhãn hợp lệ. Nhờ vậy clip chỉ có nhãn đánh nhau vẫn tham gia train. |
| Lấy mẫu | Mỗi epoch duyệt clip nền một lần, lặp mỗi cửa sổ sự kiện train ba lần rồi xáo trộn. Điều này tăng tần suất các tình huống hiếm nhưng không tạo thêm nguồn mới. |
| Tối ưu trên máy local | 5 epoch, seed 42, batch 1, tích lũy gradient 16 lần. Đóng băng backbone trừ block cuối, cập nhật các đầu ra và nhánh pose. AdamW dùng LR `2e-5` cho block video và `2e-4` cho phần còn lại, weight decay `0.05`. |
| Ổn định và bộ nhớ | Warmup khoảng 10% số bước rồi giảm LR theo cosine; mixed precision FP16, GradScaler và cắt chuẩn gradient ở 1.0. Bộ trích pose chạy CPU trong notebook local. |

Sau mỗi epoch, notebook tính F1 hành động và macro-F1 tiếp xúc trên validation. Khi đủ ba lớp tiếp xúc, điểm chọn checkpoint là trung bình hai F1; nếu thiếu lớp thì dùng F1 hành động. Chỉ thay `best.pt` khi điểm tăng. Lần chạy hiện tại chọn **epoch 2**: action F1 **0,9900**, contact macro-F1 **0,7079**, điểm chọn **0,8490**. Epoch 3–5 bằng điểm nên giữ checkpoint epoch 2. [Lịch sử validation](train_v2/runs/local_gpu_01/history.json)

### Chạy lại notebook train

1. Mở [train_v2_vscode.ipynb](train_v2/train_v2_vscode.ipynb), chọn kernel Python 3.11 của `.venv`. Kiểm tra ô đầu in đúng GPU và phép tính CUDA thành công.
2. Kiểm tra `DATA_ROOT`, `LABEL_DIR` và đặt **`OUT` thành một thư mục run mới**. Notebook chủ động dừng nếu thư mục đó đã có `best.pt`. Khởi tạo mặc định dùng trọng số Kinetics; trọng số V1 có nguy cơ đã học video nay thuộc test V2 vì split khác nhau.
3. Chạy **Run All**. Trước epoch đầu, kiểm tra bảng split, ảnh mẫu và `pose_preflight.csv`. Sau train, xem `history.json`, `best.pt`, `policy.json`, `test_predictions.csv` và `test_summary.csv` trong `OUT`.
4. Server hiện trỏ đến `RUN = train_v2/runs/local_gpu_01` trong [demo_server.py](demo-web/demo_server.py). Muốn thử run mới, trỏ `RUN` tới đúng thư mục chứa **cả checkpoint lẫn policy của cùng lần train**, rồi khởi động lại V2. Phần phát lại KO trên web vẫn dùng `recordings/windows.csv`; nếu đổi model, cũng cần tạo lại dự đoán KO tương ứng trước khi so sánh.
5. Nếu dùng Colab, mở [train_v2_colab.ipynb](train_v2/train_v2_colab.ipynb) và cấu hình đường dẫn Drive theo [hướng dẫn train](train_v2/README.md). Cấu hình fine-tune của Colab và bản local 4 GB có thể cho kết quả khác nhau.

## 5. Phân biệt đánh giá trong notebook với quyết định trên webcam

| Chế độ | Quy trình ra quyết định |
|---|---|
| V1 webcam | Resize 256 × 256, crop giữa 224 × 224, lấy 16 frame, tính điểm hành động. Điểm từ 0,61 phát cảnh báo. |
| V2 notebook, clip nền | 16 frame letterbox trên cả video, pose bằng 0. Điểm hành động dùng để tính các chỉ số RLVS/Hockey theo clip. |
| V2 notebook, sự kiện/KO | Quét nhiều độ dài, dùng thông tin pose khi có; đối với KO bước trượt mặc định 0,2 giây. Chính sách có thể xuất `normal`, `suspected` hoặc `fight`. |
| V2 webcam | Chạy nhánh video trên cửa sổ gần nhất. Khi điểm từ **0,5**, chạy bộ phát hiện người ở frame giữa và frame cuối. Ít nhất một trong hai frame phải có **từ hai người trở lên** mới tạo cảnh báo nghi ngờ. Thiếu cặp thì trả `no_pair` và không mở clip mới. |

Chính sách hiện tại có `contact_ready=false`. Điều kiện bật nhánh xác nhận yêu cầu đủ ba lớp tiếp xúc trong dữ liệu train đã thấy và các mẫu validation có pose đạt chất lượng. Vì vậy, cảnh báo hiện tại có nghĩa **cần xem lại**. Kiểm tra hai người chỉ là điều kiện ghi log trên webcam: hai người đứng sát nhau vẫn có thể gây báo nhầm, và che khuất có thể khiến bộ đếm bỏ sót người thứ hai. Webcam dùng bộ phát hiện người riêng ở bước này; không chạy đủ GRU tiếp xúc như notebook quét KO. [Policy của run hiện tại](train_v2/runs/local_gpu_01/policy.json)

## 6. Kết quả hiện có và cách đọc

Các số sau thuộc **V2**, lớp dương là đánh nhau. Mỗi hàng tính trên một dự đoán hành động cho mỗi clip. Chúng không đo trực tiếp hiệu quả của điều kiện hai người trên webcam.

| Tập | Video | Báo nhầm | Bỏ sót | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RLVS test | 287 | 2 | 2 | 98,61% | 98,61% | 98,61% | 98,61% |
| Hockey Fights | 1.000 | 480 | 2 | 51,80% | 50,92% | 99,60% | 67,39% |
| RWF-2000, split `val` | 400 | 31 | 21 | 87,00% | 85,24% | 89,50% | 87,32% |

Hockey có 500 clip bình thường, 480 clip vượt ngưỡng: tỷ lệ báo nhầm **96%**. Đây là hạn chế lớn khi chuyển bối cảnh. Với tình huống che tay trong test chi tiết, 0/2 cửa sổ bị báo nhầm, nhưng cả hai đến từ **một nguồn**. Cú đánh nhanh có 3/3 cửa sổ được gắn nghi ngờ, cũng chỉ từ một nguồn. Nên báo cả số nguồn khi trình bày các kết quả này. [Tổng hợp theo tình huống](train_v2/runs/local_gpu_01/test_summary.csv) · [Dự đoán RWF](demo-web/rwf_eval.csv)

Mở [demo.ipynb](demo.ipynb) để xem nhãn thật, nhãn cảnh báo, confusion matrix, ROC, đồ thị theo thời gian và video minh họa. Cả bốn clip KO từng có cửa sổ vượt ngưỡng; riêng `KO.mp4` có 108/125 cửa sổ nghi ngờ, điểm cao nhất 88,87%. Nhãn thật của các clip KO là nhãn cấp video theo mô tả của người dùng. Số cửa sổ vượt ngưỡng không phải số lần đánh hoặc số lần tiếp xúc.

## 7. So sánh hai bản bằng dữ liệu thực tế của bạn

Dùng cùng các file quay từ camera triển khai. Đặt mỗi clip vào thư mục theo **nhãn thật**:

```text
test_thuc_te/
  normal/   # Đi lại, vung tay che người, chạm thân thiện, hoạt động thể thao
  fight/    # Clip có hành động đánh nhau, gồm cả sự kiện ngắn
```

Chạy ở thư mục ATTT:

```powershell
.\.venv\Scripts\python.exe .\demo-web\compare_servers.py .\test_thuc_te --output .\ket_qua_so_sanh.csv
```

Script đọc cả thư mục con, hỗ trợ MP4, AVI, MOV và MKV. Không cần mở web. Hai bản lần lượt nhận cùng video, cùng cách mô phỏng luồng 16 FPS và cùng bước xét nửa giây. Một video tính là có cảnh báo nếu ít nhất một cửa sổ kích hoạt. V2 có thêm bước kiểm tra hai người; phép đo này so **hai pipeline triển khai**.

- Terminal in TP, FP, FN, TN, Precision, Recall và F1. CSV chứa `truth`, `v1_alert`, `v2_alert` và điểm đỉnh của từng bản. Với V2, điểm đỉnh có thể cao nhưng `v2_alert=0` vì thiếu cặp người.
- Đọc riêng các dòng `truth=0, alert=1` (báo nhầm) và `truth=1, alert=0` (bỏ sót). So sánh hành vi một người, hai người che nhau và cú đánh nhanh. Quyết định bản nào phù hợp dựa trên mức bỏ sót và số cảnh báo nhân viên có thể xem lại.
- Để đo vận hành liên tục, ghi thêm số sự kiện báo nhầm mỗi giờ, thời điểm sự kiện thật và thời điểm web báo. Script trên đo theo clip; độ trễ, FPS camera thực tế và tỷ lệ báo nhầm mỗi giờ cần một phép đo riêng.
- Khi bổ sung dữ liệu để fine-tune, giữ các đoạn từ cùng video/camera-sự kiện trong một split. Chọn ngưỡng bằng validation và giữ test riêng cho lần đánh giá cuối. Tên thư mục `NonFight` đủ cho nhãn hành động sơ bộ; nhãn chi tiết như `arm_overlap` hoặc `no_touch` cần xem lại đoạn cụ thể.

## 8. Xử lý lỗi thường gặp

| Hiện tượng | Cách kiểm tra |
|---|---|
| `WinError 10048` | Mở URL tương ứng để xem server đang chạy. Dùng `netstat -ano \| findstr :8878` hoặc cổng 8877 để kiểm tra. Muốn đổi tham số, dừng đúng terminal của phiên cũ bằng Ctrl+C. |
| Webcam bị chặn | Mở đúng URL localhost thay vì chạy trực tiếp HTML. Cho phép Camera trong quyền trang web. Nếu cần, kiểm tra bằng ứng dụng Camera của Windows rồi đóng ứng dụng trước khi thử lại. |
| Thiếu `best.pt` hoặc sai kiến trúc | So đường dẫn với bảng ở mục 1. V1 lưu state dict VideoMAE; V2 lưu cả hai nhánh trong trường `model`. Dùng đúng server cho định dạng tương ứng. |
| Thiếu FFmpeg hoặc không phát được clip | Kiểm tra `ffmpeg -version`, quyền ghi thư mục `recordings`, sự tồn tại của video nguồn và cột `path` trong CSV. |
| CUDA hết bộ nhớ | Dừng server/terminal dùng GPU chưa cần, rồi chạy lại. Khi train trên laptop, dùng notebook local batch 1. |
| Thấy điểm cao nhưng không có log V2 | Xem trạng thái đếm người và nguồn đang chọn. Điểm cao một mình chưa đủ điều kiện mở clip từ webcam. |

