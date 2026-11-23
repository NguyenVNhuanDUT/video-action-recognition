# Hướng dẫn cài đặt và chạy dự án phát hiện đánh nhau

Hướng dẫn dành cho **Windows, PowerShell và VS Code**. Dự án có hai phiên bản V1 và V2; đã có checkpoint thì không cần huấn luyện lại để chạy web.

## 1. Chuẩn bị

- Cài Python **3.11** và kiểm tra bằng `py -3.11 --version`.
- Cài FFmpeg, thêm thư mục `bin` của FFmpeg vào biến môi trường `PATH`, sau đó mở lại terminal. Kiểm tra bằng `ffmpeg -version`.
- Dùng Edge hoặc Chrome để mở giao diện và cấp quyền webcam.
- GPU NVIDIA có driver tương thích giúp chạy nhanh hơn; server tự chọn CPU nếu CUDA không khả dụng. CPU có thể xử lý chậm.

Mở thư mục dự án trong VS Code, chọn **Terminal → New Terminal → PowerShell**. Tất cả lệnh dưới đây chạy tại thư mục gốc `ATTT`:

```powershell
Set-Location 'D:\TaiLieu\2026-2027\ATTT'
```

Nếu chép dự án sang nơi khác, thay đường dẫn trên bằng thư mục mới. Không chép môi trường `.venv` từ máy cũ để sử dụng trực tiếp; hãy tạo lại trên máy mới.

## 2. Cài môi trường Python

Nếu dùng máy hiện tại và `.venv` đã chạy được, chuyển đến bước kiểm tra cuối mục này. Trên máy mới, tạo môi trường:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

**Chọn một trong hai cách cài PyTorch sau.** Các phiên bản bên dưới khớp với PyTorch và Transformers đang có trong dự án.

Máy có GPU NVIDIA, dùng bản CUDA 12.4:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

Máy chạy CPU:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

Cài thư viện phục vụ hai web server:

```powershell
.\.venv\Scripts\python.exe -m pip install transformers==4.53.3 numpy pandas opencv-python-headless pillow
```

Nếu cần chạy notebook hoặc huấn luyện, cài thêm:

```powershell
.\.venv\Scripts\python.exe -m pip install scikit-learn matplotlib tqdm ipykernel matplotlib-inline
```

Kiểm tra môi trường:

```powershell
.\.venv\Scripts\python.exe -c "import torch, torchvision, transformers, cv2, numpy, pandas; print('Import OK'); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
ffmpeg -version
```

`CUDA: False` là bình thường nếu cài bản CPU. Các lệnh gọi thẳng Python trong `.venv`, nên không cần chạy `Activate.ps1` hoặc thay đổi execution policy.

## 3. Kiểm tra các tệp cần thiết

| Phiên bản | Tệp bắt buộc |
|---|---|
| V1 | `violence_project/checkpoints/best.pt`, `violence_project/web/config.json` |
| V2 | `train_v2/runs/local_gpu_01/best.pt`, `train_v2/runs/local_gpu_01/policy.json`, `train_v2/cache/hf/config.json` |
| V2 – bộ phát hiện người | `train_v2/cache/torch/hub/checkpoints/keypointrcnn_resnet50_fpn_coco-fc266e95.pth` |
| V2 – dữ liệu khởi động | `KO_1s.mp4`, `demo-web/recordings/windows.csv` và các video được tham chiếu trong cột `path` của CSV |

Các checkpoint và cấu hình mô hình trong bảng đã có ở thư mục dự án tại thời điểm viết hướng dẫn. Khi chuyển máy, cần chép đầy đủ các tệp này, mã nguồn và giao diện web. Cài thư viện Python không tự tạo checkpoint.

**Lưu ý khi chuyển máy:** `demo-web/recordings/windows.csv` đang chứa đường dẫn tuyệt đối dạng `D:\TaiLieu\2026-2027\ATTT\KO_1s.mp4`. Mở CSV và sửa cột `path` cho đúng vị trí video trên máy mới. Giữ nguyên các cột và kết quả dự đoán còn lại. Chỉ đổi đường dẫn tới đúng video cũ; nếu thay bằng video khác thì cần tính lại dự đoán.

## 4. Chạy V1

Trong terminal tại thư mục gốc dự án:

```powershell
.\.venv\Scripts\python.exe .\violence_project\web\server.py
```

Đợi server in địa chỉ rồi mở **http://127.0.0.1:8878/**.

Muốn xem lại bốn video mẫu, dừng phiên V1 đang chạy bằng **Ctrl+C**, sau đó chạy:

```powershell
.\.venv\Scripts\python.exe .\violence_project\web\server.py --video .\KO.mp4 --video .\KO_1s.mp4 --video .\KO3_2s.mp4 --video .\KO4_dai.mp4
```

V1 phân tích các video trước khi mở giao diện nên có thể phải chờ. Có thể truyền một hoặc nhiều `--video`. Nếu cần đổi cổng:

```powershell
.\.venv\Scripts\python.exe .\violence_project\web\server.py --port 8888
```

Khi đó mở `http://127.0.0.1:8888/`.

## 5. Chạy V2

Trong terminal tại thư mục gốc dự án:

```powershell
.\.venv\Scripts\python.exe .\demo-web\demo_server.py
```

Đợi dòng `Model check:` và địa chỉ server, sau đó mở **http://127.0.0.1:8877/**.

V2 kiểm tra mô hình bằng `KO_1s.mp4` khi khởi động. Phần phát lại video sử dụng kết quả đã lưu trong `demo-web/recordings/windows.csv`; webcam gửi khung hình lên server để dự đoán mới. V2 hiện yêu cầu phát hiện ít nhất hai người khi điểm hành động vượt ngưỡng để tạo cảnh báo webcam.

Muốn chạy cả hai bản, dùng **hai terminal riêng**. Nếu GPU thiếu bộ nhớ, chạy lần lượt từng bản.

## 6. Sử dụng và dừng chương trình

1. Mở địa chỉ web của phiên bản vừa chạy.
2. Chọn **Camera máy tính → Bật webcam**, cho phép trình duyệt truy cập camera.
3. Quan sát trạng thái, điểm dự đoán và danh sách cảnh báo. Chọn video mẫu trong danh sách nguồn nếu muốn xem phát lại.
4. Mở **Xem đoạn đã cắt** để xem clip cảnh báo. Chọn **Lưu log** để giữ sự kiện; **Không đánh nhau · Xóa đoạn** sẽ xóa clip cảnh báo và ẩn sự kiện khỏi danh sách.
5. Khi kết thúc, bấm **Dừng ghi** trên web trước, sau đó nhấn **Ctrl+C** trong terminal chạy server.

Đầu ra được lưu riêng cho từng bản:

| Phiên bản | Thư mục clip | Cơ sở dữ liệu sự kiện |
|---|---|---|
| V1 | `violence_project/web/recordings/` | `violence_project/web/recordings/events.sqlite` |
| V2 | `demo-web/recordings/` | `demo-web/recordings/events.sqlite` |

Clip bằng chứng không có âm thanh. Chạy web qua địa chỉ server, không mở trực tiếp tệp HTML.

## 7. Xử lý lỗi thường gặp

| Lỗi hoặc hiện tượng | Cách xử lý |
|---|---|
| Không nhận lệnh `py` | Kiểm tra đã cài Python 3.11 cùng Python Launcher, rồi mở lại terminal. |
| `ModuleNotFoundError` | Cài thư viện bằng đúng `.\.venv\Scripts\python.exe -m pip` và chạy server bằng Python trong `.venv`. |
| Không nhận lệnh `ffmpeg` | Thêm thư mục chứa `ffmpeg.exe` vào `PATH`, mở lại terminal rồi kiểm tra `ffmpeg -version`. |
| Thiếu `best.pt`, `policy.json` hoặc trọng số pose | Đối chiếu mục 3, chép lại đúng tệp vào đúng thư mục. Checkpoint V1 và V2 có định dạng khác nhau. |
| V2 lỗi ngay ở `Model check` | Kiểm tra `KO_1s.mp4` tồn tại, đọc được và không bị rỗng; đồng thời kiểm tra checkpoint và cấu hình. |
| V2 lỗi đường dẫn hoặc không phát được video mẫu | Kiểm tra mọi đường dẫn trong cột `path` của `windows.csv` và sự tồn tại của video tương ứng. |
| `WinError 10048` hoặc báo đã chạy | Mở URL để kiểm tra phiên đang chạy; muốn chạy lại với tham số mới thì dừng terminal cũ bằng Ctrl+C. V1 hỗ trợ đổi cổng bằng `--port`. |
| Webcam không lên | Cho phép camera trong trình duyệt và cài đặt quyền riêng tư của Windows; đóng ứng dụng khác đang sử dụng camera. |
| CUDA hết bộ nhớ | Dừng server hoặc notebook khác đang dùng GPU, rồi chạy lại từng server. |
| Điểm V2 cao nhưng không có cảnh báo webcam | Kiểm tra trạng thái phát hiện hai người và nguồn đang xem; điểm hành động cao chưa đủ để mở clip. |

## 8. Huấn luyện lại nếu cần

Để chạy demo với checkpoint hiện có, có thể bỏ qua mục này.

- V1: mở `trainV1.ipynb` và làm theo các ô trong notebook.
- V2 trên Windows: mở `train_v2/train_v2_vscode.ipynb`, chọn kernel `.venv`, chạy ô kiểm tra CUDA rồi các ô tiếp theo. Notebook huấn luyện local yêu cầu GPU CUDA.
- V2 trên Colab: xem `train_v2/README.md` và notebook `train_v2/train_v2_colab.ipynb`.

Huấn luyện cần dữ liệu và trọng số nền tương ứng; lần tải trọng số đầu tiên cần mạng nếu cache chưa có. Xem thêm [hướng dẫn chạy và đánh giá V1/V2](HUONG_DAN_CHAY_V1_V2.md) để biết quy trình chi tiết.
