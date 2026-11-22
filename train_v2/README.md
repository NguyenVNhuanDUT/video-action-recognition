# Huấn luyện V2 trên Colab

### Chạy trên Windows bằng VS Code

Mở thư mục ATTT trong VS Code, cài extension Python và Jupyter. Chạy trong terminal PowerShell ở thư mục ATTT:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install ipykernel "transformers>=4.45,<5" numpy pandas scikit-learn opencv-python-headless matplotlib tqdm
```

Mở `train_v2_vscode.ipynb`, chọn **Select Kernel → Python Environments → .venv**. Ô đầu in tên GPU, VRAM và chạy phép tính CUDA. Sau khi ô này chạy thành công, chạy các ô còn lại. Lần đầu cần mạng để tải trọng số. Notebook đọc video và CSV trực tiếp từ project.

Cho RTX 3050 Laptop 4 GB, bản local dùng batch 1, tích lũy 16 lần, chỉ cập nhật block cuối của VideoMAE cùng các đầu ra. Bộ ước lượng tư thế chạy CPU. Cấu hình này làm kết quả huấn luyện có thể khác bản Colab fine-tune toàn bộ. Model tốt nhất được ghi vào `train_v2/runs/local_gpu_01/best.pt`. Nếu GPU còn ít VRAM, đóng ứng dụng đang dùng GPU trước khi chạy.

### Chạy trên Colab

Mở **train_v2_colab.ipynb**, chọn GPU rồi chạy các ô theo thứ tự. Notebook mang theo bản chụp của `events.csv`, `source_groups.csv` và ba CSV trong `splits/`. Với dữ liệu Drive như bản V1, ô đầu vào tìm RLVS trong `/content/data`, trong thư mục dataset trên Drive hoặc trong các file `dataset-*.zip` dưới `MyDrive/lab_ATvsBMTT`. Nếu dữ liệu ở nơi khác, chỉnh DATA_ROOT hoặc giải nén để đường dẫn `DATA_ROOT/dataset/Real Life Violence Dataset` tồn tại.

| Tập | Video RLVS | Sự kiện đã xem | Nguồn của các sự kiện | Cửa sổ bổ sung |
|---|---:|---:|---:|---:|
| Train | 1.414 | 7 | 6 | 17 |
| Validation | 310 | 5 | 4 | 13 |
| Test | 287 | 5 | 4 | 12 |

Các sự kiện là chú thích chi tiết trên chính 17 video RLVS trong bảng. Ảnh `review_events.jpg` cho thấy ba frame quanh từng mốc. Xem video gốc nếu cần kiểm tra lại một nhãn. Trong nhóm arm_overlap, cả ba split đều có một ví dụ. Ở mẫu test NV_575, hai cầu thủ tranh bóng, hành động mang nhãn 0 còn tiếp xúc tay mang nhãn unclear vì bị che. Các ví dụ cú đánh, chạm thân thiện và giơ tay bình thường cũng được ghi riêng. Nhãn sự kiện chỉ mô tả những gì quan sát được.

Bản chia tập này giữ cùng tập cho 147 đường dẫn được ánh xạ trong source_groups.csv. Các nhóm được lập từ việc đối chiếu cảnh và tìm clip gần trùng bằng dấu vân tay ở ba thời điểm. Notebook còn kiểm tra SHA-256 của video và đối chiếu nguồn giữa các split. Vẫn có thể còn những bản cắt khác cảnh hoặc khác góc từ một nguồn chưa được nhận ra. Các clip KO ngoài RLVS được giữ cho bước thử sau khi train.

Mỗi epoch duyệt các clip nền một lần và lặp mỗi cửa sổ sự kiện ba lần. Với bảy sự kiện train, đây là bộ dữ liệu thử quy trình. Chỉ số trên một hoặc hai nguồn cho mỗi tình huống có thể đổi nhiều vì một lỗi. Đặc biệt, hãy xem cả false_logs và missed trong test_summary.csv.

### Khi chạy

- Mặc định dùng trọng số Kinetics để khởi tạo. Split V2 đã được chia lại nên trọng số V1 có thể từng học một số video nay thuộc test.
- Ô xem mẫu hiển thị 16 frame và các khớp của từng người. Ô kiểm tra GPU trước epoch đầu chạy thử hai nhánh và ghi `pose_preflight.csv` cho các cửa sổ sự kiện. Cảnh quá đông hoặc che nhiều có thể không theo dõi được cả hai người.
- Khi hoàn thành, `OUT` có `best.pt`, `policy.json`, lịch sử và các bảng dự đoán. Đầu ra mặc định ở `MyDrive/train_v2_contact_prepared_run1`. Chọn tên mới cho lần chạy tiếp theo.
- Muốn mở rộng dữ liệu, đặt CSV đã cập nhật trên Drive rồi chỉnh LABEL_DIR. Với video đã có trong RLVS, giữ split theo các CSV đã đóng gói. Đặt ba clip KO vào `MyDrive/train_v2_ko` để ô cuối tự tìm.

Đã kiểm tra cục bộ việc dựng manifest, các split theo nguồn đã biết và SHA-256, cùng khả năng giải mã 16 frame của cả 42 cửa sổ ở chế độ đánh giá. Các phép đo pose, loss và hiệu quả nhận diện cần chạy trên môi trường train.
