# Lý thuyết và hỏi đáp bảo vệ mô hình V2

## Mở đầu trong 60 giây

> V2 phát hiện đoạn video có dấu hiệu đánh nhau. Em dùng VideoMAE để học diễn biến từ chuỗi ảnh RGB, kết hợp GRU hai chiều để học chuyển động khớp của hai người ở các đoạn được chú thích. Em bổ sung nhãn cho 17 sự kiện, trong đó có cú đánh nhanh, vung tay che người, chạm thân thiện và giằng co. Quanh mốc sự kiện, em lấy cửa sổ 0,5, 1 và 3 giây nếu khoảng chú thích cho phép. Mỗi cửa sổ lấy 16 frame. Mô hình học hai đầu ra, hành động đánh nhau và khả năng quan sát tiếp xúc. Trên 287 clip nền của RLVS test, nhánh hành động đạt 98,61% accuracy. Khi chạy webcam, web dùng điểm hành động và kiểm tra số người để tạo cảnh báo nghi ngờ cho người quan sát xem lại.

## 1. Quy trình xử lý dữ liệu

### Chia tập theo nguồn và kiểm tra rò rỉ dữ liệu

V2 dùng **Real Life Violence Dataset (RLVS)** với 2.011 video. Mỗi video gốc có nhãn hành động cấp clip, `0` là bình thường và `1` là đánh nhau. Trong dữ liệu này có thêm 17 sự kiện được chú thích thủ công theo thời gian.

| Tập | Clip gốc RLVS | Sự kiện chú thích | Cửa sổ bổ sung |
|---|---:|---:|---:|
| Train | 1.414 | 7 | 17 |
| Validation | 310 | 5 | 13 |
| Test | 287 | 5 | 12 |

**Kỹ thuật chia theo nhóm nguồn (*group-based split*)** giữ những clip đã xác định cùng cảnh hoặc cùng nguồn trong một tập. Ví dụ, `NV_46` và `NV_104` cùng nhóm `football_red_blue` nên đều thuộc train. Notebook đọc danh sách split đã chuẩn bị, đối chiếu `source_id`, chuẩn hóa đường dẫn và tính **SHA-256**. Nếu cùng nhóm nguồn, cùng đường dẫn hoặc cùng mã băm xuất hiện ở nhiều tập, notebook dừng. SHA-256 phát hiện file giống hệt nhau. Bản cắt hoặc bản nén lại cần nhận diện thêm bằng nhóm nguồn. Sau bước này mới tạo cửa sổ, nên các cửa sổ chồng lấn từ một sự kiện đi theo split của video gốc. [Nhóm nguồn](train_v2/source_groups.csv) · [Notebook train](train_v2/train_v2_vscode.ipynb)

Khi báo cáo *data leakage*, phân biệt ba mục đích. **Train** cập nhật trọng số, **validation** chọn checkpoint và ngưỡng, **test** đo hiệu quả cuối. Cả 42 cửa sổ bổ sung chỉ là các đoạn dẫn xuất của 17 sự kiện, vì vậy khi báo cáo phải nêu cả số cửa sổ lẫn số nguồn.

### Gán nhãn sự kiện và tạo đầu vào

**Gán nhãn sự kiện theo thời gian (*temporal event annotation*)** ghi `start_s` và `end_s` cho khoảng đã xem, `event_s` cho mốc tương tác và `min_window_s` cho độ dài ngắn nhất giữ được ngữ cảnh. Nhãn **hành động** mô tả tình huống. Nhãn **tiếp xúc** mô tả điểm chạm quan sát được, gồm `touch`, `no_touch` và `unclear`. Ô tiếp xúc bỏ trống nghĩa là thiếu nhãn để tính loss, còn `unclear` là một lớp riêng cho tình huống đã xem nhưng điểm chạm bị khuất. [Bảng nhãn gốc](train_v2/events.csv) · [Ảnh đối chiếu](train_v2/review_events.jpg)

| Bước | Kỹ thuật trong V2 | Mục đích |
|---|---|---|
| Chọn phạm vi lấy hình | Clip nền lấy 16 frame trải trên toàn clip. Sự kiện có chú thích tạo các đoạn ngắn quanh mốc, giới hạn trong khoảng đã gán nhãn. | Giữ dữ liệu nền đủ lớn và lấy thêm mẫu tập trung vào diễn biến hiếm. |
| Lấy mẫu theo thời gian | Chia mỗi đoạn thành 16 khoảng. Khi train lấy ngẫu nhiên một vị trí trong từng khoảng (*temporal jitter*). Với sự kiện, còn dịch mốc tương đối trong khoảng 35–65% chiều dài cửa sổ nếu biên cho phép. Khi đánh giá, lấy vị trí giữa mỗi khoảng. | Tăng đa dạng về thời điểm lấy frame khi train và giữ kết quả đánh giá có thể lặp lại. |
| Chuẩn hóa ảnh và tăng cường dữ liệu | Đổi BGR sang RGB, resize giữ tỷ lệ, thêm viền thành 224 × 224 (*letterbox*), chuẩn hóa theo ImageNet. Khi lật ngang, đổi tọa độ và các cặp khớp trái/phải tương ứng. | Giữ hình học cơ thể và sự nhất quán giữa ảnh, tọa độ khớp. |
| Tăng tần suất mẫu hiếm | Mỗi epoch duyệt clip nền một lần, lặp mỗi cửa sổ sự kiện train ba lần rồi xáo trộn. | Tăng cơ hội học cú đánh nhanh và cảnh bình thường dễ nhầm. Số nguồn độc lập vẫn là số video đã thu thập. |
| Chọn nhãn tiếp xúc cho cửa sổ | Chỉ tính loss tiếp xúc nếu có frame lấy mẫu nằm cách `event_s` tối đa khoảng 1,1 frame gốc và có nhãn tiếp xúc. Nhãn hành động vẫn được dùng. | Tránh giám sát điểm chạm ở cửa sổ mà model chưa thấy đúng thời điểm. |

> **Cách nói với giảng viên.** Em chia dữ liệu theo nguồn trước khi cắt, sau đó dùng hai mức gán nhãn. Clip nền cung cấp nhãn hành động cả clip. Các đoạn khó được đánh thêm mốc thời gian và nhãn tiếp xúc để model học tập trung vào tương tác.

## 2. Vung tay che người phía sau

**Vì sao khó?** Một camera RGB quan sát hình chiếu 2D. Tay người phía trước có thể đi ngang mặt hoặc thân người phía sau trên ảnh dù họ cách nhau theo chiều sâu. Tín hiệu hình ảnh chồng lên nhau có thể xuất hiện trong cảnh vung tay bình thường, chạm thân thiện và đánh nhau. V2 xử lý sự mơ hồ này ở cả dữ liệu lẫn mô hình.

| Kỹ thuật | Cách làm cụ thể |
|---|---|
| **Mẫu âm khó (*hard negatives*)** | Dùng clip `action=0` có hình ảnh dễ gây nhầm như vung tay che người, hai thân người chồng lên nhau hoặc cử chỉ tay. `NV_46.mp4` là mẫu train `arm_overlap / no_touch`, có mốc 2,60 giây. `NV_575.mp4` là mẫu test `arm_overlap / unclear`, có mốc 2,36 giây. |
| **Học đa nhiệm (*multi-task learning*)** | Tách đầu phân loại hành động hai lớp và đầu tiếp xúc ba lớp. Ví dụ, `friendly_touch` dạy mô hình rằng tiếp xúc cũng xuất hiện trong hành động bình thường. Mẫu `unclear` giữ được thông tin về che khuất. |
| **Ghép cặp người qua thời gian** | Keypoint R-CNN lấy khớp từng frame. Notebook dùng tâm hai vai làm mốc để so độ dịch chuyển của hai cách ghép cặp giữa các frame. Frame thiếu cặp đủ rõ, thiếu vai/cổ tay theo tiêu chí hoặc ghép danh tính mơ hồ được điền 0 cho chuỗi pose. |
| **Hiệu chỉnh có điều kiện chất lượng pose** | GRU hai chiều mã hóa chuỗi khớp của hai người. V2 chỉ cộng hiệu chỉnh từ pose vào logits hành động khi cửa sổ đạt tiêu chí chất lượng, gồm có pose ở ít nhất một nửa số frame. Khi chất lượng thấp, nhánh RGB tiếp tục cho điểm hành động. |

**Minh chứng test lại trên `NV_575.mp4`.** Nhãn thật của cả hai cửa sổ là bình thường, tiếp xúc `unclear`. Mốc 2,36 giây nằm trong cả hai cửa sổ.

| Thời gian trong video gốc | Điểm hành động V2 | Kết quả ở ngưỡng 50% |
|---|---:|---|
| 1,86–2,86 s | 3,96% | Bình thường |
| 1,72–3,75 s | 14,89% | Bình thường |

V2 không báo nhầm ở hai cửa sổ của ca này. Cả hai lấy từ **một nguồn**, nên kết quả minh họa khả năng xử lý ca cụ thể. Muốn đo đóng góp riêng của pose, cần so cùng những cửa sổ này với và không có pose, rồi lặp trên nhiều nguồn. [Kết quả test lại](train_v2/runs/local_gpu_01/arm_overlap_retest.csv) · [Tổng hợp test theo tình huống](train_v2/runs/local_gpu_01/test_summary.csv)

> **Cách nói với giảng viên.** Em tạo các mẫu âm khó vì tay chồng lên thân người trong ảnh có thể chỉ là che khuất. Em giữ hai nhãn riêng cho hành động và tiếp xúc, dùng chuỗi ảnh để nhìn diễn biến và chỉ bổ sung hiệu chỉnh từ khung xương khi theo dõi đủ rõ.

## 3. Cú đánh nhanh

**Vì sao dễ bỏ sót?** Nếu trải 16 frame trên một đoạn dài, các frame lấy được có thể rơi chủ yếu vào trước và sau cú ra đòn. Cú đánh nhanh cần đủ mật độ lấy mẫu quanh sự kiện, đồng thời cần ngữ cảnh để phân biệt với cử chỉ bình thường.

### Lấy mẫu nhiều độ dài quanh mốc

**Kỹ thuật cửa sổ thời gian đa tỉ lệ (*multi-scale temporal windowing*)** thử độ dài **0,5 / 1 / 3 giây**. Mỗi độ dài được giới hạn theo khoảng chú thích và bỏ nếu ngắn hơn `min_window_s`. Nếu khoảng chú thích chỉ dài 1,73 giây, nhánh 3 giây tạo cửa sổ 1,73 giây. Vị trí bắt đầu cũng được giới hạn để cửa sổ nằm trong đoạn đã gán nhãn.

| Cửa sổ mục tiêu | Khoảng giữa 16 vị trí lấy mẫu | Vai trò |
|---|---:|---|
| 0,5 giây | Khoảng 31 ms | Tập trung mật độ lấy mẫu quanh động tác ngắn. |
| 1 giây | Khoảng 63 ms | Bao quát lúc bắt đầu và diễn biến ngay sau tương tác. |
| 3 giây | Khoảng 188 ms | Giữ thêm ngữ cảnh trước và sau, tùy độ dài chú thích. |

Đây là khoảng cách giữa các **vị trí lấy mẫu lý thuyết**. Vị trí được quy về chỉ số frame gốc, nên với video FPS thấp, hai vị trí có thể lấy cùng một frame. Cửa sổ 0,5 giây có mật độ vị trí gấp sáu lần cửa sổ 3 giây ở cùng ngân sách 16 frame.

**Minh chứng ở test `V_101.mp4`.** Đoạn chú thích là 1,10–2,83 giây, mốc sự kiện 1,91 giây, nhãn hành động đánh nhau và nhãn tiếp xúc `unclear`. Ba cửa sổ **0,5 / 1 / 1,73 giây** lần lượt đạt điểm hành động **98,06% / 99,76% / 99,78%**, đều được gắn `suspected`. Pose của cả ba cửa sổ dưới tiêu chí chất lượng, nên điểm hành động trong ca này đến từ nhánh RGB. Đây là ba cửa sổ của **một nguồn**. [Nhãn sự kiện](train_v2/events.csv) · [Dự đoán test](train_v2/runs/local_gpu_01/test_predictions.csv)

### Tìm sự kiện trên video mới và lưu bằng chứng

- **Khi huấn luyện và test theo nhãn sự kiện**, người gán nhãn cung cấp `event_s`. Đây là mốc để chuẩn bị đầu vào.
- **Khi quét video KO**, notebook dùng *cửa sổ trượt đa tỉ lệ* và chấm điểm mỗi bước 0,2 giây. Kết quả chỉ ra **đoạn thời gian có dấu hiệu đánh nhau**, với độ phân giải phụ thuộc độ dài cửa sổ và bước trượt.
- **Khi chạy webcam**, server xét tối đa 16 frame gần nhất mỗi 0,5 giây theo nhịp lấy mẫu mục tiêu. Khi điểm hành động đạt 0,5, server kiểm tra số người ở frame giữa và cuối. Một lần đủ điều kiện có thể mở cảnh báo, do đó cú đánh ngắn có cơ hội được ghi dù điểm hạ ngay sau đó. Bộ đệm giữ hình trước cảnh báo và gom các lần cảnh báo gần nhau vào cùng clip. [Notebook quét video](demo.ipynb) · [Backend webcam](demo-web/demo_server.py) · [Bộ đệm và ghi clip](demo-web/monitor.py)

> **Cách nói với giảng viên.** Khi train, em cắt nhiều độ dài quanh mốc đã gán nhãn. Đoạn ngắn giữ chi tiết thời gian, đoạn dài giữ ngữ cảnh. Khi demo video, em trượt cửa sổ dọc theo clip. Với webcam, điểm vượt ngưỡng và điều kiện hai người có thể kích hoạt ngay một clip có hình trước sự kiện.

## Kiến trúc và huấn luyện

| Câu hỏi | Trả lời |
|---|---|
| VideoMAE học gì? | VideoMAE-base là Transformer xử lý các vùng ảnh theo cả không gian và thời gian. V2 fine-tune từ trọng số Kinetics. Đầu vào chứa 16 frame RGB, resize giữ tỉ lệ, thêm viền thành 224 × 224 và chuẩn hóa ImageNet. |
| GRU dùng để làm gì? | Keypoint R-CNN trích 9 khớp mỗi người trong cặp. Mỗi khớp có hai tọa độ chuẩn hóa và cờ quan sát. Chuỗi pose có dạng 16 × 54. GRU hai chiều mã hóa diễn biến khớp trước và sau trong cửa sổ. |
| Hai nhánh kết hợp thế nào? | Đặc trưng pose hiệu chỉnh logits hành động khi đạt điều kiện chất lượng. Đầu tiếp xúc nhận cả embedding video và pose rồi dự đoán ba lớp. Khi ghép cặp thất bại, pose frame tương ứng được biểu diễn bằng 0. Các clip nền cũng dùng pose bằng 0. |
| Nhãn thiếu xử lý thế nào? | Tổng loss gồm cross-entropy hành động và cross-entropy tiếp xúc, mỗi phần chỉ tính trên nhãn hợp lệ. Nếu frame lấy mẫu không đủ gần mốc tương tác, bỏ loss tiếp xúc cho mẫu đó. |
| Học tình huống hiếm ra sao? | Mỗi epoch duyệt clip nền một lần và lặp mỗi cửa sổ sự kiện train ba lần. Lật ngang cập nhật đồng thời ảnh và khớp trái/phải. Model xét hai thứ tự của cặp người để giảm phụ thuộc thứ tự đầu vào. |
| Train trên RTX 3050 4 GB thế nào? | 5 epoch, batch 1, tích lũy gradient 16 bước. Đóng băng backbone trừ block cuối, dùng AdamW, FP16, GradScaler, warmup rồi cosine decay. LR là 2e-5 cho block video và 2e-4 cho các phần còn lại. |
| Vì sao chọn epoch 2? | Validation đạt F1 hành động 0,9900 và macro-F1 tiếp xúc 0,7079. Điểm chọn là trung bình hai F1 khi đủ lớp tiếp xúc. Epoch 2 đạt điểm cao nhất sớm nhất nên giữ checkpoint đó. |

[Notebook train V2](train_v2/train_v2_vscode.ipynb) · [Lịch sử huấn luyện](train_v2/runs/local_gpu_01/history.json)

## Kết quả và phạm vi kết luận

| Tập V2 | Số video | Accuracy | Báo nhầm | Bỏ sót |
|---|---:|---:|---:|---:|
| RLVS test, clip nền | 287 | 98,61% | 2 | 2 |
| Hockey Fights | 1.000 | 51,80% | 480 | 2 |
| RWF-2000 validation | 400 | 87,00% | 31 | 21 |

**“Vì sao Hockey báo nhầm nhiều?”**  
Tại cùng ngưỡng hành động 0,5, 480/500 clip Hockey bình thường vượt ngưỡng. ROC AUC vẫn đạt 0,9632, nên thứ tự điểm còn có ích dù ngưỡng từ RLVS chuyển sang Hockey chưa phù hợp. Nếu triển khai ở sân thể thao, dùng validation riêng để thử ngưỡng và xem lại các cảnh model báo nhầm. [Notebook đánh giá](demo.ipynb)

**“V2 tốt hơn V1 ở đâu?”**  
Ở các lần chạy đã lưu, V1 cảnh báo 3/4 clip KO và V2 cảnh báo 4/4. Clip `KO.mp4` là ca khác biệt. V1 test theo cửa sổ server, V2 theo quét đa tỉ lệ trong notebook. Để so sánh hai pipeline trên dữ liệu thực tế, chạy cả hai trên cùng tập clip bình thường và clip có đánh nhau rồi đối chiếu FP, FN, Precision, Recall và F1. [Hướng dẫn so sánh](HUONG_DAN_CHAY_V1_V2.md)

**“RWF-2000 đã dùng để train chưa?”**  
Checkpoint V2 đang báo cáo học trên RLVS. RWF-2000 validation được dùng để đánh giá ngoài tập. Sáu đoạn RWF train/NonFight đã được đánh dấu để duyệt, trong đó năm đoạn là ứng viên fine-tune sau. [Danh sách đã gán nhãn](HUONG_DAN_CHAY_V1_V2.md#những-video-đã-được-đánh-dấu)

## Khi thầy hỏi về demo webcam

**“Có hai người là đánh nhau à?”**  
Webcam chấm điểm hành động trước. Nếu điểm đạt ngưỡng 0,5, bộ phát hiện người kiểm tra frame giữa và cuối cửa sổ. Cần có từ hai người trong cùng một frame mới ghi cảnh báo nghi ngờ. Điều kiện này giúp lọc cảnh một người, nhưng hai người đứng sát nhau vẫn có thể gây báo nhầm.

**“Web có xác nhận tay chạm thật chưa?”**  
Bản notebook có đầu dự đoán tiếp xúc. Run hiện tại đặt `contact_ready=false` vì tập kiểm chứng có pose đủ chất lượng chưa đáp ứng điều kiện hiệu chỉnh. Webcam đang triển khai dùng nhánh hành động và điều kiện số người. Người quan sát xem lại clip để quyết định lưu hay xóa cảnh báo. [Chính sách run](train_v2/runs/local_gpu_01/policy.json) · [Backend V2](demo-web/demo_server.py)

**“Khi nghi ngờ thì web làm gì?”**  
Web hiện cảnh báo trực quan, server dùng bộ đệm lấy khoảng 3 giây hình trước cửa sổ phát hiện nếu đủ dữ liệu. Với webcam, các lần cảnh báo gần nhau nối vào cùng clip. Server đóng đoạn sau khoảng 3 giây liên tục hết cảnh báo hoặc khi dừng ghi. MP4 lưu không tiếng, còn SQLite giữ thời gian, nguồn, điểm và trạng thái xem lại. [Bộ đệm và ghi clip](demo-web/monitor.py)

