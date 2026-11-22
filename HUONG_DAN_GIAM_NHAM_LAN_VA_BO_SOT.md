# Hướng dẫn giảm nhận nhầm và bỏ sót hành vi đánh nhau trong video

## 1. Phạm vi và hiện trạng project

Tài liệu này áp dụng cho `trainV1.ipynb`, `demo1.ipynb` và `chuan_bi_du_lieu.ipynb` trong project.

Bản train hiện tại fine-tune toàn bộ VideoMAE để phân loại clip RGB thành hai lớp: đánh nhau và không đánh nhau. Mỗi clip được chia thành 16 đoạn, lấy một frame trong mỗi đoạn. Khi train chọn ngẫu nhiên trong đoạn; khi đánh giá chọn ở giữa đoạn. Ảnh được resize 256×256, crop 224×224 và chuẩn hóa.

Bản demo dùng cửa sổ 3 giây, bước trượt 1 giây, sau đó lấy trung bình điểm của các cửa sổ chồng lên mỗi frame. Ngưỡng 0,61 được chọn trên validation của bản RLVS hiện tại. Đây là điểm phân loại cửa sổ được gán xuống frame, chưa phải dự đoán được huấn luyện bằng nhãn từng frame.

Hai tình huống cần xử lý đồng thời:

1. **Báo nhầm:** người phía trước vung tay, che người phía sau nhưng không đánh.
2. **Bỏ sót:** một người đấm rất nhanh một lần, người còn lại ngã ngay; không có giằng co hoặc đánh liên tục.

Mục tiêu là giảm báo nhầm mà vẫn giữ khả năng phát hiện hành động rất ngắn. Không đặt điều kiện bắt buộc phải có giằng co, nhiều cú đánh hoặc nhiều cửa sổ liên tiếp vượt ngưỡng.

## 2. Vì sao bản hiện tại có thể sai?

### 2.1. Che khuất và nhập nhằng chuyển động

Mô hình phân loại toàn cảnh, chưa theo dõi riêng từng người hoặc xác nhận tiếp xúc giữa tay và cơ thể. Vung tay, vẫy chào, đùa giỡn hoặc đi ngang camera có thể tạo hình ảnh giống hành vi đánh.

Nếu cú đánh bị che hoàn toàn, một camera RGB không thể đảm bảo xác định chính xác. Có thể dùng phần nhìn thấy trước và sau để hỗ trợ, nhưng không được coi suy đoán là bằng chứng chắc chắn có tiếp xúc.

### 2.2. Lấy mẫu thưa có thể bỏ qua cú đánh nhanh

Với clip dài L giây, 16 đoạn có độ dài khoảng L/16. Clip 10 giây tương ứng khoảng 0,625 giây mỗi đoạn. Một cú đấm chỉ xuất hiện trong vài frame có thể nằm giữa những frame được chọn.

Cửa sổ demo 3 giây với 16 frame lấy mẫu tương ứng khoảng 0,1875 giây giữa các mẫu. Bước trượt 1 giây cũng làm giới hạn vị trí đặt cửa sổ. Đây là khoảng cách lấy mẫu xấp xỉ, không phải cam kết sẽ bắt được sự kiện ngắn hơn.

Ngoài ra, camera FPS thấp, nhòe chuyển động hoặc nén mạnh có thể làm mất thông tin ngay từ video gốc. Lấy mẫu dày hơn không khôi phục được chi tiết chưa được ghi lại.

### 2.3. Trung bình điểm có thể làm yếu đỉnh sự kiện ngắn

Trong `demo1.ipynb`, hàm `analyze()` lấy trung bình điểm các cửa sổ phủ lên frame. Một cửa sổ nhận ra cú đánh có thể bị các cửa sổ chứa nhiều nền bình thường làm giảm điểm. Ngược lại, một cửa sổ điểm cao cũng có thể tô cảnh báo cho các frame không chứa cú đánh.

### 2.4. Chênh lệch miền dữ liệu

Output đã lưu của bản train RLVS ghi nhận accuracy 99,34% trên RLVS test nhưng 80,90% trên Hockey; Hockey có 166 báo nhầm và 25 bỏ sót. Các số này chưa cho biết riêng hiệu quả trên che khuất hoặc cú đánh một lần. Chưa thể quy toàn bộ lỗi cho một nguyên nhân nếu chưa xem và gắn nhãn từng trường hợp.

## 3. Ưu tiên 1: xây dựng dữ liệu đúng tình huống

### 3.1. Bổ sung mẫu không bạo lực dễ gây nhầm

Thu thập các video không bạo lực có chuyển động giống cú đánh:

- Vung tay, chỉ tay, vẫy chào gần người khác.
- Đi ngang che người, chen qua đám đông.
- Ôm, bắt tay, đập tay, đùa giỡn.
- Thể thao, ăn mừng, chuyển động nhanh.
- Vấp ngã, trượt chân, ngồi hoặc nằm xuống nhanh khi không bị tấn công.

Những video này là **hard negatives**: mẫu âm khó giúp mô hình học rằng chuyển động mạnh hoặc có người ngã chưa đủ để kết luận đánh nhau.

### 3.2. Bổ sung mẫu bạo lực ngắn

Thu thập cả các video:

- Một cú đấm hoặc tát duy nhất, sau đó kết thúc.
- Cú đánh rất nhanh, không có giằng co.
- Nạn nhân ngã ngay hoặc mất thăng bằng sau cú đánh.
- Có cú đánh nhưng nạn nhân không ngã.
- Bạo lực bị che một phần, góc quay xa, ánh sáng kém hoặc nhòe.

Không chỉ đưa mẫu đánh nhau kéo dài vào tập train. Nếu dữ liệu chủ yếu là giằng co nhiều giây, mô hình có thể học thiên lệch theo dạng đó.

Từ video thông thường chỉ nên gắn nhãn quan sát được như “ngã” hoặc “nằm bất động”; không xác nhận “bất tỉnh” nếu không có bằng chứng bổ sung đáng tin cậy.

### 3.3. Gắn nhãn theo thời gian

Nên tạo bảng sự kiện riêng, ví dụ:

```csv
video_id,source_id,start_s,end_s,event_type,occlusion,split
clip_001,source_001,2.10,2.45,single_strike,partial,train
clip_002,source_002,1.00,1.60,nonviolent_arm_swing,partial,val
```

Quy ước rõ `start_s` và `end_s` là đoạn hành động mục tiêu. Có thể gắn thêm mốc ngã hoặc phản ứng trong bảng phụ để phân tích.

Đoạn chỉ chứa nạn nhân nằm sau sự kiện không tự động là mẫu “đang đánh”. Hãy phân biệt nhãn sự kiện tấn công với nhãn hậu quả; nếu muốn nhận dạng cả hai, phải định nghĩa nhiệm vụ và nhãn tương ứng.

### 3.4. Chia tập trước khi cắt cửa sổ

Giữ toàn bộ video cùng nguồn, các bản sao và các cửa sổ cắt từ chúng trong cùng một tập. Không chia ngẫu nhiên các cửa sổ chồng lấn vào train và test.

`chuan_bi_du_lieu.ipynb` đã có dHash, Hamming và chia theo nhóm để hạn chế trùng nội dung. Cần kết hợp thêm `source_id` nếu biết nhiều clip lấy từ cùng video gốc. dHash không bảo đảm phát hiện mọi bản cắt lệch thời gian hoặc chỉnh sửa.

Notebook chuẩn bị dữ liệu hiện xây dựng hướng train RWF-2000; `trainV1.ipynb` đang đọc split RLVS. Cần chọn và ghi rõ một quy trình, rồi chỉnh train đọc đúng manifest. Không gán kết quả của bản RLVS cho quy trình RWF chưa chạy.

## 4. Ưu tiên 2: lấy mẫu đủ dày cho hành động nhanh

### 4.1. Train bằng các cửa sổ ngắn quanh sự kiện

Giữ cấu hình 16 frame ban đầu để dễ so sánh, nhưng lấy chúng từ cửa sổ ngắn hơn thay vì luôn trải đều toàn bộ video dài.

Các cấu hình thử nghiệm ban đầu, chưa phải giá trị tối ưu:

| Nhánh | Độ dài cửa sổ | Số frame | Mục đích |
|---|---:|---:|---|
| Nhanh | 0,5–1 giây | 16 | Giữ chi tiết của cú đánh ngắn |
| Ngữ cảnh | 2–3 giây | 16 | Quan sát trước/sau và tương tác |

Nếu cửa sổ chỉ có ít frame gốc, lấy lặp frame để đủ 16 không tạo thêm thông tin chuyển động. Phải ghi nhận FPS nguồn khi đánh giá.

Với mẫu dương, dịch ngẫu nhiên vị trí cửa sổ quanh mốc sự kiện để cú đánh không luôn ở chính giữa. Đảm bảo cửa sổ được gắn nhãn dương thực sự chứa hành động theo quy tắc nhãn đã định nghĩa. Cửa sổ chỉ chứa chuẩn bị hoặc hậu quả cần nhãn riêng hoặc loại khỏi bài toán phân loại cú đánh.

### 4.2. Huấn luyện phù hợp cách suy luận

Nếu demo chuyển sang cửa sổ 0,5–1 giây, cần fine-tune và đánh giá với cửa sổ tương tự. Chỉ thay độ dài cửa sổ ở demo có thể tạo chênh lệch so với dữ liệu train.

Có thể thử một mô hình học trên nhiều độ dài cửa sổ, rồi dùng cùng mô hình cho các thang thời gian. Đây là phương án cần thực nghiệm; không mặc định hai nhánh sẽ luôn cải thiện.

Chưa nên chỉ tăng từ 16 lên 32 hoặc 64 frame trong code. Cần kiểm tra cấu hình mô hình, vị trí nhúng và khả năng nạp checkpoint; số frame lớn hơn cũng tăng nhu cầu bộ nhớ và tính toán.

### 4.3. Bước trượt nhỏ hơn

Thử nhánh nhanh với bước trượt 0,1–0,25 giây và đo tốc độ trên phần cứng thực tế. Nhánh ngữ cảnh có thể chạy thưa hơn.

Cửa sổ ngắn và bước trượt nhỏ giúp tăng cơ hội quan sát cú đánh nhưng tăng số lần suy luận và số cơ hội báo nhầm. Cần đánh giá cả recall sự kiện và số cảnh báo trên mỗi giờ video.

## 5. Ưu tiên 3: thiết kế cảnh báo không bỏ qua cú đánh một lần

### 5.1. Không bắt buộc cảnh báo kéo dài

Không dùng “phải vượt ngưỡng ở nhiều cửa sổ liên tiếp” làm điều kiện duy nhất để phát hiện. Điều kiện đó có thể loại mất một cú đánh nhanh.

Thay vào đó, thử hai đường phát hiện:

- **Sự kiện ngắn:** một cửa sổ nhanh có bằng chứng mạnh tạo một sự kiện nghi vấn ngay.
- **Sự kiện kéo dài:** nhiều cửa sổ có bằng chứng vừa phải hỗ trợ nhận dạng hành động liên tục.

Các ngưỡng phải được chọn trên validation mới có chứa cả hard negatives và bạo lực ngắn. Không dùng lại 0,61 mà mặc định hiệu quả giữ nguyên.

### 5.2. Giữ điểm đỉnh, không chỉ giữ trung bình

Trong `analyze()`, nên lưu riêng thời điểm bắt đầu, kết thúc, độ dài cửa sổ và điểm dự đoán. Dùng các bản ghi này để quyết định sự kiện; đường trung bình theo frame chỉ phục vụ trực quan hóa.

Có thể so sánh trung bình, điểm lớn nhất và trung bình top-k trên validation. Điểm lớn nhất giữ được đỉnh ngắn nhưng dễ báo nhầm; không coi lấy max là giải pháp mặc định.

### 5.3. Dùng hậu quả làm bằng chứng hỗ trợ

Một cú đánh ngắn có thể được hỗ trợ bởi phản ứng ngay sau đó: né, bật người, mất thăng bằng hoặc ngã. Tuy nhiên:

- Người ngã không đồng nghĩa đã bị đánh.
- Không ngã không đồng nghĩa không có bạo lực.
- Không cần có giằng co để xác nhận một sự kiện nghi vấn.

Nếu dùng frame sau cú đánh, phải tính thời gian chờ đó vào độ trễ cảnh báo. Demo hiện tại xử lý video đã có sẵn, chưa chứng minh khả năng cảnh báo trực tiếp theo thời gian thực.

### 5.4. Gom cảnh báo sau khi phát hiện

Gom các cửa sổ trùng một sự kiện để tránh phát nhiều cảnh báo cho cùng một cú đánh. Việc gom thực hiện sau khi tạo ứng viên; không xóa ứng viên ngắn chỉ vì nó không kéo dài.

Lưu đoạn video có vài giây trước và sau ứng viên để người xem kiểm tra. Có thể hiển thị trạng thái “nghi vấn cần xem lại” khi che khuất làm bằng chứng không đủ rõ.

## 6. Ưu tiên 4: bổ sung thông tin từng người khi cần

Sau khi có baseline dữ liệu và lấy mẫu tốt, có thể thử:

1. Phát hiện và theo dõi từng người.
2. Cắt vùng chứa cặp người và giữ thêm biên quanh họ để không mất cánh tay.
3. Kết hợp vùng cặp người với ảnh toàn cảnh để giữ ngữ cảnh.
4. Dùng pose hoặc chuyển động làm đặc trưng phụ nếu có lợi qua thử nghiệm.

Pose có thể mất hoặc sai khớp khi che khuất, nhòe hoặc người quá nhỏ. Tracking cũng có thể đổi ID. Không dùng riêng khoảng cách cổ tay đến đầu, hộp người chồng nhau hoặc chuyển động lớn để kết luận có cú đánh.

Nếu nhiều góc camera đã có sẵn, góc nhìn bổ sung có thể giảm phần bị che. Một góc camera không nhìn thấy tiếp xúc thì không thể bảo đảm suy ra tiếp xúc từ hình ảnh đó.

## 7. Cách đánh giá để biết thay đổi có hiệu quả

### 7.1. Tạo tập kiểm tra theo tình huống

| Nhóm | Điều cần đo |
|---|---|
| Vung tay không đánh | Tỉ lệ báo nhầm |
| Che khuất không bạo lực | Tỉ lệ báo nhầm |
| Một cú đánh nhanh | Recall sự kiện và độ trễ |
| Cú đánh rồi ngã | Recall, khả năng tách khỏi ngã không do tấn công |
| Bạo lực bị che một phần | Recall và lỗi do mất thông tin |
| Video bình thường dài | Số cảnh báo nhầm mỗi giờ |

Các nhóm có thể giao nhau. Ghi số lượng mẫu và nguồn cho từng nhóm; chưa đặt mục tiêu phần trăm khi chưa biết mức khó và yêu cầu triển khai.

### 7.2. Đánh giá theo sự kiện

Ngoài accuracy, precision, recall, F1 theo clip, cần đo:

- Recall riêng cho sự kiện một cú đánh.
- Số cảnh báo nhầm/giờ trên video có thời lượng liên tục; không suy ra chỉ số này từ bộ clip cân bằng ngắn.
- Độ trễ từ lúc sự kiện bắt đầu đến lúc hệ thống phát cảnh báo.
- Số cảnh báo trùng cho cùng một sự kiện.
- Tốc độ xử lý và bộ nhớ trên phần cứng mục tiêu.

Định nghĩa trước quy tắc ghép cảnh báo với sự kiện thật, khoảng dung sai thời gian và cách đếm cảnh báo trùng. Với sự kiện rất ngắn, sai lệch nhãn nhỏ cũng ảnh hưởng lớn đến kết quả.

Chọn ngưỡng và quy tắc gom trên validation. Khóa cấu hình trước khi chạy test. Nếu tiếp tục sửa sau khi xem test, cần bộ đánh giá độc lập mới để có ước lượng cuối cùng đáng tin cậy.

### 7.3. Thử từng thay đổi

| Thử nghiệm | Thay đổi so với bản trước |
|---|---|
| A | Baseline hiện tại |
| B | Bổ sung hard negatives và mẫu cú đánh ngắn |
| C | Train/suy luận với cửa sổ ngắn và lấy mẫu dày |
| D | Kết hợp nhánh nhanh và nhánh ngữ cảnh |
| E | Thêm phát hiện/theo dõi người hoặc pose nếu còn cần |

Giữ cùng nguồn chia tập, báo cáo cả mức giảm báo nhầm và thay đổi recall của cú đánh nhanh. Tăng precision bằng cách tăng ngưỡng nhưng làm mất nhiều cú đánh nhanh không được coi là đã xử lý xong vấn đề.

## 8. Những chỗ cần chỉnh trong project

### `trainV1.ipynb`

- `read_frames()`: nhận thêm mốc bắt đầu/kết thúc cửa sổ; lấy 16 frame trong cửa sổ đó.
- `VideoDS`: đọc manifest có nhãn thời gian, bổ sung mẫu âm khó và mẫu dương ngắn.
- Phần chia tập: dùng nhóm nguồn trước khi tạo cửa sổ; kết nối rõ với manifest đã chọn.
- Vòng train: xử lý phần gradient tích lũy còn dư ở cuối epoch. Bản hiện tại chỉ cập nhật khi đủ 4 batch, nên gradient của 3 batch cuối bị bỏ ở cấu hình đã lưu. Khi xử lý nhóm cuối cần chuẩn hóa loss theo số batch thực tế trong nhóm và cập nhật lại số bước scheduler.
- Thiết lập seed Python, NumPy, PyTorch và worker của DataLoader để hỗ trợ tái lập.
- Lưu cấu hình cửa sổ, quy trình tiền xử lý, nhãn lớp và ngưỡng cùng checkpoint.

### `demo1.ipynb`

- `make_input()`: tái sử dụng đúng quy trình đầu vào đã dùng khi train.
- `analyze()`: thêm nhánh cửa sổ nhanh, lưu điểm từng cửa sổ và tạo ứng viên sự kiện trước khi làm mượt để hiển thị.
- Thay ngưỡng ghi cứng bằng cấu hình được chọn trên validation của phiên bản tương ứng.
- Giữ khả năng phát hiện từ một cửa sổ mạnh; gom cảnh báo trùng sau đó.
- Ghi rõ giới hạn hiện tại chỉ xử lý 60 giây đầu.
- Tách số liệu ghi lại và số liệu vừa tính; tránh trình bày số liệu cũ như kết quả của checkpoint mới.

### `chuan_bi_du_lieu.ipynb`

- Bổ sung `source_id`, nhãn thời gian và loại tình huống.
- Giữ các clip cùng nguồn trong một tập, kể cả khi dHash không phát hiện được chúng.
- Lưu danh sách clip loại bỏ và quy tắc lọc. Nếu loại clip khỏi test gốc, báo cáo rõ đây là tập test đã lọc; số liệu không trực tiếp tương đương với đánh giá trên toàn bộ test gốc.
- Kiểm tra lại mẫu bị gán nhóm gần trùng; không coi assert qua là bảo đảm hết mọi dạng rò rỉ dữ liệu.

## 9. Thứ tự thực hiện đề xuất

1. Lưu baseline và dự đoán hiện tại để so sánh.
2. Tạo tập tình huống có che khuất, vung tay không đánh, cú đánh một lần và ngã không do tấn công.
3. Gắn nhãn thời gian, chia theo nguồn và kiểm tra trùng.
4. Fine-tune với dữ liệu bổ sung, giữ cấu hình cũ để đo tác động của dữ liệu.
5. Thử cửa sổ ngắn lấy mẫu dày, rồi thử kết hợp ngữ cảnh dài hơn.
6. Chọn ngưỡng và cơ chế cảnh báo trên validation, đồng thời theo dõi recall cú đánh nhanh.
7. Đánh giá test độc lập, đo độ trễ và cảnh báo nhầm/giờ trước khi kết luận.

Nguyên tắc xuyên suốt: **không yêu cầu bạo lực phải kéo dài mới phát hiện; cũng không coi vung tay hoặc ngã là đủ để xác nhận có tấn công.**

---

Tài liệu được lập từ việc đọc code và output đã lưu trong ba notebook. Các cấu hình mới là đề xuất thử nghiệm, chưa được triển khai hoặc đo hiệu quả trong project.
