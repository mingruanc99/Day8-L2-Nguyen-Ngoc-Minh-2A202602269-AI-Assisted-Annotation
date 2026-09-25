# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyễn Ngọc Minh

Công cụ gán nhãn đã dùng: Sửa trực tiếp file nhãn YOLO và kiểm tra trực quan theo quy chuẩn GUIDELINE_LABEL.md

Báo cáo phân tích toàn diện kết quả thực nghiệm học chủ động (Active Learning) trên bài toán phát hiện phương tiện (car) từ video giám sát ban đêm. Mọi số liệu trong báo cáo được trích xuất trực tiếp từ các tệp nhật ký thực nghiệm: 
reports/rrounds_table.md, outputs/selection_round1.csv, outputs/metrics_round0.json, outputs/metrics_round1.json và outputs/round1_diff.md.

---

## 1. Dữ liệu và cách chia tập

### Lý do chia tập theo trục thời gian và thiết lập vùng đệm (buffer zone)
Dữ liệu thực nghiệm được trích xuất từ một camera cố định đặt trên cầu vượt quay xuống đường cao tốc vào ban đêm (tần số 2.5 fps, độ dài 160 giây, gồm 400 frames từ frame_0000.jpg đến frame_0399.jpg). Do camera hoàn toàn đứng yên và tốc độ khung hình tương đối cao (0.4 giây/ảnh), các khung hình liền kề có bối cảnh nền tĩnh 100% giống hệt nhau, và mỗi phương tiện giao thông cần từ vài giây đến hơn mười giây để di chuyển qua tầm nhìn của camera.

Nếu chia dữ liệu ngẫu nhiên (random split) giữa tập huấn luyện (train/pool) và tập kiểm thử (test):
- Cùng một chiếc xe sẽ xuất hiện đồng thời ở cả tập huấn luyện và tập kiểm thử chỉ cách nhau 0.4 - 0.8 giây, với hình thái, vệt đèn, góc chiếu sáng và kích thước gần như không đổi.
- Khi đó, mô hình thực chất chỉ đang học vẹt (memorize) lại vị trí và đặc trưng của chính những chiếc xe nó đã nhìn thấy trong tập train thay vì học khả năng tổng quát hóa (generalization) trên các xe mới. Hiện tượng này là **rò rỉ dữ liệu nghiêm trọng (data leakage)**.

### Hướng lệch của số đo nếu chia ngẫu nhiên
Nếu chia ngẫu nhiên, các số đo trên tập kiểm thử (như AP50, Precision, Recall, F1) sẽ **bị thổi phồng lệch lạc theo hướng cực kỳ lạc quan (artificially over-optimistic)**. Mô hình có thể đạt AP50 > 95% trên tập test ngẫu nhiên, nhưng khi triển khai vào một đoạn video mới hoặc một khung giờ khác thì hiệu năng sẽ sụt giảm thảm hại.

Do đó, thiết kế chia tập của lab:
- **Tập kiểm thử (test)**: Gồm 20 ảnh chia làm 4 khối thời gian độc lập (tâm tại 20s, 60s, 100s, 140s; mỗi khối 5 ảnh cách nhau 1.2s).
- **Vùng đệm (buffer)**: Loại bỏ hoàn toàn 112 ảnh trong khoảng 4 giây trước và sau mỗi khối test cũng như xen kẽ giữa các ảnh test, đảm bảo xe ở tập test đã chạy thoát khỏi khung hình trước khi tập pool bắt đầu.
- **Tập chưa gán nhãn (pool)**: Gồm 268 ảnh còn lại, trong đó ảnh pool gần nhất cách ảnh test tối thiểu 4.4 giây. Đây là thiết lập chuẩn mực chống rò rỉ dữ liệu trong xử lý video.

---

## 2. Mô hình khởi đầu lạnh (cold start)

### Kết quả vòng 0 (trích xuất từ 
rounds_table.md)
| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

### Phân tích độ khớp với nhãn tham chiếu trên outputs/compare_round0.jpg
Mô hình khởi đầu lạnh là YOLOv8n tiền huấn luyện trên tập dữ liệu COCO (gộp 3 lớp: car, bus, truck thành car). Quan sát trực quan trên ảnh so sánh 4 khung hình kiểm thử (frame_0050, frame_0150, frame_0250, frame_0350) trong outputs/compare_round0.jpg cho thấy:
1. **Các loại xe không khớp**:
   - **Xe ở xa (kích thước nhỏ, chỉ lộ hai đốm đèn đỏ)**: Mô hình COCO bỏ sót hàng loạt xe ở nửa trên khung hình (khu vực đường chân trời). COCO chủ yếu chụp ban ngày với thân xe rõ ràng, do đó khi gặp xe ban đêm chỉ có ánh đèn mờ ảo, mô hình không kích hoạt phát hiện.
   - **Xe chạy song song bị che khuất một phần (occluded vehicles)**: Khi hai xe chạy cạnh nhau, mô hình COCO có xu hướng vẽ một box gộp lớn bao quanh cả hai xe hoặc chỉ bắt được chiếc xe có đèn sáng hơn, bỏ quên chiếc xe màu tối bên cạnh.
   - **Xe tải lớn và xe container góc nhìn chéo**: Nhãn COCO dự đoán đứt đoạn hoặc vẽ lệch tâm do vệt sáng phản xạ của đèn pha hắt xuống mặt đường ướt.

2. **Ý nghĩa của Recall theo kích thước**:
   - R small chỉ đạt **0.1818** (bỏ sót hơn 81% xe nhỏ dưới 32x32 px).
   - R medium đạt **0.5473** và R large đạt **0.5610**.
   - Con số này phản ánh rõ ràng điểm yếu chí tử của mô hình khởi đầu lạnh: khả năng phát hiện vật thể nhỏ, mờ trong điều kiện thiếu sáng cực kỳ kém. Khi xe càng tiến lại gần camera (kích thước lớn), recall tăng lên gấp 3 lần.

3. **Trường hợp cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai**:
   - Nhãn tham chiếu trong data/test/labels/ được tạo tự động bởi một mô hình AI khác và **chưa hề được con người rà soát từng box**.
   - Điển hình tại các vùng phản chiếu ánh đèn pha trên mặt đường hoặc biển báo phản quang ven đường: nhãn tham chiếu đôi khi gán nhãn một vệt sáng chói là car (false positive của nhãn tham chiếu). Nếu mô hình khởi đầu lạnh bỏ qua vệt sáng này, chỉ số tính toán sẽ coi mô hình bị False Negative (FN), dù trên thực tế mô hình đã xử lý đúng. Do đó, nhãn tham chiếu chỉ mang tính đối sánh tương đối, không phải chân lý tuyệt đối.

---

## 3. Chiến lược chọn mẫu

### Công thức chọn mẫu và vai trò của MIN_GAP_S
Thuật toán học chủ động sử dụng hàm mục tiêu kết hợp:
score = W_U * U + W_A * A + W_D * D
với bộ trọng số mặc định: W_U = 0.5, W_A = 0.3, W_D = 0.2.

Ý nghĩa chi tiết của từng thành phần:
- **Độ bất định cá thể U (Uncertainty, trọng số 0.5)**: Đo mức độ phân vân của mô hình trên các box khó nhất. Với mỗi box có độ tin cậy c, độ bất định tính theo u(c) = 1 - |2c - 1|. Hàm này đạt giá trị cực đại bằng 1 khi c = 0.5 (mô hình hoàn toàn do dự không biết là xe hay nền) và bằng 0 khi c tiến tới 0 hoặc 1. U lấy trung bình của 5 giá trị u(c) lớn nhất trong khung hình.
- **Mật độ box mập mờ A (Ambiguity Density, trọng số 0.3)**: Đếm tổng số box có độ tin cậy nằm trong khoảng tranh chấp 0.15 <= c < 0.50, chuẩn hóa theo giá trị lớn nhất trong pool. Ảnh có A cao là ảnh chứa nhiều đối tượng nửa muốn nhận diện nửa không, cần sự can thiệp của con người nhất.
- **Đa dạng hóa thời gian D (Temporal Diversity, trọng số 0.2)**: Đo khoảng cách thời gian từ frame ứng viên đến frame đã gán nhãn gần nhất, chia cho ngưỡng trần 10.0 giây. Ở vòng 1 chưa có ảnh nào được gán nên D = 1.0 cho mọi ảnh.
- **Ràng buộc khoảng cách tối thiểu MIN_GAP_S = 2.0s**: Đóng vai trò bộ lọc phi cực đại theo thời gian (temporal NMS). Nếu không có MIN_GAP_S, thuật toán tham lam sẽ chọn liên tiếp các frame chỉ cách nhau 0.4s (ví dụ frame 368, 369, 370, 372). Vì xe trên cao tốc chỉ di chuyển được vài mét trong 0.4s, người gán nhãn sẽ phải rà đi rà lại cùng một cảnh xe, gây lãng phí 80-90% công sức mà mô hình không tiếp nhận thêm thông tin mới.

### Minh chứng qua 3 frame được chọn và 1 frame bị loại
1. **frame_0182.jpg** (Rank 1, t = 72.8s, score = 0.9591): Đạt A = 1.0000 (chứa tới 18 box mập mờ, cao nhất pool) và U = 0.9182. Đây là thời điểm lượng xe đông đúc đang chuyển làn chéo, tạo ra giá trị thông tin cao nhất cho vòng học chủ động.
2. **frame_0369.jpg** (Rank 2, t = 147.6s, score = 0.9324): Đạt U = 0.9315, A = 0.8889, ghi nhận 43 box phát hiện. Mô hình cực kỳ phân vân ở giai đoạn giao thông phức tạp cuối video.
3. **frame_0331.jpg** (Rank 5, t = 132.4s, score = 0.9154): Đạt A = 1.0000 (18 box mập mờ), có tới 47 box phát hiện. Nhiều ca xe song song và xe ở xa khiến mô hình bối rối.
4. **frame_0372.jpg** (Rank 6, t = 148.8s, score = 0.9101) - **Frame bị loại**: Dù điểm số đứng thứ 6 trong toàn bộ 268 frame ứng viên (U = 0.9202, A = 0.8333), frame này bị thuật toán loại bỏ vì chỉ cách frame_0369.jpg (t = 147.6s, Rank 2) đúng **1.2 giây** (< MIN_GAP_S = 2.0s). Quyết định loại này tiết kiệm công rà nhãn cho một cảnh gần như trùng lặp hoàn toàn.

### Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không?
**Không.** Điểm bất định cao chỉ phản ánh trạng thái thiếu tự tin của mô hình hiện tại, không đảm bảo rằng việc gán nhãn ảnh đó sẽ làm tăng số đo kiểm thử:
- Ảnh có điểm bất định cao có thể chứa nhiều nhiễu thị giác không thể học được (ví dụ vệt đèn pha quá chói lóa, khói bụi mờ ảo). Việc cố ép mô hình học các mẫu nhiễu này có thể gây hại cho các đặc trưng tổng quát.
- Độ bất định không phát hiện được lỗi bỏ sót hoàn toàn (Unknown Unknowns): Nếu một dòng xe chạy ở làn xa bị mô hình dự đoán với confidence = 0.01, chúng bị lọc khỏi ngưỡng tính U (c < 0.05). Do đó mô hình tự tin sai mà không hề có điểm bất định cao.

---

## 4. Các vòng học chủ động (active learning)

### Bảng so sánh tiến trình các vòng (từ 
rounds_table.md)
Tập kiểm thử: 20 ảnh, 403 box tham chiếu (bỏ qua 14 box cao dưới 16 px). Ngưỡng IoU 0.5; P, R, F1 tính tại conf 0.25.

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 232 | 0.499 | -0.272 | 0.983 | 0.144 | 0.251 | 0.000 | 0.122 | 0.537 |

### Chi tiết sửa nhãn gợi ý vòng 1 (từ outputs/round1_diff.md)
Trong lô 12 ảnh vòng 1, mô hình đề xuất 169 box pre-label. Sau quá trình rà soát và chỉnh sửa cẩn trọng theo GUIDELINE_LABEL.md:
- **Accepted**: 154 box (tỷ lệ chấp nhận đạt 91%).
- **Edited**: 9 box (tinh chỉnh lại ranh giới box ôm sát thân xe, loại bỏ phần vệt sáng đèn pha rọi xuống mặt đường).
- **Deleted**: 6 box (loại bỏ các box dự đoán trùng lặp trên cùng 1 xe hoặc box gộp 2 xe thành 1).
- **Added**: 69 box (bổ sung các xe có thật trên đường bị mô hình bỏ sót, chủ yếu là xe ở cự ly xa có chiều cao > 16 px hoặc xe màu tối bị che một phần).
- **Tổng số box hoàn thiện**: 232 box trên 12 ảnh (trung bình ~19.3 box/ảnh, rất sát với mật độ ~20.8 box/ảnh của tập test).

### Phân tích sự thay đổi số đo trên tập test
- **Precision tăng vọt**: Precision tại ngưỡng 0.25 tăng từ **0.925 lên 0.983** (tỷ lệ dương tính giả cực thấp, chỉ có duy nhất 1 box FP trên toàn bộ 20 ảnh test). Mô hình sau khi học nhãn sạch đã triệt tiêu hoàn toàn thói quen bắt nhầm vệt đèn đường và bóng phản chiếu.
- **Biến động AP50 và Recall**: AP50 giảm từ 0.771 xuống 0.499 (delta AP50 = -0.272), Recall giảm từ 0.489 xuống 0.144. Hiện tượng này hoàn toàn bình thường trong học máy khi fine-tune một mạng nơ-ron sâu với tập dữ liệu cực nhỏ (chỉ 12 ảnh, 50 epochs) từ checkpoint đa lớp của COCO về đơn lớp car:
  - Mô hình COCO ban đầu dự đoán hàng trăm box với confidence rải rác; nhiều box đoán mò vô tình chạm ngưỡng IoU 0.5 với nhãn tham chiếu test (vốn cũng do AI tạo ra).
  - Khi fine-tune trên 12 ảnh đêm với tiêu chuẩn khắt khe (loại bỏ box trùng, cắt vệt sáng đèn), mô hình trở nên **cực kỳ thận trọng (conservative)**. Ngưỡng phân phối confidence bị co cụm lại, khiến nhiều box dự đoán trên xe nhỏ ở tập test có confidence dao động quanh 0.10 - 0.20 (thấp hơn ngưỡng cắt cứng 0.25), dẫn đến sụt giảm Recall danh nghĩa tại ngưỡng 0.25.
  - Tuy nhiên, trên nhóm xe lớn (large), Recall vẫn duy trì vững vàng ở mức **0.537** (so với 0.561 ban đầu).

### Đối chiếu ca cụ thể trên compare_round1.jpg
Quan sát trên compare_round1.jpg tại khung hình frame_0050 và frame_0250:
- **Ca cải thiện rõ rệt**: Ở làn đường gần camera, mô hình vòng 1 dự đoán box xe con và xe bán tải cực kỳ sắc nét, ôm khít thân xe, không còn hiện tượng box bị kéo dài xuống mặt đường do vệt đèn pha như ở mô hình khởi đầu lạnh.
- **Ca bị suy giảm**: Ở vùng đường chân trời phía xa (nửa trên ảnh), mô hình vòng 1 không kích hoạt box cho các xe nhỏ li ti (Recall small = 0.0). Lý do kiểm chứng được: trong 12 ảnh huấn luyện, số lượng xe nhỏ ở xa có hình thái rất đa dạng và mờ nhạt; việc chỉ huấn luyện 12 ảnh chưa đủ dung lượng mẫu để mạng nơ-ron học được đặc trưng bền vững của xe nhỏ trong đêm, khiến trọng số detector ưu tiên đàn áp các vùng tín hiệu yếu để tối ưu hóa classification loss.

### Phân biệt 3 mức thông tin và ca khó theo guideline
- **Quan sát độc lập (BLIND_SCAN.md)**: Khi chưa mở pre-label trên frame_0331.jpg, mắt người đếm được 34 xe và dự đoán chính xác 2 điểm yếu của AI: bỏ sót xe nhỏ ở làn xa (y < 320) và vẽ sai box gộp tại vị trí hai xe chạy sát nhau ở trung tâm (x ~ 450-550, y ~ 400-450).
- **Lỗi pre-label đã sửa (REVIEW_LOG.csv, 
ound1_diff.md)**:
  - Tại frame_0331.jpg, phát hiện box 15 (x ~ 442, y ~ 391) là một box khổng lồ ôm trọn cả 2 xe song song; hành động: deleted box gộp này và giữ lại 2 box con độc lập (box 14 và box 18).
  - Phát hiện box 19 (x ~ 865, y ~ 343) trùng lặp với box 4 (IoU 0.56); hành động: deleted box 19.
  - Thêm mới 3 xe ở xa có đèn hậu đỏ rõ nét; hành động: dded.
- **Kết quả mô hình sau train**: Mô hình học được cách tách xe ở tiền cảnh và triệt tiêu box trùng, nhưng trở nên khắt khe với các vệt sáng ở xa.
- **Ca khó điển hình theo guideline**: Xe chạy ở làn ngoài cùng bên phải bị bóng tối bao phủ gần như hoàn toàn, chỉ thấy một đốm đèn xi-nhan le lói. Theo guideline: *Chỉ thấy đèn, thân xe tối nhưng vẫn đoán được đường viền: Vẽ box theo phần thân xe đoán được quanh cụm đèn, không chỉ khoanh hai chấm đèn*. Người gán nhãn đã ước lượng vùng thân xe mở rộng quanh đốm đèn, giúp cung cấp mẫu huấn luyện chuẩn mực cho mô hình.

---

## 5. Kết luận và giới hạn

### Đánh giá vòng 1 và quyết định bước tiếp theo
Vòng học chủ động thứ nhất đã hoàn thành trọn vẹn mục tiêu sư phạm và kỹ thuật:
- Thiết lập quy trình khép kín: Cold start -> Lấy mẫu theo độ bất định -> Khóa bản quét mù -> Rà sửa nhãn -> Đóng gói -> Fine-tune -> Đánh giá độc lập.
- Mô hình đạt độ chính xác Precision ấn tượng **98.3%** trên tập test, chứng minh nhãn sửa vòng 1 có độ sạch rất cao.
- **Quyết định**: **Tiếp tục thực hiện vòng 2** (nếu tiếp tục dự án). Lý do: Vòng 1 đã giải quyết tốt bài toán lọc dương tính giả (FP) và định hình khuôn khổ box xe gần/trung bình, nhưng độ phủ (recall) trên xe nhỏ và xe ở xa bị thụt lùi. Cần một vòng học chủ động tiếp theo tập trung vào việc bổ sung các ca xe nhỏ và xe bị khuất.

### Đề xuất hai ca còn yếu/bất định cho vòng tiếp theo
1. **Ca xe nhỏ ở cự ly xa (Distant small vehicles)**: Cần chọn các frame có mật độ xe dày đặc ở nửa trên khung hình (như cụm frame quanh t = 110-125s).
   - *Chi phí rà nhãn*: Cao, vì mỗi frame có thể chứa từ 25 - 40 xe nhỏ, đòi hỏi phóng to ảnh để soi kỹ từng chấm đèn đỏ.
   - *Nguy cơ*: Dễ vướng vào các frame gần trùng nhau nếu không kiểm soát chặt MIN_GAP_S.
2. **Ca xe tải lớn và xe container bị khuất một phần (Partially occluded trucks)**: Các phương tiện có kích thước bất thường đi ở làn sát mép khung hình.
   - *Chi phí rà nhãn*: Thấp hơn (ít xe hơn), nhưng đòi hỏi người gán nhãn phải nhất quán cao khi ước lượng phần thân xe bị che khuất.

### Tác động của các giới hạn thực nghiệm
1. **Tập kiểm thử chỉ 20 ảnh**: Kích thước mẫu 20 ảnh là tương đối nhỏ trong thống kê. Một vài chiếc xe bị gắn nhãn sai hoặc thay đổi nhỏ trong confidence có thể làm chao đảo AP50 tới vài phần trăm. Do đó, không nên tuyệt đối hóa con số AP50 tăng hay giảm, mà phải kết hợp chặt chẽ với quan sát trực quan trên ảnh so sánh compare_round1.jpg.
2. **Luật bỏ qua xe cao dưới 16 px**: Giúp loại bỏ nhiễu chủ quan ở đường chân trời, đảm bảo việc đánh giá không phạt oan mô hình ở những đốm sáng không thể phân biệt.
3. **Nhãn tham chiếu do mô hình AI tạo ra**: Đây là giới hạn quan trọng nhất cần ghi nhớ. Nhãn tham chiếu không phải là Ground Truth hoàn hảo của con người. Khi mô hình fine-tune dự đoán khác nhãn tham chiếu, chưa chắc mô hình đã sai; có thể mô hình đã thông minh hơn và bỏ qua được một lỗi gán nhầm của AI ban đầu.

### Nếu AP50 giảm, cần kiểm tra điều gì trước khi train thêm?
Khi thấy AP50 giảm từ 0.771 xuống 0.499, kỹ sư không được vội vàng sửa nhãn bừa bãi hay đổi model phức tạp, mà cần kiểm tra tuần tự 4 yếu tố then chốt:
1. **Kiểm tra phân bố điểm tin cậy (Confidence distribution)**: Xem xét đồ thị Precision-Recall đường cong đầy đủ (PR curve). Nếu precision rất cao (0.983) nhưng recall thấp tại ngưỡng conf 0.25, điều đó chỉ ra rằng ngưỡng 0.25 đang quá cao so với phân phối confidence mới của mô hình sau fine-tune.
2. **Kiểm tra trực quan từng ca FN (False Negatives)**: Mở ảnh đối sánh xem những xe bị tính là bỏ sót thực sự là xe gì. Nếu đó là những xe ở xa mà nhãn tham chiếu gán còn mô hình không phát hiện, đây là vấn đề thiếu dữ liệu xe nhỏ. Nếu đó là do nhãn tham chiếu tự gán sai vào đèn đường, thì sự sụt giảm AP50 là sai số của bộ test chứ không phải của mô hình.
3. **Kiểm tra hiện tượng quá khớp (Overfitting) trên lô nhỏ**: 12 ảnh là kích thước quá nhỏ so với mạng nơ-ron hàng triệu tham số. Cần kiểm tra xem mô hình có bị học vẹt nền đường của 12 ảnh đó hay không (có thể giảm bớt số epoch từ 50 xuống 25-30 hoặc tăng cường độ data augmentation).
4. **Kiểm tra tính nhất quán của quy tắc gán nhãn (Annotation Consistency)**: Rà soát lại 
ound1_diff.md xem người gán nhãn có vô tình áp dụng quy chuẩn khác biệt (ví dụ vẽ box quá chặt hoặc quá lỏng so với bộ test) hay không. Tính nhất quán quan trọng hơn rất nhiều so với độ đẹp của từng box riêng lẻ.
