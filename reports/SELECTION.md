# Vì sao chọn lô này?

Trong 50 dòng đứng đầu outputs/selection_round1.csv, chọn năm frame bạn sẽ ưu tiên nếu chỉ có
ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét
ảnh gần trùng hoặc trường hợp model không dự đoán được box:

Nếu chỉ có ngân sách rà 5 ảnh, việc chọn mẫu không thể lấy đơn thuần top 5 rank đầu (vì rank 2, 6 và rank 4, 5 rất sát nhau về thời gian, gây lặp thông tin). Danh sách 5 frame ưu tiên tối ưu nhất gồm:
1. **frame_0182.jpg** (Rank 1, t = 72.8s, Score = 0.9591, U = 0.9182, A = 1.0000, D = 1.0000, 28 boxes, 18 ambiguous): Frame có điểm bất định tổng hợp cao nhất toàn bộ tập pool. Số lượng box mập mờ đạt mức tối đa (A = 1.0) tại đoạn giữa video khi mật độ phương tiện đông đúc, góc nhìn xe chuyển hướng trên làn đường mang lại giá trị thông tin cao nhất.
2. **frame_0369.jpg** (Rank 2, t = 147.6s, Score = 0.9324, U = 0.9315, A = 0.8889, D = 1.0000, 43 boxes, 16 ambiguous): Frame có số lượng box lớn (43 boxes) ở giai đoạn cuối video (t = 147.6s). Độ bất định U rất cao (0.9315). Đây là frame đại diện tuyệt vời cho mật độ giao thông dày đặc.
3. **frame_0326.jpg** (Rank 4, t = 130.4s, Score = 0.9155, U = 0.9310, A = 0.8333, D = 1.0000, 39 boxes, 15 ambiguous): Đại diện cho cụm giao thông t ~ 130s. Chọn frame này và **bỏ qua frame_0331.jpg** (Rank 5, t = 132.4s, Score = 0.9154) và **bỏ qua frame_0330.jpg** (Rank 12, t = 132.0s) vì chúng cách nhau chỉ 0.4s - 2.0s; việc gán cả hai ảnh gần trùng này gây lãng phí 20% ngân sách mà không bổ sung thêm tri thức mới cho mô hình.
4. **frame_0099.jpg** (Rank 8, t = 39.6s, Score = 0.9063, U = 0.9460, A = 0.7778, D = 1.0000, 29 boxes, 14 ambiguous): Độ bất định U lên tới 0.9460 (thuộc nhóm cao nhất pool). Frame này nằm ở giai đoạn sớm (t = 39.6s), giúp phân bổ đều dữ liệu huấn luyện theo trục thời gian thay vì dồn hết vào nửa cuối video.
5. **frame_0227.jpg** (Rank 11, t = 90.8s, Score = 0.8915, U = 0.9164, A = 0.7778, D = 1.0000, 37 boxes, 14 ambiguous): Nằm ở khoảng thời gian t = 90.8s, lấp khoảng trống thời gian giữa frame_0182 (72.8s) và frame_0326 (130.4s). Mô hình có độ phân vân cao (U = 0.9164) với 37 xe trong khung hình.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet:
- **frame_0182.jpg** (Rank 1, Score 0.9591): Trên CSV, frame này đạt điểm tối đa ở thành phần A = 1.0 (18 box có độ tin cậy nằm trong vùng mập mờ 0.15 - 0.50). Trên contact sheet selection_round1.jpg, ảnh cho thấy dòng xe đang chuyển làn với nhiều góc nhìn chéo, đèn xe tạo bóng phản chiếu phức tạp khiến model dao động mạnh giữa car và nền.
- **frame_0331.jpg** (Rank 5, Score 0.9154): CSV ghi nhận tới 47 boxes phát hiện và 18 ambiguous boxes. Trên contact sheet, đây là frame có mật độ xe dày đặc nhất, xuất hiện nhiều ca xe chạy song song sát nhau ở cự ly trung bình và các xe chỉ có hai chấm đèn mờ ở làn xa.
- **frame_0392.jpg** (Rank 15, Score 0.8874): CSV cho thấy chỉ số U cao kỷ lục 0.9747 (trung bình 5 box khó nhất có độ bất định gần như tuyệt đối, conf sát 0.50). Trên contact sheet, frame ở cuối video (t = 156.8s) ghi nhận các xe tải lớn và xe con ở mép khung hình bị khuất một phần khiến mô hình cực kỳ phân vân.

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:
- **frame_0372.jpg** (Rank 6, Score = 0.9101, U = 0.9202, A = 0.8333, n_boxes = 42) có điểm số đứng thứ 6 trong toàn bộ 268 frame ứng viên, cao hơn nhiều frame được chọn như frame_0312 (Rank 7), frame_0099 (Rank 8). Tuy nhiên, frame này **không được chọn** vì thời điểm t = 148.8s của nó chỉ cách frame_0369.jpg (t = 147.6s, đã được chọn ở Rank 2) đúng **1.2 giây**, vi phạm ràng buộc khoảng cách tối thiểu MIN_GAP_S = 2.0s. Trong 1.2s, các xe trên cao tốc chỉ dịch chuyển vài mét, bối cảnh cảnh nền và ánh sáng giữ nguyên; việc thuật toán loại bỏ frame này là hoàn toàn chính xác để tránh lãng phí chi phí gán nhãn vào dữ liệu thừa thãi (redundant data).
- Tương tự, **frame_0330.jpg** (Rank 12, Score = 0.8899, có tới 53 boxes - cao nhất toàn pool) cũng bị loại vì cách frame_0331.jpg chỉ 0.4 giây.

Điều phép chọn này chưa chứng minh về chất lượng mô hình:
- Phép chọn mẫu theo Uncertainty Sampling chỉ phản ánh **độ phân vân nội tại của mô hình hiện tại** (các box có confidence xấp xỉ 0.5), hoàn toàn không đồng nghĩa với:
  1. **Không chứng minh các box được chọn là sai hay đúng**: Một box có conf = 0.5 có thể là một xe thật sự (đúng nhưng thiếu tự tin) hoặc một bóng đèn đường (dự đoán sai).
  2. **Không phản ánh những lỗi mà mô hình hoàn toàn không biết (Unknown Unknowns)**: Nếu mô hình bỏ sót hoàn toàn một chiếc xe (confidence < 0.05), chiếc xe đó không hề đóng góp vào U hay A. Do đó, uncertainty sampling có xu hướng thiên vị các vùng mô hình đã ngờ ngợ thấy hơn là những tình huống mới lạ bị bỏ sót hoàn toàn.
  3. **Không đảm bảo AP50 trên tập test sẽ tăng**: Việc đưa 12 ảnh có độ bất định cao vào fine-tune có thể khiến mô hình bị co cụm (overfit) vào phân phối ánh sáng cục bộ của lô ảnh đó, hoặc trở nên quá thận trọng làm giảm recall trên tập kiểm thử độc lập.
