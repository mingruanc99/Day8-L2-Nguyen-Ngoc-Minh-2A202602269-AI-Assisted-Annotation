# Quét độc lập trước khi xem pre-label

Frame: to_label/round1/images/train/frame_0331.jpg

Số xe nhìn thấy bằng mắt: 34

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:
1. Cụm xe ở làn đường xa phía trên bên trái (khu vực y < 320 px), xe kích thước nhỏ chỉ thấy hai chấm đèn hậu đỏ mờ nhạt dễ bị AI bỏ sót do độ tương phản thấp.
2. Khu vực giữa đường gần trung tâm (x ~ 450-550, y ~ 400-450) có hai xe chạy song song sát nhau dễ bị AI vẽ một box gộp to hoặc bị trùng lặp box; và xe bị khuất ở mép phải ảnh.

Chạy python tools/lock_blind.py ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
