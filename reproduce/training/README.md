# Train mẫu bằng source frozen

Phần này cung cấp code/config để **chạy lại một RF fit đại diện** của Noise,
Rotation hoặc Target smoothing. Không phải một trình tối ưu mới hay model tự viết.

`source/` giữ các file gốc của dự án nghiên cứu, gồm đủ18file lõi và các worker
`representative_pilot`, `published_pilot_worker`, repeat adapter, data/evaluator và
transport. Hash ở `../configs/frozen_training_sources.json`. Phần model thực tế
được import từ upstream tác giả tại commit `9d54cf53...` sau khi clone.

## Môi trường đã pin của dự án

Lệnh train mẫu dùng **Python 3.8**, package/version và `uv.lock` gốc. Không cài bộ Torch cũ
vào kernel Python 3.11+ đang dùng để đọc notebook.

```powershell
cd reproduce/training/source
uv sync
.venv\Scripts\python.exe ../../scripts/train_sample.py --prepare
.venv\Scripts\python.exe ../../scripts/train_sample.py --run noise_features
```

Thay `noise_features` bằng `rotation` hoặc `target_smoothing` cho hai ví dụ còn lại.
Trên Linux/macOS dùng `.venv/bin/python`.

`--prepare` clone repo tác giả nếu chưa có, kiểm231hash nguồn và tải đúng OpenML
task/version/checksum công khai. Không sửa checkout đã tồn tại nếu khác commit.
`--run` gọi worker đã có với cấu hình công bố và seed của mẫu, trần1800giây/fit,
không tự thử lại. Cả ba mẫu đều RF/CPU; giữ nguyên cấu hình, không thay sang GPU estimator.
Weights/checkpoint, raw input và output mới chỉ nằm trong `../.runtime/`, bị Git bỏ qua.

## Đã chạy hay chưa?

Ba bộ numeric trong `../results/numeric/` là **output lịch sử đã train thật và audit**.
Lượt đóng gói đã kiểm lại các giá trị numeric và chạy notebook phân tích; **chưa thực thi
ba lệnh train lại ở trên**, chưa chạy thêm benchmark. Output mới không tự ghi đè bộ
golden đã xuất trong `results/` và không tự được cộng vào PDF.

Mã API đi kèm trong18file frozen chỉ phục vụ tính toàn vẹn source snapshot; script
mẫu không bật web server, cloud, database hay authentication. W&B ở chế độ disabled.

Các output lịch sử được tạo trên nhiều môi trường CPU/GPU tương thích đã kiểm chứng.
Pin Python 3.8 ở đây mô tả môi trường wrapper đóng gói, không khẳng định mọi lượt
lịch sử đã dùng cùng interpreter.
