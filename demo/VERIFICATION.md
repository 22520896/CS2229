# Kiểm chứng demo · 28/11/2026

Môi trường: Windows, Python 3.12.14; phiên bản dependency trong
`requirements-lock.txt`. Demo chỉ đọc kết quả đã lưu, không training hay gửi job.

| Kiểm tra | Kết quả |
| --- | --- |
| `python -m unittest discover -s demo/tests -v` | 16/16 pass |
| Test hiện có tại `tests/` | 9/9 pass |
| Test hiện có tại `reproduce/tests/` | 4/4 pass |
| Ruff · lỗi import, cú pháp và tên biến | Pass |
| Mypy · app, evidence, charts, inventory | Pass |
| PowerShell parser · `run.ps1` | Pass |
| Local HTTP / health và giao diện trong Codex | Pass |

Các test kiểm tra 32 hash nguồn, 632 ô / 1.556 fit của ba lưới công bố,
seed/repeats, raw accuracy và dữ liệu mẫu numeric. Đường Paper tính từ bảng
per-dataset khớp Figure 2, 4b và 5 tại `1e-12`, gồm cả hai mức loại feature
của Rotation. Mốc chuẩn hóa không đổi khi ẩn model. CSV/JSON giữ nguyên giá trị
khi export; thiếu hoặc sửa nguồn sẽ báo lỗi và không hiện biểu đồ.
Hash CSV/JSON chuẩn hóa riêng ký tự xuống dòng CRLF/LF, có test portability;
numeric NPZ vẫn được đối chiếu nguyên byte.

AppTest kiểm tra năm nhóm thực nghiệm, đổi nguồn benchmark, bộ lọc, tìm kiếm,
đổi trang và giữ state, empty state, error state, bảng mẫu và nút download.
Đã xem trực tiếp biểu đồ, bảng tiến độ và tìm Figure 19 trong trình duyệt Codex.
Chưa kiểm tra trực tiếp trên thiết bị mobile hoặc bằng screen reader.

Một lần đo server local dùng khoảng **156 MiB RAM**, không có tiến trình con
training; đây là số đo của phiên kiểm tra, không phải cam kết tài nguyên.
Streamlit không có bước production build riêng: kiểm tra bằng khởi động app,
HTTP health, type-check và AppTest. Server QA được dừng sau khi kiểm tra.

Không sửa thuật toán, model, preprocessing, optimizer, seed, cấu hình training,
kết quả gốc hoặc PDF. Snapshot vẫn là 02:00 ngày 28/11/2026; chưa tích hợp kết quả
sau cutoff. 27/40 chỉ là phân tích Figure từ CSV tác giả; ba lưới dùng cấu hình
công bố không đồng nghĩa replay đầy đủ HPO hay hoàn thành full paper.
