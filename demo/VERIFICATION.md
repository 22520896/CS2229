# Kiểm chứng demo · 28/11/2026

Windows/Python 3.12.14; dependencies trong `requirements-lock.txt`.

| Kiểm tra | Kết quả |
| --- | --- |
| `python -m unittest discover -s demo/tests -v` | 27/27 pass |
| Test hiện có `tests/` | 9/9 pass |
| Test hiện có `reproduce/tests/` | 4/4 pass |
| Ruff và Mypy · 9 module ứng dụng | Pass |
| Source demo gốc | App, README, dependencies và sáu core files nguyên byte, đúng commit/hash |
| Live smoke · Electricity thật | 13 fit thành công: boundary RF/MLP, smoothing, add, removal, rotation và pruned rotation |

Smoke dùng 1.000 train / 500 test, seed 0; tổng khoảng 5,84 giây trong lần đo.
Đây là phép thử tính năng nhỏ, không phải ước lượng thời gian full benchmark.
Test so adapter với trực tiếp core gốc: preprocessing, các score và labels khớp;
nhãn test không đổi. RF/BLAS chỉ giới hạn số luồng, không đổi estimator/formula.
Kiểm tra độc quyền một action và không train do mở trang/đổi bộ lọc.

32 nguồn kết quả giữ hash. Đường Paper main/per-dataset khớp `1e-12` cho Figure2,
4b và 5 (hai mức pruning). CSV/JSON round-trip, seed/repeats và numeric samples
được kiểm tra. Hash CSV/JSON hỗ trợ CRLF/LF; NPZ vẫn byte-exact.
CSV benchmark được kiểm schema và tính bằng engine đã đối chiếu R; normalization
không phụ thuộc bộ chọn ẩn model. Giữ separate source/cohort và không ngoại suy HPO.

AppTest kiểm tra năm module gốc, prepare/run, giữ output/state qua đổi trang,
empty/error states và nguồn thiếu/hash sai. Không lưu trạng thái button/download
như input để tránh kích hoạt lại action hoặc lỗi read-only widget.
Bảng paper scope không có giá trị 100% cho grid đại diện: các tỷ lệ thiếu mẫu số
là null; phân tích CSV tác giả nằm riêng. Unit fixtures không dùng làm output sản phẩm.

Không sửa thuật toán/kết quả V2, notebook, numeric source hay PDF. Không gửi job
Kaggle/Dell/Desktop và không bật schedule. Local live chỉ chạy các smoke nhỏ và
các action do người dùng bấm. Chưa kiểm tra bằng screen reader hoặc thiết bị mobile.
Streamlit không có bước production build riêng; kiểm bằng khởi động, HTTP health,
AppTest, lint và type-check. Server local được giữ chạy theo yêu cầu host của người dùng.

Đã xem trực tiếp trong Codex: RF/MLP fit Electricity, hai heatmap và train/test score; bảng tiến độ theo paper, chi tiết phạm vi và trạng thái chưa định lượng.
