# Kiểm chứng demo · 28/11/2026

Windows/Python 3.12.14; dependencies trong `requirements-lock.txt`.

| Kiểm tra | Kết quả |
| --- | --- |
| `python -m unittest discover -s demo/tests -v` | 31/31 pass |
| Test hiện có `tests/` | 9/9 pass |
| Test hiện có `reproduce/tests/` | 4/4 pass |
| Ruff và Mypy · 10 module ứng dụng | Pass |
| Source demo gốc | App, README, dependencies và sáu core files nguyên byte, đúng commit/hash |
| Live smoke · Electricity thật | 13 fit thành công: boundary RF/MLP, smoothing, add, removal, rotation và pruned rotation |

Smoke dùng 1.000 train / 500 test, seed 0; tổng khoảng 5,84 giây trong lần đo.
Đây là phép thử tính năng nhỏ, không phải ước lượng thời gian full benchmark.
Test so adapter với trực tiếp core gốc: preprocessing, các score và labels khớp;
nhãn test không đổi. RF/BLAS chỉ giới hạn số luồng, không đổi estimator/formula.
Kiểm tra độc quyền một action và không train do mở trang/đổi bộ lọc.

33 nguồn kết quả giữ hash. Đường Paper main/per-dataset khớp `1e-12` cho Figure2,
4b và 5 (hai mức pruning). CSV/JSON round-trip, seed/repeats và numeric samples
được kiểm tra. Hash CSV/JSON hỗ trợ CRLF/LF; NPZ vẫn byte-exact.
CSV benchmark được kiểm schema và tính bằng engine đã đối chiếu R; normalization
không phụ thuộc bộ chọn ẩn model. Giữ separate source/cohort và không ngoại suy HPO.

AppTest kiểm tra năm module gốc, prepare/run, giữ output/state qua đổi trang,
empty/error states và nguồn thiếu/hash sai. Không lưu trạng thái button/download
như input để tránh kích hoạt lại action hoặc lỗi read-only widget.
Bảng paper scope không có giá trị 100% cho grid đại diện: các % là ước lượng ngân sách cấu hình theo source gốc, có công thức và mẫu số;
phân tích CSV tác giả nằm riêng. Unit fixtures không dùng làm output sản phẩm.

Không sửa thuật toán/kết quả V2, notebook, numeric source hay PDF. Không gửi job
Kaggle/Dell/Desktop và không bật schedule. Local live chỉ chạy các smoke nhỏ và
các action do người dùng bấm. Chưa kiểm tra bằng screen reader hoặc thiết bị mobile.
Streamlit không có bước production build riêng; kiểm bằng khởi động, HTTP health,
AppTest, lint và type-check. Server local được giữ chạy theo yêu cầu host của người dùng.

Đã xem trực tiếp trong Codex: RF/MLP fit Electricity, hai heatmap và train/test score; bảng tiến độ theo paper, chi tiết phạm vi và trạng thái chưa định lượng.

Cập nhật multi-case: 31 demo tests (44 tổng với hai bộ cũ). Rotation chọn đồng thời
pruning0/50, Smoothing chọn covariance thường/robust, Removal chọn nhánh giữ/bỏ.
Adapter từng nhánh khớp lời gọi core gốc; plot gộp giữ variant và không gộp nhầm mean.
Unit test kiểm tra một action, state, lựa chọn rỗng, số series và phép tính tiến độ.
Smoke mới trên 3 dataset thật: 12 RF rotation fits + 4 RF covariance fits hoàn tất;
4,46 giây trong lần đo này, không phải benchmark hiệu năng. Điểm smoke không cộng
vào tiến độ. CSV cache không commit; source URL/hash/rows lưu ở dataset_sources.json.
Progress targets được xuất từ projection ablation đã kiểm SHA-256, budget lấy từ
src.analysis; numerator chỉ đếm cấu hình đủ seed. Benchmark giữ 59 setting nhãn
gốc, 437 cặp model-dataset × 400. Kiểm test Noise/Rotation5%, Smoothing100/60%,
50K numerator38.006; không gán 100% từ một grid cấu hình chọn sẵn.

Đã kiểm trực tiếp trong Codex: ba dataset xanh ở đầu dropdown, Rotation chạy cả giữ toàn bộ/bỏ50% trên cùng plot với 4 series RF/GBT, bảng % ước lượng và cách tính.
