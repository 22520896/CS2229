# Demo local

Cải tiến từ [LoylP/tabular-trees](https://github.com/LoylP/tabular-trees/tree/e9fce1eba2fb498b9571b33a331832a3eef63626).
Giữ năm nhóm demo, Streamlit/Plotly và màu model; bổ sung kết quả tự chạy và phạm vi paper.

## Chạy

Python 3.11–3.12; CPU đủ. Từ thư mục repository:

```powershell
python -m venv .venv-demo
.venv-demo\Scripts\python.exe -m pip install -r demo/requirements.txt
.\demo\run.ps1 -Python .venv-demo\Scripts\python.exe
```

Mở http://127.0.0.1:8503/. `-Port 8504` để đổi cổng.
`requirements-lock.txt` lưu môi trường đã kiểm tra. Chỉ bind localhost, không telemetry.

## Ba trang

**Demo tương tác** giữ năm module của bản gốc:

| Module | Thao tác |
| --- | --- |
| Benchmark | Xem đường Paper đã kiểm chứng; đọc CSV/CSV.GZ hoặc upload CSV, đổi nhóm, số hoán vị và ngân sách, tính lại phân tích. Có đường default và trục log. |
| Decision boundary | Chuẩn bị dữ liệu; chọn hai feature hoặc tính top-2 theo RF; chạy RF/MLP; chỉnh vùng nhìn mà không train lại. |
| Target smoothing | Chọn hệ số covariance, số feature và covariance thường/robust; chạy và xem tỷ lệ nhãn train đổi. |
| Noise / removal | Chọn tỷ lệ thêm/bỏ feature; tùy chọn GBT học trên feature bị loại. |
| Rotation | Có/không xoay và tùy chọn bỏ 50% feature trước khi xoay. |

Dataset, target, số mẫu train/test, seed và model nằm trong **Dữ liệu demo** ở sidebar.
Dataset công khai tải từ mirror Hugging Face được bản gốc sử dụng; cache tại
`.cache/demo-live/`. Nhấn **Tải dataset** nếu chưa có, rồi **Chuẩn bị dữ liệu**.
Model chỉ train sau khi bấm **Chạy demo** hoặc nút tính RF importance. Một action,
một luồng CPU; giới hạn RAM và kích thước tải bảo vệ máy local. Không chạy Kaggle.

Live demo dùng đúng lõi RF/GBT/MLP scikit-learn của repo tham khảo, giữ MLP 4×256,
200 epoch tối đa và early stopping. Đây là minh họa mẫu nhỏ, không HPO; không được
coi là train lại ResNet/FT-Transformer hoặc là tái hiện đầy đủ paper. Cấu hình live
không thay đổi các kết quả V2 đã lưu. Demo 2D không được ghi là đã tái hiện Figure 19.

**Kết quả tự chạy** tập trung Rotation, Noise features và Target smoothing.
Chọn tổng hợp/từng dataset, normalized Paper/Ours hoặc accuracy thô, ẩn/hiện model.
Đường liền là Paper; đứt là Ours. Có CSV, chi tiết validation/test/seed, JSON cấu hình
và link notebook. Confusion matrix chỉ hiện ở ba mẫu RF/Electricity có numeric evidence.
Benchmark tự chạy ít ngân sách được giữ dưới expander/bảng chứng cứ; không làm plot chính.
Feature removal đại diện vẫn được lưu, không gắn nhãn full-paper.

**Tiến độ theo paper** dùng phạm vi thí nghiệm gốc, không dùng kế hoạch chạy đại diện
làm mẫu số. Bảng nêu phần đã làm, yêu cầu paper và phần còn thiếu. **Chưa định lượng** nghĩa là
chưa có mẫu số đầy đủ/comparable để cho một tỷ lệ tự train; không tự suy ra 100% từ
168/168, 224/224 hoặc 240/240 ô của grid một cấu hình công bố/điều kiện.
Khối lượng grid/HPO đã chạy nằm trong expander riêng và không có % paper giả định.
27/40 Figure IDs (67,5%) chỉ là độ phủ **phân tích CSV tác giả**, được hiển thị riêng.
Danh mục có Figure 1–40, Table 1–8 và các mục phụ lục, với tìm kiếm và nguồn notebook.

## Nguồn và kiểm chứng

`upstream/` giữ app và sáu file lõi của demo gốc **nguyên byte**, commit
`e9fce1eba2fb498b9571b33a331832a3eef63626`, kiểm tra bằng `source.json`.
UI được tổ chức lại; adapter chỉ giới hạn RF/BLAS ở một luồng, ghi warning và kiểm soát
khi chạy. Model/preprocessing/transforms của demo gốc không bị viết lại.
Riêng phân tích CSV benchmark dùng `src.analysis` đã được repository đối chiếu R,
thay cho sampler NumPy giản lược của demo tham khảo; giữ default-first, R RNG,
normalization trước khi ẩn model, và validation-only selection. Test kiểm tra parity.

Snapshot kết quả tự train giữ nguyên **02:00 ngày 28/11/2026**, baseline `d78cfd6`.
Ba grid là một cấu hình validation-selected công bố/điều kiện, đủ original auto seeds;
chưa replay toàn bộ HPO/shuffles và chưa full paper. Không sửa PDF trong lượt này.
Paper/Ours ba cơ chế dùng cùng cohort và mốc normalization tác giả; không clip điểm.
Bảng per-dataset Figure18/21/23/24 được chuẩn hóa để khớp Figure2/4b/5 tại `1e-12`.
Ẩn model không thay mốc. Benchmark của hai nguồn vẫn giữ cohort/ngân sách riêng.

`data/manifest.json` khóa 32 nguồn đã lưu. Hash CSV/JSON chỉ chuẩn hóa xuống dòng
CRLF thành LF; numeric NPZ đối chiếu nguyên byte. Thiếu hoặc sửa nguồn sẽ báo lỗi.
`prepare_data.py` chỉ là công cụ maintainer; app không gọi nó. Không upload dữ liệu
live, CSV người dùng, weights hoặc cache khi push repository.

## Kiểm tra

```powershell
python -m unittest discover -s demo/tests -v
```

[Test và giới hạn](VERIFICATION.md). Mọi output của smoke hiển thị cho người dùng
đều tính từ dữ liệu thật; dữ liệu sinh trong unit test không được dùng làm kết quả demo.

```text
demo/
  app.py                 navigation và kết quả đã lưu
  live.py, live_ui.py     adapter và UI cho demo gốc
  benchmark*.py          CSV và engine phân tích đã đối chiếu R
  evidence.py, charts.py đọc bằng chứng và plot
  inventory.py          phạm vi paper và danh mục
  progress_ui.py        tiến độ, không lấy pilot làm mẫu số
  upstream/             source demo gốc đã pin
  data/                 hash, normalization và trạng thái snapshot
  tests/                parity và các luồng app
  run.ps1               khởi động localhost
```
