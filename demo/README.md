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
| Target smoothing | Chọn nhiều hệ số và covariance thường/robust để so cùng plot; xem tỷ lệ nhãn train đổi. |
| Noise / removal | Chọn nhiều tỷ lệ thêm/bỏ; chọn nhánh giữ lại/bị loại hoặc cả hai (nhánh bị loại dùng GBT). |
| Rotation | Chọn giữ toàn bộ / bỏ 50% feature hoặc cả hai; cùng plot, split và seed. |

Dataset, target, số mẫu train/test, seed và model nằm trong **Dữ liệu demo** ở sidebar.
Dataset công khai tải từ mirror Hugging Face được bản gốc sử dụng; cache tại
`.cache/demo-live/`. Nhấn **Tải dataset** nếu chưa có, rồi **Chuẩn bị dữ liệu**.
Ba dataset **electricity, MagicTelescope, bank-marketing** đã được chuẩn bị trên máy này;
danh sách đưa các dataset có cache lên đầu, đánh dấu **🟢 Có sẵn**. Các máy clone mới
vẫn cần tải; trạng thái được xác định từ file thực tế, không gán cứng.
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

**Tiến độ theo paper** có **% ước lượng khối lượng cấu hình**, dùng ngân sách của
paper/repo gốc, không lấy số case trong queue làm mẫu số. Công thức là:

`100 × cấu hình đủ original repeats đã kiểm chứng / ngân sách cấu hình tham chiếu`.

Ba cơ chế dùng cohort/model/điều kiện từ CSV nguồn đã kiểm hash, nhân với ngân sách
HPO ở hạng 20/30/60 của script. Vì chỉ chạy một cấu hình công bố mỗi điều kiện,
Noise và Rotation ước lượng 5%, Smoothing khoảng 1,67% **về khối lượng HPO**;
đủ grid điều kiện không thành 100% paper. Hạng có default/rank gaps nên đây là ước
lượng ngân sách, không phải đếm chính xác lịch sử, độ giống kết quả hay thời gian.
Số hoán vị không nhân vào lượt train vì dùng lại output đã có.

Benchmark: tổng cấu hình đủ seed trong snapshot chia ngân sách 400 cho 437 cặp
model-dataset của Figure1 (35 numeric setting × 7 model + 24 categorical × 8).
Mẫu số không bị thu xuống catalog 57 setting của nhóm. 50K dùng 38.006 nhóm HPO
đủ repeats đã audit và cùng mốc ngân sách; đây là ước lượng từ tổng số, chưa chứng
minh paired/cohort parity. Những nhánh chưa có tự train đúng phạm vi là 0%.
Tài liệu/hồi quy từ dữ liệu tác giả không có % tự train; được báo riêng. Không cộng
các Figure dùng chung training thành một tỷ lệ full paper.

Mỗi hàng mở được **Phạm vi gốc & phần còn thiếu**, có tử số/mẫu số, nguồn và giả định.
`data/progress_targets.json` lưu grid của tác giả; `prepare_progress.py` chỉ dành cho
maintainer, không được app gọi. 27/40 Figure IDs (67,5%) chỉ là độ phủ **phân tích
CSV tác giả**, hiển thị riêng. Danh mục có Figure1–40, Table1–8 và các mục phụ lục.

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

`data/manifest.json` khóa 33 nguồn đã lưu. Hash CSV/JSON chỉ chuẩn hóa xuống dòng
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
  progress_ui.py        tiến độ ước lượng, không lấy pilot làm mẫu số
  progress_estimates.py công thức và tử số/mẫu số theo paper
  upstream/             source demo gốc đã pin
  data/                 hash, normalization và trạng thái snapshot
  tests/                parity và các luồng app
  run.ps1               khởi động localhost
```
