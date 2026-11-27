# Demo local

**Hai trang: Thực nghiệm và Tiến độ & bằng chứng.** Giữ nền sáng, bố cục
Streamlit/Plotly và màu model từ demo [LoylP/tabular-trees](https://github.com/LoylP/tabular-trees/tree/e9fce1eba2fb498b9571b33a331832a3eef63626).
Không tự train, tải dataset hay gửi job khi mở trang hoặc đổi bộ lọc.

## Chạy

Tại thư mục repository, dùng Python **3.11–3.12**, CPU đủ:

```powershell
python -m venv .venv-demo
.venv-demo\Scripts\python.exe -m pip install -r demo/requirements.txt
.\demo\run.ps1 -Python .venv-demo\Scripts\python.exe
```

Mở **http://127.0.0.1:8503/**. Đổi cổng bằng `-Port 8504`.
Hoặc chạy trực tiếp:

```powershell
python -m streamlit run demo/app.py --server.address 127.0.0.1 --server.port 8503 --server.headless true --browser.gatherUsageStats false --theme.base light
```

`demo/requirements-lock.txt` lưu phiên bản đã kiểm tra trên Windows/Python 3.12.
Không cần PyTorch, CUDA hay model weights. Dữ liệu dùng lại từ `reproduce/`,
`figures/` và `evidence/`; chạy được offline sau khi cài dependencies.

## Dùng

- **Thực nghiệm:** chọn Rotation, Noise features hoặc Target smoothing. Xem tổng hợp,
  từng dataset, raw accuracy hay chuẩn hóa; bật/tắt model và xuất CSV.
- **Chi tiết case:** validation/test, đủ seed, cấu hình công bố và nguồn output.
  Confusion matrix chỉ xuất hiện cho ba mẫu RF/Electricity có numeric evidence.
- **Benchmark:** chọn Paper hoặc Ours và nhóm dữ liệu. Hai nguồn giữ ngân sách,
  tập dataset và normalization riêng; không overlay một phép so chưa cân bằng.
- **Feature removal:** các case đại diện đã lưu, không tuyên bố full grid.
- **Tiến độ:** độ phủ với mẫu số/đơn vị từng hàng; danh mục Figure 1–40,
  Table 1–8 và các mục phụ lục. Có tìm kiếm, bộ lọc và link notebook.

Các lựa chọn được giữ khi đổi trang. Mở hoặc đổi tab không gọi `fit`.
Không thêm upload CSV bất kỳ hoặc dự đoán dữ liệu mới bằng model đã bị dọn weights.

## Dữ liệu và phạm vi

- Snapshot số liệu: **02:00 ngày 28/11/2026**, commit
  `d78cfd6a9e35af50862b2c37fc1c62e52ae0ee48`. Training đã dừng theo yêu cầu người dùng.
  Kết quả sau cutoff không tự cộng; không rebuild PDF trong lượt làm demo.
- Ba grid dùng một cấu hình validation-selected tác giả công bố cho mỗi điều kiện,
  đủ original auto repeats. Không replay toàn bộ HPO/shuffles và chưa full paper.
- Paper/Ours ở ba cơ chế dùng cùng dataset và mốc chuẩn hóa **tác giả**, theo
  `(score − q10) / (max − q10)` từng dataset/experiment. Bộ chọn model chỉ ẩn/hiện,
  không tính lại mốc. Không clip điểm và không chọn model theo test.
- Bảng per-dataset Figure18/21/23/24 của tác giả là accuracy thô; adapter chuẩn hóa
  bằng cùng anchors rồi lấy trung bình. Kết quả được đối chiếu với Figure2/4b/5
  ở sai số `1e-12`, kể cả cờ **rotation ON** của phụ lục và **deactivated** của hình chính.
- Benchmark dùng các đường đã được kiểm chứng trong repository; không dùng lại
  phiên bản sampler đơn giản hóa của demo tham khảo.
- `%` trong danh mục Figure chỉ có nghĩa **đã phân tích lại CSV**, không là tự train.
  Phần tự chạy có độ phủ riêng. Case đại diện không bị gán % tùy ý.
- Figure35–38 là **40K estimators/patience**, không phải 40K mẫu train.

`data/manifest.json` khóa hash của 32 nguồn nhỏ đang sử dụng. Nếu file thiếu hoặc bị
thay, app báo lỗi và không hiển thị số liệu đó. Mốc chuẩn hóa và công thức được kiểm
tra với `src.analysis.normalize`. Chỉ maintainer dùng `prepare_data.py` khi cố ý cập nhật
nguồn đã audit; app không gọi script này. Nguồn kết quả và mọi công thức training gốc
được giữ nguyên.
Hash CSV/JSON chỉ chuẩn hóa xuống dòng CRLF thành LF để dùng được sau khi clone
trên Windows/Linux; không sửa nội dung file. Numeric NPZ được hash nguyên byte.

## Kiểm tra

```powershell
python -m unittest discover -s demo/tests -v
```

Kiểm tra bao gồm parity main/per-dataset figures, normalized/raw scores, độ phủ/repeats,
numeric samples, không training, ngân sách HPO thật, trạng thái bộ lọc và các luồng app.
Test dùng `streamlit.testing.v1.AppTest`; không train và không gửi request Kaggle.
Kết quả kiểm chứng và giới hạn được lưu trong [VERIFICATION.md](VERIFICATION.md).

## Cây thư mục

```text
demo/
  app.py            hai trang và state
  evidence.py       adapter dữ liệu, hash, normalization
  charts.py         biểu đồ Plotly
  inventory.py      danh mục paper và độ phủ
  data/             anchors, provenance, manifest, execution state
  tests/            parity và AppTest
  prepare_data.py   công cụ maintainer, không được app gọi
  run.ps1           chỉ bind localhost, không telemetry
```

Ứng dụng mới được viết để đọc bằng chứng của nhóm; không sao chép source của
repo tham khảo không có LICENSE. Repo tham khảo được ghi nhận cho cách tổ chức demo
và bảng màu. Code khoa học/tài liệu tác giả giữ nguồn riêng trong bộ `reproduce/`.
