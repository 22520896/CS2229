# Paper và thực nghiệm tái hiện

**Mở trước: [PDF chính](report/bao-cao-tai-hien.pdf).** Bản này tổng hợp paper,
kết quả tự train V2 + mở rộng, ba cơ chế, phần phân tích kết quả tác giả và các pilot.
Snapshot dùng dữ liệu đã audit lúc **02:00 ngày 28/11/2026**. Các queue còn chạy
được giữ nguyên và sẽ bổ sung sau; **chưa full paper**.

| Bằng chứng | Đã thực hiện | Ý nghĩa |
|---|---|---|
| Noise features | 168/168 ô; 420/420 fit | Một cấu hình công bố/điều kiện, đủ original auto seeds |
| Rotation | 224/224 ô; 560/560 fit | Đủ có/không xoay, trước/sau loại 50% biến |
| Target smoothing | 240/240 ô; 576/576 fit | Bốn mức hệ số, giữ nhãn validation/test gốc |
| HPO tự train | Numeric tree 39.058/39.600; categorical tree 28.508/28.800 | Chỉ đếm cấu hình đủ repeats; xem bảng đầy đủ trong PDF |
| Phân tích kết quả tác giả | 27/40 figure IDs | Tái chạy phân tích và vẽ từ CSV, không ghi là tự train |

100% ba grid trên **không** tương đương replay toàn bộ HPO, shuffles hay full paper.
Không có một % hoàn thành chung. Cấu hình luôn chọn bằng validation; không chọn
theo điểm test. Legacy không trộn vào kết quả chính.

## Cây thư mục

```text
reproduce/
├── report/       PDF chính duy nhất
├── notebooks/    3 notebook tự train, một cơ chế/figure mỗi notebook, đã có output
├── results/      CSV độ phủ, 899 nhóm ghi nhận, bảng điểm và 3 numeric samples
├── configs/      Tham số công bố, protocol, source record và hash nguồn frozen
├── figures/      PNG để đọc và SVG để dùng lại trong báo cáo/slide
├── scripts/      Check, vẽ, dựng PDF và lệnh train mẫu thủ công
├── training/     Code frozen và hướng dẫn môi trường training gốc
└── tests/        Kiểm tra dữ liệu thật, reject metric/hash bị sửa
```

27 notebook phân tích tác giả và hai notebook phụ lục được giữ tại
[notebooks/ gốc](../notebooks/), không copy lại vào đây.
[Mục lục và lệnh phân tích tác giả](../README.md#notebook) ·
[Đối chiếu Python–R](../evidence/reference_method.md) ·
[Phạm vi phụ lục](../evidence/appendix_sources.md).

## Chạy offline phần trình bày

Python 3.11 trở lên; CPU đủ, không tải model và không cần GPU.
Chạy từ thư mục gốc repository:

```powershell
python -m pip install -r reproduce/requirements.txt
python reproduce/scripts/check_results.py
python -m unittest discover -s reproduce/tests -v
python reproduce/scripts/plots.py
python reproduce/scripts/build_report.py
```

Mở ba notebook trong VS Code/Jupyter rồi **Run All**. Chúng chỉ đọc output đã lưu,
tính trung bình trình bày và vẽ; **không gọi fit**, không chạy lại HPO và không gửi job.
Mọi hình trong ba notebook được sinh từ dữ liệu thật, không từ tọa độ số hóa PDF.

| Paper | Notebook tự train | Phạm vi |
|---|---|---|
| Figure 2/18 | [Target smoothing](notebooks/figure_02_target_smoothing_self_trained.ipynb) | 15 dataset, bốn model, bốn mức |
| Figure 4b/21 | [Noise features](notebooks/figure_04_noise_features_self_trained.ipynb) | 14 dataset, bốn model, ba mức |
| Figure 5/23–24 | [Rotation](notebooks/figure_05_rotation_self_trained.ipynb) | 14 dataset, bốn model, bốn điều kiện |

Muốn **thực sự train lại** một fit RF đại diện bằng code gốc, xem
[training/README.md](training/README.md). Môi trường training Python 3.8 và môi trường
phân tích Python 3.11+ được tách rõ. Lệnh train là tùy chọn thủ công, chưa được chạy
trong lượt đóng gói này; output lịch sử vẫn được giữ nguyên.

## Đọc dữ liệu và kiểm chứng

- `results/mechanism_cells.csv`: 899 nhóm ghi nhận trong union, gồm 632 ô grid tham chiếu
  và các nhóm removal/smoothing bổ sung. Chỉ dùng dòng `reference_grid=True` cho ba hình
  mới; một ô có mean khi đủ seed.
- `results/coverage.csv`: numerator, denominator và đơn vị riêng từng hàng. Mẫu số tự
  chạy dùng catalog 57 thiết lập; nhãn Figure 1 nguồn là 59 thiết lập. Hai phạm vi không
  được âm thầm đồng nhất. Chi tiết ở `configs/protocol.json`.
- `results/balanced_scores.csv`: 24/57 thiết lập với ngân sách chung trong từng cột,
  default-first, 15 hoán vị. Bảng chốt 25/11; fit mới không tự đổi bảng này.
- `results/sample_fits.json` và `results/numeric/`: ba fit RF đại diện trên Electricity,
  có nhãn, dự đoán và rowID; không có raw features hay weights.
- `results/source_audits.json`: hash của 89 nguồn cơ chế đã audit. Bộ gọn chỉ tái kiểm
  numeric của **ba mẫu kèm theo**, không tự nhận đã audit lại toàn bộ lịch sử trong bundle.

Kiểm tra giữ tolerance `rtol=atol=1e-12`, kiểm tra confusion matrix, split ID không trùng,
checksum và đủ repeats. 18 file lõi trong `training/source/` giữ nguyên byte so với bản
nguồn đã pin. `manifest.json` ràng buộc các file và nguồn phụ thuộc trong repository.

Tại cutoff, đợt native gap có 11 success, 5 OOM, 9 timeout và 1.464 fit chờ/đang chạy.
Giữ mọi lỗi và chưa tính queue thành success. Không có benchmark mới được gửi hay
cancel trong lượt tổng hợp.

## Nguồn

[Paper NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html) ·
[Repo tác giả tại commit đã pin](https://github.com/LeoGrin/tabular-benchmark/tree/9d54cf53d9fd3159e367e70a00005f4fcbf2c79d) ·
[Kho nghiên cứu đầy đủ, riêng tư](https://github.com/NgVuxKhaiTam/CS2229.26.1.CH.02-Project).
Các phần sinh từ nguồn nào được ghi riêng trong `configs/source_versions.json`.

Demo mới chưa được triển khai trong đợt đóng gói; sẽ thảo luận riêng.
