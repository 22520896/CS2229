# CS2229 · Vì sao mô hình cây vẫn mạnh trên dữ liệu bảng?

Tái lập phân tích của Grinsztajn, Oyallon và Varoquaux tại NeurIPS 2022. **27/40 hình** được tính lại từ kết quả từng lượt tác giả công bố. Phần tự train V2 có phạm vi riêng, với số lượt và độ bao phủ được báo cáo riêng.

- [Báo cáo PDF](report/bao-cao.pdf)
- [Slide HTML](slides/index.html) · Clone hoặc tải ZIP, rồi mở file này trong trình duyệt. Dùng ←/→ để chuyển slide, F để bật hoặc thoát toàn màn hình.
- [Kết quả đối chiếu Python–R](evidence/python_r_parity.json)

## Notebook

| Hình | Notebook | Nội dung |
| --- | --- | --- |
| 1 | [Benchmark](notebooks/figure_01_benchmark.ipynb) | Bốn ô so sánh theo ngân sách HPO và đường cong đối chứng V2 |
| 2 | [Làm trơn mục tiêu](notebooks/figure_02_smoothing.ipynb) | Điểm chuẩn hóa của tác giả và accuracy thô của pilot V2 |
| 3 | [Độ quan trọng đặc trưng](notebooks/figure_03_feature_importance.ipynb) | GBT trên hai nhóm đặc trưng giữ lại và bị loại |
| 4 | [Đặc trưng ít thông tin](notebooks/figure_04_uninformative_features.ipynb) | Hai nhánh loại và thêm đặc trưng, kèm hai cách tính dải Figure 4b |
| 5 | [Phép xoay](notebooks/figure_05_rotation.ipynb) | Hai điều kiện xoay và 381 cặp đối chứng tự train |
| 6 · 7 | [Đặc trưng số](notebooks/figure_06_time_numerical.ipynb) · [Có biến phân loại](notebooks/figure_07_time_categorical.ipynb) | So theo thời gian tìm kiếm |
| 8–11 | [8](notebooks/figure_08_train_size_numerical_classification.ipynb) · [9](notebooks/figure_09_train_size_numerical_regression.ipynb) · [10](notebooks/figure_10_train_size_categorical_classification.ipynb) · [11](notebooks/figure_11_train_size_categorical_regression.ipynb) | Cùng nhóm dữ liệu, train 10K và 50K |
| 12–17 | [12](notebooks/figure_12_numerical_classification.ipynb) · [13](notebooks/figure_13_numerical_regression.ipynb) · [14](notebooks/figure_14_categorical_classification.ipynb) · [15](notebooks/figure_15_categorical_regression.ipynb) · [16](notebooks/figure_16_large_numerical.ipynb) · [17](notebooks/figure_17_large_categorical.ipynb) | Điểm thô theo từng bộ dữ liệu, cả hai quy mô |
| 18 · 20–24 | [18](notebooks/figure_18_smoothing_by_dataset.ipynb) · [20](notebooks/figure_20_feature_importance_by_dataset.ipynb) · [21](notebooks/figure_21_feature_addition_by_dataset.ipynb) · [22](notebooks/figure_22_feature_removal_by_dataset.ipynb) · [23](notebooks/figure_23_rotation_by_dataset.ipynb) · [24](notebooks/figure_24_rotation_pruned_by_dataset.ipynb) | Làm trơn, thêm hoặc bỏ đặc trưng và xoay theo từng dữ liệu |
| 33 · 34 | [Phân loại](notebooks/figure_33_bayesian_classification.ipynb) · [Hồi quy](notebooks/figure_34_bayesian_regression.ipynb) | Random search và tìm kiếm Bayes |
| 39 · 40 | [Đặc trưng số](notebooks/figure_39_tuned_estimators_numeric.ipynb) · [Có biến phân loại](notebooks/figure_40_tuned_estimators_categorical.ipynb) | Tính lại đối chiếu cố định và tune số cây từ hai checkpoint công bố |
| A.1 · A.3 | [Danh mục benchmark và siêu tham số](notebooks/appendix_a1_a3_catalog.ipynb) | 57 thiết lập, 8 bảng và 70 mục siêu tham số |
| A.4 | [Bốn bảng hồi quy](notebooks/appendix_a4_regression.ipynb) | Hệ số Python, đối chiếu R và cột hệ số công bố |

Biểu đồ được lưu sẵn trong notebook. `figures/` chứa hình và các bảng số. PDF dùng các hình chính và một số phụ lục. Slide tập trung vào Figure 1–5.

Figure 19, 25–32 và 35–38 đang chờ đầu vào để chạy lại. [Phụ lục nguồn và trạng thái](evidence/appendix_sources.md)

## Chạy lại

Cần Python 3.11 trở lên. Notebook phân tích chạy bằng CPU. Lần đầu cần Internet để lấy CSV công khai của hình đang chạy. Mã kiểm SHA-256 trên luồng tải và ghi các cột phân tích vào `.cache/`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.author_data ablations
python -m src.author_data benchmark
python -m src.notebooks 3
python -m src.notebooks 4
python -m src.notebooks 2
python -m src.notebooks 5
python -m src.notebooks 1
python -m src.appendix_notebooks 6
python -m src.appendix_notebooks a4
python -m src.appendix_notebooks catalog
python -m src.published_notebooks 39
python -m src.verify --reanalyze
python -m src.verify_appendix --reanalyze
python -m src.verify_published --reanalyze
python -m src.build_report
```

Trên Linux/macOS, kích hoạt bằng `source .venv/bin/activate`. Có thể mở notebook trong Jupyter hoặc VS Code và bấm **Run All**. Kiểm tra offline các kết quả đã lưu bằng `python -m unittest discover -s tests`.

`src/analysis.py` tái lập lọc dữ liệu, chuẩn hóa, xáo thứ tự, chọn theo validation và tổng hợp theo từng hình. Seed 0 cố định, dùng chuỗi uniform MT19937 của R. Figure 4b chính dùng script tác giả với 15 lần xáo và dải phân vị 20%–80%. Bảng riêng dùng 30 lần xáo và min–max theo phần mô tả của paper.

## Nguồn điểm và phạm vi

- **Điểm tác giả** phục vụ các hình tái lập. [Nguồn và cách kiểm chứng Figure 1–5](evidence/reference_method.md) · [Phạm vi phụ lục](evidence/appendix_sources.md)
- **V2 + mở rộng** là đối chứng tự train đã kiểm chứng. Có 25.124 lượt thành công ở V2 chính, 208.742 ở phần mở rộng và 24/57 thiết lập đủ đầu vào để so cùng ngân sách giữa các họ mô hình. [Phạm vi và số tổng hợp](evidence/project_snapshot.json)
- **Quy trình train minh họa** chỉ ra các điểm vào của workflow tác giả. Notebook Figure 2–5 hiển thị cả mẫu cấu hình GBT trích nguyên giá trị từ CSV đã xác minh. Notebook mặc định đọc điểm validation/test đã lưu. [Mẫu cấu hình công bố](evidence/published_config_examples.json)

[Paper NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html) · [Repo tác giả](https://github.com/LeoGrin/tabular-benchmark/tree/9d54cf53d9fd3159e367e70a00005f4fcbf2c79d) · [CSV benchmark](https://ndownloader.figshare.com/files/40081681) · [CSV ablation](https://github.com/LeoGrin/tabular-benchmark/blob/f3fe60384ba7852e3c0b755c20cce478857d9878/analyses/results/random_search_xps.csv)
