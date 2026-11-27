"""Paper-level scope; representative results are never a 100% denominator."""

import streamlit as st
import pandas as pd

from demo import evidence as E, inventory as I


def render():
    st.header("Tiến độ theo paper")
    st.caption(
        "Snapshot 28/11 · Chưa full paper · Chưa định lượng khi thiếu mẫu số toàn quy trình · Demo local không được cộng"
    )
    mode = st.radio(
        "Danh mục",
        ["Thực nghiệm theo paper", "Hình & phụ lục"],
        horizontal=True,
        label_visibility="collapsed",
        key="progress_mode",
    )
    if mode == "Thực nghiệm theo paper":
        table = I.paper_scope()
        displayed = table[
            ["Paper", "Hạng mục", "Tự train theo paper (%)", "Đã thực hiện"]
        ]
    else:
        table = I.paper_inventory()
        a, b = st.columns([2, 1])
        text = a.text_input(
            "Tìm Figure / Table",
            placeholder="Ví dụ: Figure 19, rotation, SAINT",
            key="inventory_search",
        )
        status = b.selectbox(
            "Phân tích CSV tác giả",
            ["Tất cả", "Đã có", "Chưa có"],
            key="inventory_status",
        )
        if text:
            table = table.loc[
                table.astype(str)
                .apply(
                    lambda column: column.str.contains(text, case=False, regex=False)
                )
                .any(axis=1)
            ]
        missing = table["Phân tích tác giả"].str.startswith(("Chưa", "Thiếu"))
        if status == "Chưa có":
            table = table.loc[missing]
        elif status == "Đã có":
            table = table.loc[~missing]
        displayed = table[
            [
                "ID",
                "Nội dung",
                "Tự train theo paper (%)",
                "Phân tích tác giả",
                "Tự chạy",
                "Bằng chứng",
            ]
        ]
    displayed = displayed.copy()
    displayed["Tự train theo paper (%)"] = displayed["Tự train theo paper (%)"].map(
        lambda value: "Chưa định lượng" if pd.isna(value) else f"{value:.0f}%"
    )
    st.dataframe(
        displayed,
        hide_index=True,
        width="stretch",
        height="auto",
        column_config={
            "Bằng chứng": st.column_config.LinkColumn(display_text="Mở"),
            "Tự train theo paper (%)": st.column_config.TextColumn(width="medium"),
            "Paper": st.column_config.TextColumn(width="small"),
            "Hạng mục": st.column_config.TextColumn(width="medium"),
            "Đã thực hiện": st.column_config.TextColumn(width="large"),
        },
    )
    if mode == "Thực nghiệm theo paper":
        with st.expander("Phạm vi gốc & phần còn thiếu"):
            selected = st.selectbox(
                "Hạng mục paper", table.Paper.tolist(), key="scope_detail"
            )
            row = table.set_index("Paper").loc[selected]
            st.write("**Paper:** " + row["Phạm vi gốc"])
            st.write("**Đã làm:** " + row["Đã thực hiện"])
            st.write("**Còn thiếu:** " + row["Còn thiếu / giới hạn"])
    st.download_button(
        "CSV phạm vi paper",
        table.to_csv(index=False).encode("utf-8-sig"),
        "paper-scope.csv",
        "text/csv",
        key="progress_export",
    )
    with st.expander("Khối lượng đã chạy · không phải % paper"):
        st.dataframe(I.work_volume(), hide_index=True, width="stretch")
        st.caption(
            "Các ô đủ seed của ba cơ chế dùng một cấu hình công bố/điều kiện. Chưa replay toàn bộ HPO/shuffles."
        )
    with st.expander("Phân tích lại kết quả tác giả"):
        st.write(
            "27/40 Figure IDs (67,5%) đã được phân tích lại từ CSV công bố. Đây là độ phủ phân tích hình, không phải % tự train hay % hoàn thành paper."
        )
        st.markdown(
            f"[Notebook và nguồn]({E.link('README.md')}) · [Phạm vi phụ lục]({E.link('evidence/appendix_sources.md')})"
        )
