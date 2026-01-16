import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


DATA_FILE = "S관(3공장) 계획대시보드.csv"
FX_FILE = "환율기준.csv"
SPEC_NUMBER_RE = re.compile(r"[+-]\d+\.\d{2}")
CYL_AXIS_RE = re.compile(r"([+-]\d+\.\d{2})(\d{3})")


def format_date_series(series):
    dates = pd.to_datetime(series, errors="coerce")
    return dates.dt.strftime("%Y-%m-%d").fillna("")


def parse_spec_from_code(code):
    if not code:
        return "", "", ""
    text = str(code)
    numbers = SPEC_NUMBER_RE.findall(text)
    power = numbers[0] if numbers else ""
    cylinder = numbers[1] if len(numbers) > 1 else ""
    axis = ""
    cyl_match = CYL_AXIS_RE.search(text)
    if cyl_match:
        axis = cyl_match.group(2)
        if not cylinder:
            cylinder = cyl_match.group(1)
    return power, cylinder, axis


def add_spec_columns(view):
    def pick_code(row):
        for col in ("품목코드", "Q코드", "R코드"):
            value = row.get(col, "")
            if pd.notna(value):
                text = str(value).strip()
                if text:
                    return text
        return ""

    specs = view.apply(
        lambda row: parse_spec_from_code(pick_code(row)),
        axis=1,
        result_type="expand",
    )
    view["파워"] = specs[0]
    view["실린더(ADD)"] = specs[1]
    view["축"] = specs[2]


def load_data(path):
    last_error = None
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise last_error


def load_fx(path):
    last_error = None
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise last_error


def main():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:  # noqa: BLE001
        get_script_run_ctx = None

    if get_script_run_ctx and get_script_run_ctx() is None:
        print("이 파일은 Streamlit으로 실행해야 합니다.")
        print("streamlit run streamlit_dashboard.py")
        return

    st.set_page_config(page_title="S관(3공장) 계획 대시보드", layout="wide")
    st.title("S관(3공장) 계획 대시보드")

    data_path = Path(DATA_FILE)
    if not data_path.exists():
        st.error(f"파일을 찾을 수 없습니다: {DATA_FILE}")
        return
    try:
        updated_at = datetime.fromtimestamp(data_path.stat().st_mtime)
        st.caption(f"업데이트 일자: {updated_at:%Y-%m-%d %H:%M}")
    except OSError:
        pass

    try:
        df = load_data(data_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"데이터를 불러오지 못했습니다: {exc}")
        return

    fx_path = Path(FX_FILE)
    fx_map = {}
    if fx_path.exists():
        try:
            fx_df = load_fx(fx_path)
            fx_df.columns = [c.strip() for c in fx_df.columns]
            fx_df["기준"] = fx_df["기준"].astype(str).str.strip()
            if "원" not in fx_df.columns and "\\" in fx_df.columns:
                fx_df = fx_df.rename(columns={"\\": "원"})
            fx_map = fx_df.set_index("기준")[["원", "$"]].to_dict(orient="index")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"환율 파일을 불러오지 못했습니다: {exc}")
    else:
        st.warning(f"환율 파일을 찾을 수 없습니다: {FX_FILE}")

    required_cols = {"신규분류코드", "출고예상일", "생산필요량"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"필수 컬럼이 없습니다: {', '.join(sorted(missing))}")
        return

    df["출고예상일"] = pd.to_datetime(df["출고예상일"], errors="coerce")
    df["생산필요량"] = pd.to_numeric(df["생산필요량"], errors="coerce").fillna(0)
    if "수량" in df.columns:
        df["수량"] = pd.to_numeric(df["수량"], errors="coerce").fillna(0)

    st.markdown(
        """
        <style>
        .block-separator { border-top: 1px solid #D9D9D9; margin: 10px 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    category_values = sorted(
        value for value in df["신규분류코드"].dropna().unique().tolist() if value != ""
    )

    with st.sidebar:
        st.header("필터")
        selected_categories = st.multiselect(
            "신규분류코드",
            options=category_values,
            default=category_values,
        )
        view_mode = st.radio("보기", ["생산필요만", "전체"], horizontal=True)
        sort_by_ship = st.checkbox("출고예상일 빠른순", value=True)

    filtered = df.copy()
    if selected_categories:
        filtered = filtered[filtered["신규분류코드"].isin(selected_categories)]
    else:
        filtered = filtered.iloc[0:0]

    if view_mode == "생산필요만":
        filtered = filtered[filtered["생산필요량"] > 0]

    if sort_by_ship:
        filtered = filtered.sort_values(by="출고예상일", ascending=True)

    tab_all, tab_color, tab_clear = st.tabs(["전체", "Color생산계획", "Clear생산계획"])

    with tab_all:
        st.subheader("S관 수주접수 현황")
        initial_count = (
            filtered["이니셜"].dropna().astype(str).str.strip().nunique()
            if "이니셜" in filtered.columns
            else 0
        )
        item_count = (
            filtered["품명"].dropna().astype(str).str.strip().nunique()
            if "품명" in filtered.columns
            else 0
        )
        spec_count = (
            filtered["판매코드"].dropna().astype(str).str.strip().nunique()
            if "판매코드" in filtered.columns
            else 0
        )
        order_qty_sum = (
            float(filtered["잔여수주수량"].sum())
            if "잔여수주수량" in filtered.columns
            else 0.0
        )
        total_need = float(filtered["생산필요량"].sum())

        amount_krw = 0.0
        amount_usd = 0.0
        if (
            "수주금액" in filtered.columns
            and "화폐" in filtered.columns
            and "이니셜" in filtered.columns
            and fx_map
        ):
            amt_series = pd.to_numeric(
                filtered["수주금액"].astype(str).str.replace(",", ""), errors="coerce"
            ).fillna(0.0)
            temp = pd.DataFrame(
                {
                    "이니셜": filtered["이니셜"].astype(str).str.strip(),
                    "화폐": filtered["화폐"].astype(str).str.strip(),
                    "수주금액": amt_series,
                }
            )
            temp["원환산"] = 0.0
            temp["달러환산"] = 0.0
            for currency, fx in fx_map.items():
                mask = temp["화폐"] == currency
                if not mask.any():
                    continue
                temp.loc[mask, "원환산"] = temp.loc[mask, "수주금액"] * float(
                    fx.get("원", 0)
                )
                temp.loc[mask, "달러환산"] = temp.loc[mask, "수주금액"] * float(
                    fx.get("$", 0)
                )
            grouped = temp.groupby("이니셜", dropna=True)[["원환산", "달러환산"]].max()
            amount_krw = float(grouped["원환산"].sum())
            amount_usd = float(grouped["달러환산"].sum())

        row1 = st.columns(3)
        row1[0].metric("이니셜수", f"{initial_count:,}")
        row1[1].metric("품목수", f"{item_count:,}")
        row1[2].metric("규격수", f"{spec_count:,}")

        st.divider()
        row2 = st.columns(2)
        row2[0].metric("수주금액(원)", f"{amount_krw:,.0f}")
        row2[1].metric("수주금액($)", f"{amount_usd:,.2f}")

        st.divider()
        row3 = st.columns(2)
        row3[0].metric("수주수량합계", f"{order_qty_sum:,.0f}")
        row3[1].metric("생산필요량 합계", f"{total_need:,.0f}")

        st.subheader("상세 목록")
        hide_columns = {
            "잔여수량",
            "사출창고",
            "분리창고",
            "검사접착",
            "누수규격검사",
            "사출필요량",
            "분리필요량",
            "하이드레이션/전면검사필요량",
            "접착/멸균필요량",
            "누수/규격검사필요량",
        }
        detail_cols = [col for col in filtered.columns if col not in hide_columns]
        search_text = st.text_input("상세 목록 검색", value="", placeholder="검색어 입력")
        detail_df = filtered[detail_cols].copy()
        if "출고예상일" in detail_df.columns:
            detail_df["출고예상일"] = format_date_series(detail_df["출고예상일"])
        if search_text.strip():
            mask = detail_df.astype(str).apply(
                lambda row: row.str.contains(search_text, case=False, na=False).any(),
                axis=1,
            )
            detail_df = detail_df[mask]
        st.dataframe(detail_df, width="stretch", height=650)

    def build_focus_table(view_df, show_codes):
        stock_cols = ["사출창고", "분리창고", "검사접착", "누수규격", "완제품"]
        need_cols = ["사출필요량", "분리필요량", "수화필요량", "접착필요량", "누수/규격필요량"]
        view = view_df.copy()
        rename_map = {
            "잔여수주수량": "잔여수주량",
            "누수규격검사": "누수규격",
            "제품재고": "완제품",
            "하이드레이션/전면검사필요량": "수화필요량",
            "접착/멸균필요량": "접착필요량",
            "누수/규격검사필요량": "누수/규격필요량",
        }
        view = view.rename(columns=rename_map)
        add_spec_columns(view)
        if "출고예상일" in view.columns:
            view["출고예상일"] = format_date_series(view["출고예상일"])

        code_columns = ["품목코드", "Q코드", "R코드", "파워", "실린더(ADD)", "축"]
        columns = ["품명", "신규분류코드"]
        if show_codes:
            columns.extend(["품목코드", "Q코드", "R코드"])
        columns.extend(["파워", "실린더(ADD)", "축"])
        columns.extend(
            [
                "출고예상일",
                "잔여수주량",
                "생산필요량",
                "사출창고",
                "분리창고",
                "검사접착",
                "누수규격",
                "완제품",
                "사출필요량",
                "분리필요량",
                "수화필요량",
                "접착필요량",
                "누수/규격필요량",
            ]
        )
        available = [col for col in columns if col in view.columns]
        view = view[available].copy()
        if search_text.strip():
            mask = view.astype(str).apply(
                lambda row: row.str.contains(search_text, case=False, na=False).any(),
                axis=1,
            )
            view = view[mask]
        top_labels = []
        for col in view.columns:
            if col in stock_cols:
                top_labels.append("재고현황")
            elif col in need_cols:
                top_labels.append("생산필요량")
            else:
                top_labels.append("")
        view.columns = pd.MultiIndex.from_arrays([top_labels, view.columns])

        def border_styles(data):
            styles = pd.DataFrame("", index=data.index, columns=data.columns)

            def append_style(series, addition):
                return series.apply(lambda value: f"{value}; {addition}" if value else addition)

            stock_first = ("재고현황", stock_cols[0])
            stock_last = ("재고현황", stock_cols[-1])
            need_first = ("생산필요량", need_cols[0])
            need_last = ("생산필요량", need_cols[-1])
            if stock_first in styles.columns:
                styles.loc[:, stock_first] = append_style(
                    styles.loc[:, stock_first], "border-left: 2px solid #777"
                )
            if stock_last in styles.columns:
                styles.loc[:, stock_last] = append_style(
                    styles.loc[:, stock_last], "border-right: 2px solid #777"
                )
            if need_first in styles.columns:
                styles.loc[:, need_first] = append_style(
                    styles.loc[:, need_first], "border-left: 2px solid #777"
                )
            if need_last in styles.columns:
                styles.loc[:, need_last] = append_style(
                    styles.loc[:, need_last], "border-right: 2px solid #777"
                )
            for col in stock_cols:
                key = ("재고현황", col)
                if key in styles.columns:
                    styles.loc[:, key] = append_style(
                        styles.loc[:, key], "background-color: #EAF2FF"
                    )
            for col in need_cols:
                key = ("생산필요량", col)
                if key in styles.columns:
                    styles.loc[:, key] = append_style(
                        styles.loc[:, key], "background-color: #FFF1E6"
                    )
            return styles

        header_styles = [
            {"selector": "th.col_heading.level0", "props": [("text-align", "center")]},
            {"selector": "th.col_heading.level1", "props": [("text-align", "center")]},
        ]
        sticky_idx = None
        for idx, (group, col) in enumerate(view.columns):
            if group == "재고현황":
                header_styles.append(
                    {
                        "selector": f"th.col_heading.level0.col{idx}",
                        "props": [("background-color", "#AFC8FF"), ("color", "#0B3B8C")],
                    }
                )
            elif group == "생산필요량":
                header_styles.append(
                    {
                        "selector": f"th.col_heading.level0.col{idx}",
                        "props": [("background-color", "#FFBE8C"), ("color", "#8A3B00")],
                    }
                )
            if col in stock_cols:
                header_styles.append(
                    {
                        "selector": f"th.col_heading.level1.col{idx}",
                        "props": [("background-color", "#C7DCFF"), ("color", "#0B3B8C")],
                    }
                )
            elif col in need_cols:
                header_styles.append(
                    {
                        "selector": f"th.col_heading.level1.col{idx}",
                        "props": [("background-color", "#FFD0B0"), ("color", "#8A3B00")],
                    }
                )
            if col == "품명" and sticky_idx is None:
                sticky_idx = idx

        if sticky_idx is not None:
            header_styles.append(
                {
                    "selector": f"th.col_heading.level0.col{sticky_idx}",
                    "props": [
                        ("position", "sticky"),
                        ("left", "0px"),
                        ("z-index", "5"),
                        ("background-color", "#F5F5F5"),
                    ],
                }
            )
            header_styles.append(
                {
                    "selector": f"th.col_heading.level1.col{sticky_idx}",
                    "props": [
                        ("position", "sticky"),
                        ("left", "0px"),
                        ("z-index", "6"),
                        ("background-color", "#F5F5F5"),
                    ],
                }
            )
            header_styles.append(
                {
                    "selector": f"td.col{sticky_idx}",
                    "props": [
                        ("position", "sticky"),
                        ("left", "0px"),
                        ("z-index", "4"),
                        ("background-color", "#FFFFFF"),
                    ],
                }
            )

        styled = view.style.apply(border_styles, axis=None).set_table_styles(header_styles)
        return styled

    def render_focus_table(view_df, title, key_suffix):
        st.markdown(f"**{title}**")
        show_codes = st.checkbox(
            "코드 컬럼 펼치기",
            value=False,
            key=f"show_codes_{key_suffix}",
        )
        table = build_focus_table(view_df, show_codes=show_codes)
        st.dataframe(table, width="stretch", height=650)

    def render_injection_summary(view_df, title):
        st.markdown(f"**{title}**")
        if "R코드" not in view_df.columns:
            st.info("R코드 컬럼이 없습니다.")
            return
        summary = view_df.copy()
        summary = summary[summary["R코드"].astype(str).str.strip() != ""]
        if "품명" not in summary.columns:
            summary["품명"] = ""
        for col in ("생산필요량", "사출필요량", "사출창고"):
            if col not in summary.columns:
                summary[col] = 0
        for col in ("생산필요량", "사출필요량", "사출창고"):
            if col in summary.columns:
                summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0)
        if "사출필요량" in summary.columns:
            summary = summary[summary["사출필요량"] > 0]
        if summary.empty:
            st.info("사출필요량이 있는 품목이 없습니다.")
            return
        summary["출고예상일"] = pd.to_datetime(summary["출고예상일"], errors="coerce")
        grouped = (
            summary.groupby(["R코드", "품명"], dropna=True)
            .agg(
                {
                    "생산필요량": "sum",
                    "사출필요량": "sum",
                    "사출창고": "sum",
                    "출고예상일": "min",
                }
            )
            .reset_index()
        )
        specs = grouped["R코드"].apply(parse_spec_from_code).apply(pd.Series)
        specs.columns = ["파워", "실린더(ADD)", "축"]
        grouped = pd.concat([grouped, specs], axis=1)
        grouped["출고예상일"] = format_date_series(grouped["출고예상일"])
        ordered_cols = [
            col
            for col in ["R코드", "품명", "생산필요량", "사출창고", "사출필요량", "출고예상일"]
            if col in grouped.columns
        ]
        grouped = grouped[
            [*ordered_cols[:2], "파워", "실린더(ADD)", "축", *ordered_cols[2:]]
        ]
        grouped = grouped.sort_values(by="출고예상일", ascending=True)

        def highlight_injection(data):
            styles = pd.DataFrame("", index=data.index, columns=data.columns)
            for col in ("사출창고",):
                if col in styles.columns:
                    styles.loc[:, col] = "background-color: #EAF2FF"
            for col in ("생산필요량", "사출필요량"):
                if col in styles.columns:
                    styles.loc[:, col] = "background-color: #FFF1E6"
            return styles

        styled = grouped.style.apply(highlight_injection, axis=None)
        st.dataframe(styled, width="stretch", height=650)

    def render_focus_tabs(view_df, title):
        st.subheader(title)
        if "신규분류코드" in view_df.columns:
            categories = sorted(
                value
                for value in view_df["신규분류코드"].dropna().unique().tolist()
                if str(value).strip()
            )
        else:
            categories = []
        tab_names = ["전체"] + categories + ["사출공정"]
        tabs = st.tabs(tab_names)
        with tabs[0]:
            render_focus_table(view_df, "전체", f"{title}_all")
        for idx, category in enumerate(categories, start=1):
            with tabs[idx]:
                render_focus_table(
                    view_df[view_df["신규분류코드"] == category],
                    f"{category}",
                    f"{title}_{category}",
                )
        with tabs[-1]:
            render_injection_summary(view_df, f"{title} 사출공정 요약")

    has_category = "신규분류코드" in filtered.columns
    color_mask = (
        filtered["신규분류코드"].astype(str).str.contains("color", case=False, na=False)
        if has_category
        else pd.Series([False] * len(filtered), index=filtered.index)
    )
    with tab_color:
        render_focus_tabs(filtered[color_mask], "Color생산계획")
    with tab_clear:
        render_focus_tabs(filtered[~color_mask], "Clear생산계획")


if __name__ == "__main__":
    main()
