import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


DATA_FILE = "S관(3공장) 계획대시보드.csv"
FX_FILE = "환율기준.csv"
SPEC_NUMBER_RE = re.compile(r"[+-]\d+\.\d{2}")
CYL_AXIS_RE = re.compile(r"([+-]\d+\.\d{2})(\d{3})")
DEBUG_DUPLICATES = False
PACK_COL_CANDIDATES = ["포장단위", "포장단위명", "포장규격", "단위", "UOM", "uom", "PACK_UOM", "포장 UOM"]


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


def find_first_column(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def debug_duplicate_orders(view_df, label):
    base_cols = [col for col in ["이니셜", "품목코드", "수주번호"] if col in view_df.columns]
    pack_col = find_first_column(view_df.columns, PACK_COL_CANDIDATES)
    key_cols = base_cols + [col for col in [pack_col] if col]
    if not key_cols:
        print(f"[DEBUG] {label}: no key columns available.")
        return
    key_df = view_df[key_cols].astype(str).fillna("")
    counts = key_df.value_counts()
    dupes = counts[counts > 1]
    if dupes.empty:
        print(f"[DEBUG] {label}: no duplicate keys found.")
        return
    print(f"[DEBUG] {label}: duplicate keys {len(dupes)}")
    for key, count in dupes.head(20).items():
        print(f"[DEBUG] {key} -> {count}")


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


def join_unique(series):
    values = sorted(
        {str(value).strip() for value in series if pd.notna(value) and str(value).strip()}
    )
    return ";".join(values) if values else ""


def normalize_name_tokens(text):
    cleaned = re.sub(r"\([^)]*\)", "", str(text))
    cleaned = cleaned.replace(";", " ")
    tokens = []
    for token in cleaned.split():
        token = token.strip(" ,;")
        if not token:
            continue
        if "_" in token:
            parts = token.split("_")
            if any(char.isdigit() for char in parts[-1]):
                token = parts[0]
        tokens.append(token)
    return tokens


def representative_name(series):
    names = [str(value).strip() for value in series if pd.notna(value) and str(value).strip()]
    unique = list(dict.fromkeys(names))
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    token_lists = [normalize_name_tokens(name) for name in unique]
    base_tokens = token_lists[0]
    if not base_tokens:
        return unique[0]
    token_sets = [set(tokens) for tokens in token_lists[1:]]
    common = [token for token in base_tokens if all(token in s for s in token_sets)]
    if len(common) >= 2:
        return " ".join(common).strip()
    return unique[0]


def fill_object_na(frame):
    df = frame.copy()
    object_cols = df.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        df[object_cols] = df[object_cols].fillna("")
    return df


def calc_table_height(
    row_count, row_height=32, header_height=40, min_height=140, max_height=650
):
    if row_count <= 0:
        return min_height
    height = header_height + row_height * (row_count + 1)
    return max(min_height, min(max_height, int(height)))


def render_selection_sum(df, selected_rows, label="선택 합계"):
    if not selected_rows:
        return
    numeric_cols = df.select_dtypes(include="number").columns
    if numeric_cols.empty:
        return
    sums = df.iloc[selected_rows][numeric_cols].sum()
    sum_df = sums.to_frame().T
    sum_df.insert(0, "구분", label)
    format_dict = {col: "{:,.0f}" for col in numeric_cols}
    st.caption(label)
    st.dataframe(
        sum_df.style.format(format_dict, na_rep=""),
        width="stretch",
        height=calc_table_height(1, min_height=120, max_height=180),
    )


def add_due_warning(frame, date_col="출고예상일", days=7):
    if date_col not in frame.columns:
        return frame, None, None
    df = frame.copy()
    dates = pd.to_datetime(df[date_col], errors="coerce")
    today = pd.Timestamp.today().normalize()
    overdue = dates < today
    soon = dates.between(today, today + pd.Timedelta(days=days))
    date_str = dates.dt.strftime("%Y-%m-%d").fillna("")
    date_str = date_str.where(~(overdue | soon), "❗ " + date_str)
    df[date_col] = date_str
    return df, overdue, soon


def aggregate_production(view):
    if "품목코드" not in view.columns:
        return view

    pack_col = find_first_column(view.columns, PACK_COL_CANDIDATES)
    key_cols = [col for col in ["이니셜", "품목코드", "수주번호"] if col in view.columns]
    if pack_col:
        key_cols.append(pack_col)
    if key_cols:
        view = view.drop_duplicates(subset=key_cols)

    stock_cols = ["사출창고", "분리창고", "검사접착", "누수규격", "완제품"]
    sum_cols = [
        "수량",
        "잔여수주량",
        "생산필요량",
        "사출필요량",
        "분리필요량",
        "수화필요량",
        "접착필요량",
        "누수/규격필요량",
    ]
    for col in sum_cols + stock_cols:
        if col in view.columns:
            view[col] = pd.to_numeric(view[col], errors="coerce").fillna(0)

    stock_cols_available = [col for col in stock_cols if col in view.columns]
    stock_data = view.drop_duplicates(subset=["품목코드"])[
        ["품목코드"] + stock_cols_available
    ]

    agg = {}
    for col in sum_cols:
        if col in view.columns:
            agg[col] = "sum"
    if "출고예상일" in view.columns:
        agg["출고예상일"] = "min"
    for col in ["품명", "신규분류코드", "Q코드", "R코드"]:
        if col in view.columns:
            agg[col] = representative_name if col == "품명" else join_unique

    view_without_stock = view.drop(columns=stock_cols_available, errors="ignore")
    grouped = view_without_stock.groupby("품목코드", dropna=False).agg(agg).reset_index()
    if stock_cols_available:
        grouped = grouped.merge(stock_data, on="품목코드", how="left")
    return grouped


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
        st.caption(f"업데이트 일자: {updated_at:%Y-%m-%d}")
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

    rename_map = {
        "잔여수주수량": "잔여수주량",
        "누수규격검사": "누수규격",
        "제품재고": "완제품",
        "하이드레이션/전면검사필요량": "수화필요량",
        "접착/멸균필요량": "접착필요량",
        "누수/규격검사필요량": "누수/규격필요량",
    }
    df = df.rename(columns=rename_map)

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
        search_keyword = st.text_input(
            "전체 검색 (품명/이니셜/품목코드)",
            value="",
            placeholder="검색어 입력",
        )
        date_range = None
        if "출고예상일" in df.columns:
            date_min = df["출고예상일"].dropna().min()
            date_max = df["출고예상일"].dropna().max()
            if pd.notna(date_min) and pd.notna(date_max):
                date_range = st.date_input(
                    "출고예상일 범위",
                    value=(date_min.date(), date_max.date()),
                )
        ship_scope = None
        if "수출국가" in df.columns:
            ship_scope = st.selectbox("수출/국내", ["전체", "수출", "국내"], index=0)
        view_mode = st.radio("보기", ["생산필요만", "전체"], horizontal=True)
        sort_by_ship = True

    filtered = df.copy()
    if selected_categories:
        filtered = filtered[filtered["신규분류코드"].isin(selected_categories)]
    else:
        filtered = filtered.iloc[0:0]
    if ship_scope and ship_scope != "전체" and "수출국가" in filtered.columns:
        scope_series = filtered["수출국가"].astype(str).str.strip()
        scope_series = scope_series.apply(lambda value: "국내" if value == "국내" else "수출")
        filtered = filtered[scope_series == ship_scope]
    if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        filtered = filtered[
            (filtered["출고예상일"].notna())
            & (filtered["출고예상일"] >= start_date)
            & (filtered["출고예상일"] <= end_date)
        ]
    if search_keyword.strip():
        search_cols = [
            col
            for col in ["품명", "이니셜", "품목코드", "판매코드", "수주번호"]
            if col in filtered.columns
        ]
        if search_cols:
            mask = (
                filtered[search_cols]
                .astype(str)
                .apply(
                    lambda row: row.str.contains(search_keyword, case=False, na=False).any(),
                    axis=1,
                )
            )
            filtered = filtered[mask]
    filtered_base = filtered.copy()

    if view_mode == "생산필요만":
        filtered = filtered[filtered["생산필요량"] > 0]

    if sort_by_ship and "출고예상일" in filtered.columns:
        filtered = filtered.sort_values(by="출고예상일", ascending=True)

    if DEBUG_DUPLICATES:
        debug_duplicate_orders(filtered_base, "filtered_base")

    tab_all, tab_production, tab_c_support = st.tabs(
        ["S관 종합현황", "생산계획", "C관접착지원"]
    )

    with tab_all:
        summary_df = filtered_base

        def add_fx_amounts(frame):
            result = frame.copy()
            result["수주금액"] = pd.to_numeric(
                result.get("수주금액", 0).astype(str).str.replace(",", ""),
                errors="coerce",
            ).fillna(0.0)
            result["원환산"] = 0.0
            result["달러환산"] = 0.0
            if "화폐" in result.columns and fx_map:
                for currency, fx in fx_map.items():
                    mask = result["화폐"].astype(str).str.strip() == currency
                    if not mask.any():
                        continue
                    result.loc[mask, "원환산"] = result.loc[mask, "수주금액"] * float(
                        fx.get("원", 0)
                    )
                    result.loc[mask, "달러환산"] = result.loc[mask, "수주금액"] * float(
                        fx.get("$", 0)
                    )
                krw_mask = result["화폐"].astype(str).str.strip() == "KRW"
                if krw_mask.any():
                    result.loc[krw_mask, "원환산"] = result.loc[krw_mask, "수주금액"]
                    result.loc[krw_mask, "달러환산"] = result.loc[krw_mask, "수주금액"] / 1300
            return result

        month_source = "출고예상일" if "출고예상일" in summary_df.columns else None
        if month_source:
            month_df = summary_df.copy()
            month_df[month_source] = pd.to_datetime(
                month_df[month_source], errors="coerce"
            )
            month_df = month_df[month_df[month_source].notna()]
            month_df["월"] = month_df[month_source].dt.to_period("M").astype(str)
            if "수출국가" in month_df.columns:
                month_df["수출/국내"] = month_df["수출국가"].astype(str).str.strip()
                month_df["수출/국내"] = month_df["수출/국내"].apply(
                    lambda value: "국내" if value == "국내" else "수출"
                )
            else:
                month_df["수출/국내"] = "수출"
            month_df["수량"] = pd.to_numeric(
                month_df.get("수량", 0), errors="coerce"
            ).fillna(0)
            month_df["생산필요량"] = pd.to_numeric(
                month_df.get("생산필요량", 0), errors="coerce"
            ).fillna(0)

            fx_month = add_fx_amounts(month_df)

            total_counts = (
                month_df.groupby(["월", "수출/국내"])
                .agg(
                    이니셜수=("이니셜", "nunique"),
                    품목수=("품명", "nunique" if "품명" in month_df.columns else "count"),
                    규격수=("판매코드", "nunique"),
                )
                .reset_index()
            )
            remaining_df = month_df[month_df["생산필요량"] > 0]
            remain_counts = (
                remaining_df.groupby(["월", "수출/국내"])
                .agg(
                    잔여이니셜수=("이니셜", "nunique"),
                    잔여품목수=("품명", "nunique" if "품명" in remaining_df.columns else "count"),
                    잔여규격수=("판매코드", "nunique"),
                )
                .reset_index()
            )

            monthly = (
                month_df.groupby(["월", "수출/국내"], dropna=False)[["수량", "생산필요량"]]
                .sum()
                .reset_index()
            )
            if "이니셜" in fx_month.columns:
                amounts = (
                    fx_month.groupby(
                        ["월", "수출/국내", "이니셜"], dropna=False
                    )[["원환산", "달러환산"]]
                    .max()
                    .reset_index()
                    .groupby(["월", "수출/국내"], dropna=False)[["원환산", "달러환산"]]
                    .sum()
                    .reset_index()
                )
            else:
                amounts = (
                    fx_month.groupby(["월", "수출/국내"], dropna=False)[
                        ["원환산", "달러환산"]
                    ]
                    .sum()
                    .reset_index()
                )
            monthly = monthly.merge(amounts, on=["월", "수출/국내"], how="left")
            monthly = monthly.merge(total_counts, on=["월", "수출/국내"], how="left")
            monthly = monthly.merge(remain_counts, on=["월", "수출/국내"], how="left")

            monthly["진도율"] = (
                (1 - (monthly["생산필요량"] / monthly["수량"])) * 100
            ).fillna(0)

            def fmt_int(value):
                try:
                    return f"{int(round(float(value))):,}"
                except (TypeError, ValueError):
                    return "0"

            def fmt_ratio(value):
                try:
                    return f"{float(value):.2f}"
                except (TypeError, ValueError):
                    return "0.00"

            monthly["이니셜수"] = (
                monthly["이니셜수"].fillna(0).apply(fmt_int)
                + "("
                + monthly["잔여이니셜수"].fillna(0).apply(fmt_int)
                + ")"
            )
            monthly["품목수"] = (
                monthly["품목수"].fillna(0).apply(fmt_int)
                + "("
                + monthly["잔여품목수"].fillna(0).apply(fmt_int)
                + ")"
            )
            monthly["규격수"] = (
                monthly["규격수"].fillna(0).apply(fmt_int)
                + "("
                + monthly["잔여규격수"].fillna(0).apply(fmt_int)
                + ")"
            )

            monthly = monthly.rename(
                columns={
                    "수량": "월 수주수량",
                    "원환산": "수주금액(원)",
                    "달러환산": "수주금액($)",
                }
            )
            monthly = monthly.sort_values(["월", "수출/국내"])
            total_qty = monthly["월 수주수량"].sum()
            total_need = monthly["생산필요량"].sum()
            total_ratio = (1 - (total_need / total_qty)) * 100 if total_qty else 0
            total_krw = monthly["수주금액(원)"].sum()
            total_usd = monthly["수주금액($)"].sum()

            monthly["월 수주수량"] = monthly["월 수주수량"].apply(fmt_int)
            monthly["생산필요량"] = monthly["생산필요량"].apply(fmt_int)
            monthly["진도율"] = monthly["진도율"].apply(lambda v: f"{fmt_ratio(v)}%")
            monthly["수주금액(원)"] = monthly["수주금액(원)"].apply(fmt_int)
            monthly["수주금액($)"] = monthly["수주금액($)"].apply(fmt_int)
            monthly = monthly.fillna("")
            monthly = monthly[monthly["월 수주수량"] != "0"]
            monthly = monthly[
                [
                    "수출/국내",
                    "월",
                    "월 수주수량",
                    "생산필요량",
                    "진도율",
                    "이니셜수",
                    "품목수",
                    "규격수",
                    "수주금액(원)",
                    "수주금액($)",
                ]
            ]
            st.subheader("S관 수주접수 현황")
            kpi = st.columns(5)
            kpi[0].metric("월 수주수량 합계", fmt_int(total_qty))
            kpi[1].metric("생산필요량 합계", fmt_int(total_need))
            kpi[2].metric("진도율", f"{fmt_ratio(total_ratio)}%")
            kpi[3].metric("수주금액(원)", fmt_int(total_krw))
            kpi[4].metric("수주금액($)", fmt_int(total_usd))
            st.dataframe(
                monthly,
                width="stretch",
                height=calc_table_height(len(monthly), max_height=360, min_height=180),
            )

        st.subheader("상세 목록")
        hide_columns = {
            "잔여수량",
            "사출창고",
            "분리창고",
            "검사접착",
            "누수규격",
            "사출필요량",
            "분리필요량",
            "수화필요량",
            "접착필요량",
            "누수/규격필요량",
        }
        detail_cols = [col for col in filtered.columns if col not in hide_columns]
        search_text = st.text_input("상세 목록 검색", value="", placeholder="검색어 입력")
        detail_df = filtered[detail_cols].copy()
        if "수량" in detail_df.columns and "잔여수량" in detail_df.columns:
            cols = detail_df.columns.tolist()
            cols.remove("수량")
            if "잔여수량" in cols:
                idx = cols.index("잔여수량")
                cols.insert(idx, "수량")
            else:
                cols.append("수량")
            detail_df = detail_df[cols]
        if "출고예상일" in detail_df.columns:
            sort_dates = pd.to_datetime(detail_df["출고예상일"], errors="coerce")
            detail_df = (
                detail_df.assign(_sort_date=sort_dates)
                .sort_values("_sort_date", ascending=True)
                .drop(columns=["_sort_date"])
            )
        overdue_mask = None
        soon_mask = None
        if "출고예상일" in detail_df.columns:
            detail_df, overdue_mask, soon_mask = add_due_warning(detail_df, "출고예상일", 7)
        date_keywords = ("일자", "날짜", "출고", "납기")
        date_cols = [
            col
            for col in detail_df.columns
            if col != "출고예상일"
            and (
                any(key in col for key in date_keywords)
                or col.lower().endswith("date")
            )
        ]
        for col in date_cols:
            detail_df[col] = format_date_series(detail_df[col])
        numeric_cols = [
            col
            for col in detail_df.columns
            if any(key in col for key in ("수량", "단가", "금액", "필요량", "재고"))
        ]
        for col in numeric_cols:
            detail_df[col] = pd.to_numeric(
                detail_df[col].astype(str).str.replace(",", ""), errors="coerce"
            )
        detail_df = fill_object_na(detail_df)
        if search_text.strip():
            mask = detail_df.astype(str).apply(
                lambda row: row.str.contains(search_text, case=False, na=False).any(),
                axis=1,
            )
            detail_df = detail_df[mask]
        if overdue_mask is not None:
            overdue_mask = overdue_mask.reindex(detail_df.index, fill_value=False)
        if soon_mask is not None:
            soon_mask = soon_mask.reindex(detail_df.index, fill_value=False)

        def highlight_warning(data):
            styles = pd.DataFrame("", index=data.index, columns=data.columns)
            if "출고예상일" not in styles.columns:
                return styles
            for idx in data.index:
                if overdue_mask is not None and overdue_mask.loc[idx]:
                    styles.loc[idx, "출고예상일"] = "color: #D0021B; font-weight: 700;"
                elif soon_mask is not None and soon_mask.loc[idx]:
                    styles.loc[idx, "출고예상일"] = "color: #B7791F; font-weight: 700;"
            return styles

        format_dict = {}
        for col in detail_df.columns:
            if not pd.api.types.is_numeric_dtype(detail_df[col]):
                continue
            if "단가" in col or "율" in col or "비율" in col:
                format_dict[col] = "{:,.2f}"
            else:
                format_dict[col] = "{:,.0f}"
        styled = (
            detail_df.style.format(format_dict, na_rep="")
            .apply(highlight_warning, axis=None)
        )
        selection = st.dataframe(
            styled,
            width="stretch",
            height=calc_table_height(len(detail_df)),
            on_select="rerun",
            selection_mode="multi-row",
            key="detail_table",
        )
        if selection is not None:
            render_selection_sum(detail_df, selection.selection.rows, "선택 합계")

    def render_production_table(view_df, category_name, key_suffix, show_warning=False):
        st.markdown(f"**{category_name} 생산계획**")
        if view_df.empty:
            return

        aggregated = aggregate_production(view_df.copy())
        add_spec_columns(aggregated)
        if "출고예상일" in aggregated.columns:
            aggregated["출고예상일"] = pd.to_datetime(
                aggregated["출고예상일"], errors="coerce"
            )
            aggregated["_sort_date"] = aggregated["출고예상일"]
        if "생산필요량" in aggregated.columns:
            aggregated = aggregated[aggregated["생산필요량"] > 0]
        if aggregated.empty:
            return
        overdue_mask = None
        soon_mask = None
        if show_warning:
            aggregated, overdue_mask, soon_mask = add_due_warning(
                aggregated, "출고예상일", 7
            )

        show_codes = st.checkbox(
            "코드 컬럼 펼치기 (품목코드, Q코드, R코드)",
            value=False,
            key=f"show_codes_prod_{key_suffix}",
        )

        stock_cols = ["사출창고", "분리창고", "검사접착", "누수규격", "완제품"]
        need_cols = ["사출필요량", "분리필요량", "수화필요량", "접착필요량", "누수/규격필요량"]
        columns = ["품명", "신규분류코드"]
        if show_codes:
            columns.extend(["품목코드", "Q코드", "R코드"])
        columns.extend(
            ["파워", "실린더(ADD)", "축", "출고예상일", "수량", "잔여수주량", "생산필요량"]
        )
        columns.extend([col for col in stock_cols if col in aggregated.columns])
        columns.extend([col for col in need_cols if col in aggregated.columns])
        available = [col for col in columns if col in aggregated.columns]
        extra_cols = ["_sort_date"] if "_sort_date" in aggregated.columns else []
        view = aggregated[available + extra_cols].copy()

        if sort_by_ship and "_sort_date" in view.columns:
            view = view.sort_values(by="_sort_date", ascending=True)
        if "_sort_date" in view.columns:
            view = view.drop(columns=["_sort_date"])
        if "출고예상일" in view.columns:
            if show_warning:
                view["출고예상일"] = view["출고예상일"].fillna("")
            else:
                view["출고예상일"] = format_date_series(view["출고예상일"])

        search_text = st.text_input(
            "검색",
            value="",
            placeholder="검색어 입력",
            key=f"search_prod_{key_suffix}",
        )
        if search_text.strip():
            mask = view.astype(str).apply(
                lambda row: row.str.contains(search_text, case=False, na=False).any(),
                axis=1,
            )
            view = view[mask]

        view = fill_object_na(view)
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

            stock_first = ("재고현황", stock_cols[0]) if stock_cols else None
            stock_last = ("재고현황", stock_cols[-1]) if stock_cols else None
            need_first = ("생산필요량", need_cols[0]) if need_cols else None
            need_last = ("생산필요량", need_cols[-1]) if need_cols else None
            if stock_first and stock_first in styles.columns:
                styles.loc[:, stock_first] = append_style(
                    styles.loc[:, stock_first], "border-left: 2px solid #777"
                )
            if stock_last and stock_last in styles.columns:
                styles.loc[:, stock_last] = append_style(
                    styles.loc[:, stock_last], "border-right: 2px solid #777"
                )
            if need_first and need_first in styles.columns:
                styles.loc[:, need_first] = append_style(
                    styles.loc[:, need_first], "border-left: 2px solid #777"
                )
            if need_last and need_last in styles.columns:
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

        if overdue_mask is not None:
            overdue_mask = overdue_mask.reindex(aggregated.index, fill_value=False)
            overdue_mask = overdue_mask.loc[view.index]
        if soon_mask is not None:
            soon_mask = soon_mask.reindex(aggregated.index, fill_value=False)
            soon_mask = soon_mask.loc[view.index]

        def highlight_warning(data):
            styles = pd.DataFrame("", index=data.index, columns=data.columns)
            if overdue_mask is None and soon_mask is None:
                return styles
            due_key = ("", "출고예상일") if ("", "출고예상일") in styles.columns else None
            if due_key is None:
                return styles
            for idx in data.index:
                if overdue_mask is not None and overdue_mask.loc[idx]:
                    styles.loc[idx, due_key] = "color: #D0021B; font-weight: 700;"
                elif soon_mask is not None and soon_mask.loc[idx]:
                    styles.loc[idx, due_key] = "color: #B7791F; font-weight: 700;"
            return styles

        format_dict = {}
        for col in view.columns:
            series = view[col]
            if not pd.api.types.is_numeric_dtype(series):
                continue
            name = col[1] if isinstance(col, tuple) else col
            if "단가" in name or "율" in name or "비율" in name:
                format_dict[col] = "{:,.2f}"
            else:
                format_dict[col] = "{:,.0f}"
        styled = (
            view.style.format(format_dict, na_rep="")
            .apply(border_styles, axis=None)
            .apply(highlight_warning, axis=None)
            .set_table_styles(header_styles)
        )
        selection = st.dataframe(
            styled,
            width="stretch",
            height=calc_table_height(len(view)),
            on_select="rerun",
            selection_mode="multi-row",
            key=f"prod_table_{key_suffix}",
        )
        if selection is not None:
            render_selection_sum(view, selection.selection.rows, "선택 합계")

    def render_injection_summary(view_df, title):
        st.markdown(f"**{title}**")
        if "R코드" not in view_df.columns:
            st.info("R코드 컬럼이 없습니다.")
            return
        summary = view_df.copy()
        summary = summary[summary["R코드"].astype(str).str.strip() != ""]
        if "품명" not in summary.columns:
            summary["품명"] = ""
        if "사출창고" not in summary.columns:
            summary["사출창고"] = 0
        for col in ("생산필요량", "사출필요량", "사출창고"):
            if col in summary.columns:
                summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0)
        if "사출필요량" in summary.columns:
            summary = summary[summary["사출필요량"] > 0]
        if summary.empty:
            st.info("사출필요량이 있는 품목이 없습니다.")
            return

        stock_by_r = summary.drop_duplicates(subset=["R코드"])[["R코드", "사출창고"]]
        stock_by_r["사출창고"] = pd.to_numeric(
            stock_by_r["사출창고"], errors="coerce"
        ).fillna(0)

        summary["출고예상일"] = pd.to_datetime(summary["출고예상일"], errors="coerce")
        grouped = (
            summary.groupby("R코드", dropna=True)
            .agg(
                {
                    "품명": representative_name,
                    "생산필요량": "sum",
                    "사출필요량": "sum",
                    "출고예상일": "min",
                }
            )
            .reset_index()
        )
        grouped = grouped.merge(stock_by_r, on="R코드", how="left")

        specs = grouped["R코드"].apply(parse_spec_from_code).apply(pd.Series)
        specs.columns = ["파워", "실린더(ADD)", "축"]
        grouped = pd.concat([grouped, specs], axis=1)
        grouped["_sort_date"] = pd.to_datetime(grouped["출고예상일"], errors="coerce")
        grouped, overdue_mask, soon_mask = add_due_warning(grouped, "출고예상일", 7)
        ordered_cols = [
            "R코드",
            "품명",
            "파워",
            "실린더(ADD)",
            "축",
            "생산필요량",
            "사출창고",
            "사출필요량",
            "출고예상일",
        ]
        keep_cols = [col for col in ordered_cols if col in grouped.columns]
        if "_sort_date" in grouped.columns:
            keep_cols.append("_sort_date")
        grouped = grouped[keep_cols]
        if "_sort_date" in grouped.columns:
            grouped = grouped.sort_values(by="_sort_date", ascending=True)
            grouped = grouped.drop(columns=["_sort_date"])
        if overdue_mask is not None:
            overdue_mask = overdue_mask.reindex(grouped.index, fill_value=False)
        if soon_mask is not None:
            soon_mask = soon_mask.reindex(grouped.index, fill_value=False)

        def highlight_injection(data):
            styles = pd.DataFrame("", index=data.index, columns=data.columns)
            if "사출창고" in styles.columns:
                styles.loc[:, "사출창고"] = "background-color: #EAF2FF"
            for col in ("생산필요량", "사출필요량"):
                if col in styles.columns:
                    styles.loc[:, col] = "background-color: #FFF1E6"
            if "출고예상일" in styles.columns:
                for idx in data.index:
                    if overdue_mask is not None and overdue_mask.loc[idx]:
                        styles.loc[idx, "출고예상일"] = "color: #D0021B; font-weight: 700;"
                    elif soon_mask is not None and soon_mask.loc[idx]:
                        styles.loc[idx, "출고예상일"] = "color: #B7791F; font-weight: 700;"
            return styles

        format_dict = {}
        for col in grouped.columns:
            if not pd.api.types.is_numeric_dtype(grouped[col]):
                continue
            if "단가" in col or "율" in col or "비율" in col:
                format_dict[col] = "{:,.2f}"
            else:
                format_dict[col] = "{:,.0f}"
        styled = grouped.style.format(format_dict, na_rep="").apply(
            highlight_injection, axis=None
        )
        selection = st.dataframe(
            styled,
            width="stretch",
            height=calc_table_height(len(grouped)),
            on_select="rerun",
            selection_mode="multi-row",
            key=f"injection_table_{title}",
        )
        if selection is not None:
            render_selection_sum(grouped, selection.selection.rows, "선택 합계")

    has_category = "신규분류코드" in filtered_base.columns
    color_mask = (
        filtered_base["신규분류코드"].astype(str).str.contains("color", case=False, na=False)
        if has_category
        else pd.Series([False] * len(filtered_base), index=filtered_base.index)
    )

    def render_category_tabs(view_df, label, key_prefix):
        if "생산필요량" in view_df.columns:
            view_df = view_df[view_df["생산필요량"] > 0]
        if view_df.empty:
            return
        st.subheader(f"{label} 생산계획")
        if "신규분류코드" in view_df.columns:
            categories = []
            for value in view_df["신규분류코드"].dropna().unique().tolist():
                if not str(value).strip():
                    continue
                subset = view_df[view_df["신규분류코드"] == value]
                if "생산필요량" in subset.columns:
                    subset = subset[subset["생산필요량"] > 0]
                if not subset.empty:
                    categories.append(value)
            categories = sorted(categories)
        else:
            categories = []
        main_label = f"{label}현황"
        tab_names = [main_label, "사출코드"] + categories if categories else [main_label, "사출코드"]
        tabs = st.tabs(tab_names)
        with tabs[0]:
            render_production_table(
                view_df, main_label, f"{key_prefix}_all", show_warning=True
            )
        with tabs[1]:
            st.subheader("사출공정 (R코드 집계)")
            render_injection_summary(view_df, f"{label} 사출공정 요약")
        for idx, category in enumerate(categories, start=2):
            with tabs[idx]:
                render_production_table(
                    view_df[view_df["신규분류코드"] == category],
                    f"{category}",
                    f"{key_prefix}_{category}",
                    show_warning=True,
                )

    with tab_production:
        color_tab, clear_tab = st.tabs(["Color", "Clear"])
        with color_tab:
            render_category_tabs(filtered_base[color_mask], "Color", "color")
        with clear_tab:
            render_category_tabs(filtered_base[~color_mask], "Clear", "clear")

    with tab_c_support:
        # TEMP: C관접착지원 탭 (임시 운영)
        st.subheader("C관접착지원")
        support_path = Path("C관접착지원_요약.csv")
        if not support_path.exists():
            st.warning("C관접착지원_요약.csv 파일을 찾을 수 없습니다.")
        else:
            try:
                support_df = load_data(support_path)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"C관접착지원 파일을 불러오지 못했습니다: {exc}")
                support_df = None

        if support_df is None:
            st.info("C관접착지원 데이터를 표시할 수 없습니다.")
        else:
            support_df = support_df.rename(columns={col: col.strip() for col in support_df.columns})
            support_df = support_df.dropna(how="all")
            required_cols = {"제품명", "제품코드", "생산부족수량", "납기"}
            missing_cols = required_cols - set(support_df.columns)
            if missing_cols:
                st.warning(
                    f"C관접착지원 필수 컬럼이 없습니다: {', '.join(sorted(missing_cols))}"
                )
                support_df = None

        if support_df is None:
            return

        for col in ["생산부족수량", "생산실적", "진도율"]:
            if col in support_df.columns:
                support_df[col] = pd.to_numeric(
                    support_df[col].astype(str).str.replace(",", "").str.strip(),
                    errors="coerce",
                ).fillna(0)
        if "생산부족수량" in support_df.columns:
            support_df["생산부족수량"] = support_df["생산부족수량"].abs()

        summary_name_count = support_df["제품명"].nunique()
        summary_spec_count = support_df["제품코드"].nunique()
        total_shortage = support_df["생산부족수량"].sum()
        total_actual = support_df["생산실적"].sum() if "생산실적" in support_df.columns else 0
        total_ratio = (
            (total_shortage / total_actual) if total_actual else 0
        )

        kpi = st.columns(5)
        kpi[0].metric("품명", f"{summary_name_count:,}")
        kpi[1].metric("규격수", f"{summary_spec_count:,}")
        kpi[2].metric("요청수량", f"{int(total_shortage):,}")
        kpi[3].metric("생산실적", f"{int(total_actual):,}")
        kpi[4].metric("진도율", f"{total_ratio * 100:.2f}%")

        display_cols = ["제품코드", "제품명", "파워", "생산부족수량", "납기", "생산실적", "진도율"]
        available_cols = [col for col in display_cols if col in support_df.columns]
        view = support_df[available_cols].copy()
        view = fill_object_na(view)
        if "진도율" in view.columns:
            view["진도율"] = view["진도율"].apply(lambda v: f"{v * 100:.2f}%")
        format_dict = {}
        for col in view.columns:
            if pd.api.types.is_numeric_dtype(view[col]) and col != "진도율":
                format_dict[col] = "{:,.0f}"
        styled = view.style.format(format_dict, na_rep="")
        st.dataframe(
            styled,
            width="stretch",
            height=calc_table_height(len(view), max_height=520),
        )


if __name__ == "__main__":
    main()
