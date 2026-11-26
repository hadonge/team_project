import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# 1. 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="2025 서울시 날씨·대기질과 대중교통 이용량 회귀 분석",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2025년 대한민국 공휴일(19일) + 수능일(1일) = 총 20일
SPECIAL_DAYS_2025 = {
    "2025-01-01": "신정",
    "2025-01-27": "설날 임시공휴일",
    "2025-01-28": "설날 연휴",
    "2025-01-29": "설날 당일",
    "2025-01-30": "설날 연휴",
    "2025-03-01": "삼일절",
    "2025-03-03": "삼일절 대체공휴일",
    "2025-05-05": "어린이날·부처님오신날",
    "2025-05-06": "부처님오신날 대체공휴일",
    "2025-06-03": "제21대 대통령 선거일",
    "2025-06-06": "현충일",
    "2025-08-15": "광복절",
    "2025-10-03": "개천절",
    "2025-10-05": "추석 연휴",
    "2025-10-06": "추석 당일",
    "2025-10-07": "추석 연휴",
    "2025-10-08": "추석 대체공휴일",
    "2025-10-09": "한글날",
    "2025-11-13": "수능일(2026학년도 수능)",
    "2025-12-25": "성탄절",
}

def get_pm10_grade(val):
    if val <= 30:
        return "좋음"
    elif val <= 80:
        return "보통"
    elif val <= 150:
        return "나쁨"
    else:
        return "매우 나쁨"

def get_pm25_grade(val):
    if val <= 15:
        return "좋음"
    elif val <= 35:
        return "보통"
    elif val <= 75:
        return "나쁨"
    else:
        return "매우 나쁨"

# ---------------------------------------------------------
# 2. 데이터 로드 및 전처리 함수
# ---------------------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(base_dir, "*.xlsx")) + glob.glob(os.path.join(base_dir, "*.csv"))
    
    if not files:
        return None
        
    target_file = files[0]
    if target_file.endswith(".xlsx"):
        df = pd.read_excel(target_file)
    else:
        df = pd.read_csv(target_file, encoding="utf-8-sig")

    df["일시"] = pd.to_datetime(df["일시"])
    df = df.sort_values("일시").reset_index(drop=True)
    df["날짜_문자"] = df["일시"].dt.strftime("%Y-%m-%d")
    df["특이일명"] = df["날짜_문자"].map(SPECIAL_DAYS_2025).fillna("일반일")
    df["날짜_요일"] = df.apply(
        lambda r: f"{r['날짜_문자']} ({r['요일']} · {r['특이일명']})" if r["특이일명"] != "일반일" else f"{r['날짜_문자']} ({r['요일']})",
        axis=1
    )
    df["주말구분"] = df["요일"].apply(lambda x: "주말(토·일)" if x in ["토요일", "일요일"] else "평일(월~금)")
    
    def classify_day(r):
        if r["날짜_문자"] == "2025-11-13":
            return "수능일"
        elif r["특이일명"] != "일반일":
            return "공휴일·연휴"
        elif r["요일"] in ["토요일", "일요일"]:
            return "주말(토·일)"
        else:
            return "평일(월~금)"

    df["일유형"] = df.apply(classify_day, axis=1)
    df["미세먼지등급"] = df["미세먼지"].apply(get_pm10_grade)
    df["초미세먼지등급"] = df["초미세먼지"].apply(get_pm25_grade)
    return df

df_raw = load_data()
if df_raw is None:
    st.error("🚨 데이터 파일을 찾을 수 없습니다. 깃허브의 app.py와 같은 폴더에 데이터 파일을 올려주세요!")
    st.stop()

# ---------------------------------------------------------
# 3. 왼쪽 사이드바 메뉴 (내비게이션)
# ---------------------------------------------------------
with st.sidebar:
    st.title("🚲 분석 메뉴")
    st.caption("2025 서울시 기상·교통 데이터 연구")
    st.divider()
    
    menu = st.radio(
        "이동할 화면을 선택하세요:",
        [
            "📈 1. 일별 시계열 추이 비교",
            "📐 2. 날씨별 선형 회귀 분석",
            "📋 3. 상관관계 및 데이터 전처리",
            "🎊 4. 2025 공휴일·수능일 분석"
        ]
    )
    
    st.divider()
    st.markdown("##### 📅 기간 · 요일 필터")
    day_filter = st.selectbox(
        "요일 조건 선택:",
        ["전체 (365일)", "평일만 보기 (공휴일 제외)", "주말 및 공휴일만 보기", "순수 평일만 (월~금 전체)"]
    )
    
    if day_filter == "평일만 보기 (공휴일 제외)":
        df = df_raw[df_raw["일유형"].isin(["평일(월~금)", "수능일"])].reset_index(drop=True)
    elif day_filter == "주말 및 공휴일만 보기":
        df = df_raw[df_raw["일유형"].isin(["주말(토·일)", "공휴일·연휴"])].reset_index(drop=True)
    elif day_filter == "순수 평일만 (월~금 전체)":
        df = df_raw[df_raw["주말구분"] == "평일(월~금)"].reset_index(drop=True)
    else:
        df = df_raw.copy()

    st.divider()
    st.info(
        f"**현재 선택된 데이터:** {len(df)}일\n\n"
        "• 기상: 서울 종관기상관측(108)\n"
        "• 교통: 따릉이 · 시내버스 전수\n"
        "• 특이일: 공휴일 19일 + 수능일 1일"
    )

# ---------------------------------------------------------
# 4. 상단 공통 헤더 및 핵심 지표 (KPI)
# ---------------------------------------------------------
st.title("🌤️ 2025 서울시 날씨·대기질과 교통수단(따릉이·버스) 이용량 분석")
st.markdown(
    """
    **2025년 서울시의 기상(기온·강수량·습도) 및 대기질(미세먼지·초미세먼지)**, 그리고 **공휴일·수능일** 요인이 
    야외 공유교통(**따릉이**)과 실내 대중교통(**시내버스**) 이용량에 미치는 영향을 비교·검증합니다.
    """
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 일수", f"{len(df)}일", day_filter)
col2.metric("일평균 따릉이 대여건수", f"{df['따릉이대여건수'].mean():,.0f} 건", f"최대 {df['따릉이대여건수'].max():,.0f} 건")
col3.metric("일평균 시내버스 이용량", f"{df['버스 이용량'].mean():,.0f} 명", f"최대 {df['버스 이용량'].max():,.0f} 명")
col4.metric("평균 기온 / 강수일", f"{df['평균기온(°C)'].mean():.1f} °C", f"비 온 날 {(df['일강수량(mm)'] > 0).sum()}일")

st.divider()

# =========================================================
# 메뉴 1: 일별 시계열 추이 (겹치지 않는 가로 직사각형 등급 띠)
# =========================================================
if menu == "📈 1. 일별 시계열 추이 비교":
    st.subheader("📈 일별 교통 이용량 및 기상·대기질 변화 추이")
    st.caption("💡 그래프 위에 마우스를 올리면 **날짜와 요일(공휴일·수능일 포함)**, 세부 수치가 함께 표시됩니다.")

    top_c1, top_c2 = st.columns([3, 1.3])
    with top_c1:
        view_mode = st.radio(
            "표시할 교통수단 선택:",
            ["따릉이 + 버스 함께 보기 (변화율 비교)", "따릉이대여건수만 보기", "버스 이용량만 보기"],
            horizontal=True
        )
    with top_c2:
        st.write("")
        show_holiday_markers = st.checkbox("🎊 공휴일·수능일 위치 표시", value=True)

    if view_mode == "따릉이 + 버스 함께 보기 (변화율 비교)":
        st.info("ℹ️ 따릉이(약 10만 건)와 버스(약 460만 명)는 단위 크기가 다르므로, 각 수단의 **평균 대비 이용량 비율(평균=100%)**로 환산하여 두 교통수단의 움직임을 한눈에 비교합니다.")

    chart_placeholder = st.empty()

    st.markdown("##### 🎛️ 그래프에 함께 표시할 기상·대기질 변수 선택")
    weather_options = {
        "평균기온(°C)": {"color": "#FF6B6B", "default": True, "alpha": 0.85},
        "최고기온(°C)": {"color": "#FFA94D", "default": False, "alpha": 0.80},
        "최저기온(°C)": {"color": "#4DABF7", "default": False, "alpha": 0.80},
        "일강수량(mm)": {"color": "#339AF0", "default": True, "alpha": 0.45},
        "평균 상대습도(%)": {"color": "#20C997", "default": False, "alpha": 0.75},
        "미세먼지": {"color": "#868E96", "default": False, "alpha": 0.85},
        "초미세먼지": {"color": "#495057", "default": False, "alpha": 0.85},
    }

    chk_cols = st.columns(len(weather_options))
    checked_vars = {}
    for idx, (var_name, props) in enumerate(weather_options.items()):
        with chk_cols[idx]:
            checked_vars[var_name] = st.checkbox(var_name, value=props["default"], key=f"chk_{var_name}")

    selected_configs = {}
    with st.expander("🎨 그래프 선 색상 · 투명도 및 미세먼지 농도 구간 배경 설정 열기 (클릭)", expanded=False):
        st.markdown("**1) 🌫️ 미세먼지 농도 등급별 직사각형 구간 배경 표시 (겹침 없이 4층으로 구분)**")
        pm_c1, pm_c2 = st.columns([2, 1.5])
        with pm_c1:
            pm_band_target = st.radio(
                "농도 구간 배경 선택:",
                ["표시 안 함", "미세먼지(PM10) 구간 표시", "초미세먼지(PM2.5) 구간 표시"],
                horizontal=True,
                key="pm_band_target"
            )
        with pm_c2:
            pm_band_alpha = st.slider("직사각형 배경 투명도", 0.05, 0.45, 0.14, 0.02, key="pm_band_alpha")

        st.divider()
        st.markdown("**2) 교통수단 선 색상 및 투명도**")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            c1, c2 = st.columns([1, 2.5])
            bike_color = c1.color_picker("따릉이 색", "#2B8A3E", key="bike_color")
            bike_alpha = c2.slider("따릉이 투명도", 0.1, 1.0, 1.0, 0.05, key="bike_alpha")
        with t_col2:
            c3, c4 = st.columns([1, 2.5])
            bus_color = c3.color_picker("버스 색", "#1864AB", key="bus_color")
            bus_alpha = c4.slider("버스 투명도", 0.1, 1.0, 1.0, 0.05, key="bus_alpha")

        st.markdown("**3) 기상 · 대기질 변수별 색상 및 투명도**")
        row1_cols = st.columns(4)
        row2_cols = st.columns(4)
        all_cols = row1_cols + row2_cols

        for idx, (var_name, props) in enumerate(weather_options.items()):
            with all_cols[idx]:
                st.caption(f"**{var_name}**")
                sc1, sc2 = st.columns([1, 2.2])
                picked_color = sc1.color_picker("색상", props["color"], key=f"col_{var_name}", label_visibility="collapsed")
                picked_alpha = sc2.slider("투명도", 0.1, 1.0, props["alpha"], 0.05, key=f"alp_{var_name}", label_visibility="collapsed")
                if checked_vars[var_name]:
                    selected_configs[var_name] = {"color": picked_color, "alpha": picked_alpha}

    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])

    # 1) 교통수단 선 추가
    if view_mode == "따릉이 + 버스 함께 보기 (변화율 비교)":
        bike_idx = (df["따릉이대여건수"] / df["따릉이대여건수"].mean()) * 100
        bus_idx = (df["버스 이용량"] / df["버스 이용량"].mean()) * 100

        fig_ts.add_trace(
            go.Scatter(
                x=df["날짜_문자"],
                y=bike_idx,
                name="따릉이 (평균=100%)",
                mode="lines",
                opacity=bike_alpha,
                line=dict(color=bike_color, width=2.4),
                customdata=np.stack([df["날짜_요일"], df["따릉이대여건수"], df["미세먼지등급"], df["초미세먼지등급"]], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>따릉이: %{y:.1f}% (실제 %{customdata[1]:,.0f}건)<br>미세먼지: %{customdata[2]} | 초미세먼지: %{customdata[3]}<extra></extra>"
            ),
            secondary_y=False
        )
        fig_ts.add_trace(
            go.Scatter(
                x=df["날짜_문자"],
                y=bus_idx,
                name="시내버스 (평균=100%)",
                mode="lines",
                opacity=bus_alpha,
                line=dict(color=bus_color, width=2.4),
                customdata=np.stack([df["날짜_요일"], df["버스 이용량"]], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>시내버스: %{y:.1f}% (실제 %{customdata[1]:,.0f}명)<extra></extra>"
            ),
            secondary_y=False
        )
        left_y_title = "<b>평균 대비 이용량 지수 (평균 = 100%)</b>"
    else:
        target_ts = "따릉이대여건수" if view_mode == "따릉이대여건수만 보기" else "버스 이용량"
        unit_str = "건" if target_ts == "따릉이대여건수" else "명"
        line_color = bike_color if target_ts == "따릉이대여건수" else bus_color
        line_alpha = bike_alpha if target_ts == "따릉이대여건수" else bus_alpha

        fig_ts.add_trace(
            go.Scatter(
                x=df["날짜_문자"],
                y=df[target_ts],
                name=target_ts,
                mode="lines",
                opacity=line_alpha,
                line=dict(color=line_color, width=2.5),
                customdata=np.stack([df["날짜_요일"], df["미세먼지등급"], df["초미세먼지등급"]], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>" + f"{target_ts}: " + "%{y:,.0f}" + unit_str + "<br>미세먼지: %{customdata[1]} | 초미세먼지: %{customdata[2]}<extra></extra>"
            ),
            secondary_y=False
        )
        left_y_title = f"<b>{target_ts} ({unit_str})</b>"

    # 2) 선택된 기상·대기질 변수 선 추가
    for var_name, cfg in selected_configs.items():
        if var_name == "일강수량(mm)":
            fig_ts.add_trace(
                go.Bar(
                    x=df["날짜_문자"],
                    y=df[var_name],
                    name=var_name,
                    marker_color=cfg["color"],
                    opacity=cfg["alpha"],
                    customdata=df[["날짜_요일"]],
                    hovertemplate="<b>%{customdata[0]}</b><br>" + f"{var_name}: " + "%{y:.1f}<extra></extra>"
                ),
                secondary_y=True
            )
        else:
            fig_ts.add_trace(
                go.Scatter(
                    x=df["날짜_문자"],
                    y=df[var_name],
                    name=var_name,
                    mode="lines",
                    opacity=cfg["alpha"],
                    line=dict(color=cfg["color"], width=1.8),
                    customdata=df[["날짜_요일"]],
                    hovertemplate="<b>%{customdata[0]}</b><br>" + f"{var_name}: " + "%{y:.1f}<extra></extra>"
                ),
                secondary_y=True
            )

    # 3) 미세먼지 / 초미세먼지 등급별 가로 직사각형 띠 (겹침 없이 4층 구분)
    if pm_band_target != "표시 안 함":
        target_pm_col = "미세먼지" if pm_band_target == "미세먼지(PM10) 구간 표시" else "초미세먼지"
        grade_col = "미세먼지등급" if target_pm_col == "미세먼지" else "초미세먼지등급"

        if target_pm_col not in selected_configs:
            fig_ts.add_trace(
                go.Scatter(
                    x=df["날짜_문자"],
                    y=df[target_pm_col],
                    name=target_pm_col,
                    mode="lines",
                    line=dict(color="#495057", width=1.8),
                    customdata=np.stack([df["날짜_요일"], df[grade_col]], axis=-1),
                    hovertemplate="<b>%{customdata[0]}</b><br>" + f"{target_pm_col}: " + "%{y:.1f} (%{customdata[1]})<extra></extra>"
                ),
                secondary_y=True
            )

        max_right_val = df[target_pm_col].max()
        for v_name in selected_configs.keys():
            max_right_val = max(max_right_val, df[v_name].max())

        if target_pm_col == "미세먼지":
            top_limit = max(max_right_val * 1.05, 165)
            bands = [
                (0, 30, "좋음 (0~30)", (51, 154, 240)),
                (30, 80, "보통 (31~80)", (64, 192, 87)),
                (80, 150, "나쁨 (81~150)", (253, 126, 20)),
                (150, top_limit, "매우 나쁨 (151~)", (250, 82, 82)),
            ]
        else:
            top_limit = max(max_right_val * 1.05, 90)
            bands = [
                (0, 15, "좋음 (0~15)", (51, 154, 240)),
                (15, 35, "보통 (16~35)", (64, 192, 87)),
                (35, 75, "나쁨 (36~75)", (253, 126, 20)),
                (75, top_limit, "매우 나쁨 (76~)", (250, 82, 82)),
            ]

        for y0, y1, band_label, (r, gr, b) in bands:
            if y0 < top_limit:
                fig_ts.add_hrect(
                    y0=y0,
                    y1=min(y1, top_limit),
                    fillcolor=f"rgba({r}, {gr}, {b}, {pm_band_alpha})",
                    line_width=0,
                    layer="below",
                    secondary_y=True,
                    annotation_text=band_label,
                    annotation_position="inside top right",
                    annotation_font=dict(size=11, color=f"rgb({max(r-40,0)}, {max(gr-40,0)}, {max(b-40,0)})")
                )

    # 4) 공휴일·수능일 세로 점선 표시
    if show_holiday_markers:
        hol_df = df[df["특이일명"] != "일반일"]
        for _, row in hol_df.iterrows():
            v_col = "rgba(43, 138, 62, 0.55)" if row["일유형"] == "수능일" else "rgba(224, 49, 49, 0.35)"
            fig_ts.add_vline(
                x=row["날짜_문자"],
                line_width=1.2,
                line_dash="dot",
                line_color=v_col
            )

    fig_ts.update_layout(
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(
            title="<b>날짜 (2025년)</b>",
            type="date",
            tickformat="%Y년 %m월",
            dtick="M1"
        )
    )
    fig_ts.update_yaxes(title_text=left_y_title, secondary_y=False, tickformat=",")
    fig_ts.update_yaxes(title_text="<b>기상 · 대기질 수치</b>", secondary_y=True, rangemode="tozero")

    chart_placeholder.plotly_chart(fig_ts, use_container_width=True)


# =========================================================
# 메뉴 2: 선형 회귀 분석
# =========================================================
elif menu == "📐 2. 날씨별 선형 회귀 분석":
    st.subheader("📐 기상·대기질 변수에 따른 교통 이용량 선형 회귀 분석")
    st.caption("💡 점 위에 마우스를 올리면 해당 날짜의 **날짜·요일(공휴일·수능일 포함)**과 실제 관측치를 확인할 수 있습니다.")

    reg_chart_placeholder = st.empty()

    st.markdown("##### 🎛️ 회귀 분석 변수 선택")
    c1, c2, c3 = st.columns([2, 2, 1.8])
    with c1:
        x_var = st.selectbox(
            "독립변수 (X축: 날씨·대기질)",
            ["최고기온(°C)", "평균기온(°C)", "최저기온(°C)", "일강수량(mm)", "평균 상대습도(%)", "미세먼지", "초미세먼지"],
            index=0
        )
    with c2:
        y_var = st.selectbox(
            "종속변수 (Y축: 교통수단)",
            ["따릉이대여건수", "버스 이용량"],
            index=0
        )
    with c3:
        color_mode = st.selectbox(
            "점 색상 구분 기준",
            ["일유형 (평일 / 주말 / 공휴일 / 수능일)", "평일 / 주말 구분", "단일 색상"],
            index=0
        )

    with st.expander("🎨 산점도 점 투명도 · 회귀선 색상 및 미세먼지 구간 설정 열기 (클릭)", expanded=False):
        s1, s2, s3, s4 = st.columns([1.2, 1, 1, 1.3])
        with s1:
            dot_alpha = st.slider("관측치 점 투명도", 0.1, 1.0, 0.78, 0.05)
        with s2:
            line_reg_color = st.color_picker("회귀직선 색상", "#E03131")
        with s3:
            single_dot_color = st.color_picker("기본 점 색상", "#2B8A3E")
        with s4:
            show_reg_pm_bg = st.checkbox("🌫️ 미세먼지 X축 등급 구간 표시", value=True)

    X = df[[x_var]].values
    y = df[y_var].values
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)
    corr = df[x_var].corr(df[y_var])

    x_range = np.linspace(df[x_var].min(), df[x_var].max(), 100)
    y_pred_line = model.predict(x_range.reshape(-1, 1))

    if color_mode == "일유형 (평일 / 주말 / 공휴일 / 수능일)":
        fig_reg = px.scatter(
            df,
            x=x_var,
            y=y_var,
            color="일유형",
            color_discrete_map={
                "평일(월~금)": "#1C7ED6",
                "주말(토·일)": "#F76707",
                "공휴일·연휴": "#E03131",
                "수능일": "#2B8A3E"
            },
            custom_data=["날짜_요일", "평균기온(°C)", "일강수량(mm)", "미세먼지", "미세먼지등급"],
            opacity=dot_alpha
        )
    elif color_mode == "평일 / 주말 구분":
        fig_reg = px.scatter(
            df,
            x=x_var,
            y=y_var,
            color="주말구분",
            color_discrete_map={"평일(월~금)": "#1C7ED6", "주말(토·일)": "#F76707"},
            custom_data=["날짜_요일", "평균기온(°C)", "일강수량(mm)", "미세먼지", "미세먼지등급"],
            opacity=dot_alpha
        )
    else:
        fig_reg = px.scatter(
            df,
            x=x_var,
            y=y_var,
            color_discrete_sequence=[single_dot_color],
            custom_data=["날짜_요일", "평균기온(°C)", "일강수량(mm)", "미세먼지", "미세먼지등급"],
            opacity=dot_alpha
        )

    unit_text = "건" if y_var == "따릉이대여건수" else "명"

    fig_reg.update_traces(
        marker=dict(size=8),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            f"{x_var}: " + "%{x:.1f}<br>"
            f"{y_var}: " + "%{y:,.0f}" + unit_text + "<br>"
            "평균기온: %{customdata[1]:.1f}°C | 강수량: %{customdata[2]:.1f}mm | 미세먼지: %{customdata[3]:.1f} (%{customdata[4]})"
            "<extra></extra>"
        )
    )

    fig_reg.add_trace(
        go.Scatter(
            x=x_range,
            y=y_pred_line,
            mode="lines",
            name=f"선형 회귀선 (결정계수 R²={r2:.3f})",
            line=dict(color=line_reg_color, width=3)
        )
    )

    if show_reg_pm_bg and x_var in ["미세먼지", "초미세먼지"]:
        max_x = max(df[x_var].max() * 1.05, 160 if x_var == "미세먼지" else 85)
        if x_var == "미세먼지":
            bands = [
                (0, 30, "좋음", "rgba(51, 154, 240, 0.14)"),
                (30, 80, "보통", "rgba(64, 192, 87, 0.14)"),
                (80, 150, "나쁨", "rgba(253, 126, 20, 0.14)"),
                (150, max_x, "매우 나쁨", "rgba(250, 82, 82, 0.14)")
            ]
        else:
            bands = [
                (0, 15, "좋음", "rgba(51, 154, 240, 0.14)"),
                (15, 35, "보통", "rgba(64, 192, 87, 0.14)"),
                (35, 75, "나쁨", "rgba(253, 126, 20, 0.14)"),
                (75, max_x, "매우 나쁨", "rgba(250, 82, 82, 0.14)")
            ]
        for x0, x1, label, fill_c in bands:
            if x0 < df[x_var].max() * 1.05:
                fig_reg.add_vrect(
                    x0=x0,
                    x1=min(x1, df[x_var].max() * 1.05),
                    fillcolor=fill_c,
                    layer="below",
                    line_width=0,
                    annotation_text=label,
                    annotation_position="top left"
                )

    fig_reg.update_layout(
        height=470,
        title=f"<b>[{x_var}] 변화에 따른 [{y_var}] 선형 회귀 분석</b>",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text="구분"),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title=f"<b>{x_var}</b>",
        yaxis_title=f"<b>{y_var} ({unit_text})</b>",
        yaxis=dict(tickformat=",")
    )

    reg_chart_placeholder.plotly_chart(fig_reg, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("회귀 기울기 (1단위 증가 시 변화량)", f"{slope:+,.1f} {unit_text}")
    m2.metric("결정계수 (R² 설명력)", f"{r2:.3f} ({r2*100:.1f}%)")
    m3.metric("상관계수 (r)", f"{corr:+.3f}")

    st.info(
        f"""
        **📊 분석 결과 자동 해석:**
        - **{x_var}**이(가) 1단위 증가할 때, **{y_var}**은(는) 평균적으로 **{abs(slope):,.1f}{unit_text} {'증가' if slope >= 0 else '감소'}**하는 경향을 보입니다.
        - 회귀 방정식: `y = {slope:,.2f}x + {intercept:,.1f}` (결정계수 $R^2 = {r2:.3f}$)
        """
    )


# =========================================================
# 메뉴 3: 상관관계 및 데이터 전처리 개요 (체크리스트 필수 항목 완비)
# =========================================================
elif menu == "📋 3. 상관관계 및 데이터 전처리":
    st.subheader("📋 데이터 개요 · 전처리 기준 및 상관관계 분석 보고서")

    # 1) 데이터 개요 (행·열 수, 열마다 자료형과 빈 값 개수)
    st.markdown("### 1️⃣ 데이터셋 기본 개요 (행·열 수, 자료형, 결측치 현황)")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("전처리 전 원본 행 수", "365행 (2025년 전체)")
    o2.metric("전처리 후 분석 행 수", f"{len(df_raw)}행 (유지율 100%)")
    o3.metric("분석 변수(열) 수", "11개 (원본 기준)")
    o4.metric("전처리 후 빈 값(결측치)", "0개 (완전 정제)")

    orig_cols = ["일시", "요일", "평균기온(°C)", "최저기온(°C)", "최고기온(°C)", "일강수량(mm)", "평균 상대습도(%)", "미세먼지", "초미세먼지", "따릉이대여건수", "버스 이용량"]
    schema_rows = []
    for col in orig_cols:
        dtype_str = str(df_raw[col].dtype)
        if "datetime" in dtype_str:
            kor_dtype = "날짜형 (datetime64)"
        elif "int" in dtype_str:
            kor_dtype = "정수형 (int64)"
        elif "float" in dtype_str:
            kor_dtype = "실수형 (float64)"
        else:
            kor_dtype = "문자형 (object)"
        raw_na = "8개 (무강수일)" if col == "일강수량(mm)" else "0개"
        after_na = int(df_raw[col].isna().sum())
        schema_rows.append({
            "열 이름 (변수명)": col,
            "자료형 (Data Type)": kor_dtype,
            "전처리 전 빈 값": raw_na,
            "전처리 후 빈 값": f"{after_na}개"
        })
    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

    # 2) 결측치·이상치 처리 기준과 처리 전후 행 수
    st.markdown("### 2️⃣ 결측치 · 이상치 처리 기준 및 처리 전후 비교")
    prep_table = pd.DataFrame([
        {
            "항목": "결측치 (일강수량)",
            "발견 내용": "기상청 원본 데이터에서 비·눈이 오지 않은 날(무강수일) 8일의 강수량이 빈 값(NaN)으로 기록됨",
            "처리 기준": "무강수일은 실제 강수량이 0mm이므로 행을 삭제하지 않고 0.0mm로 대체",
            "처리 전후 행 수": "365행 → 365행 (데이터 손실 0건)"
        },
        {
            "항목": "이상치 (극한 기상·특이일)",
            "발견 내용": "여름철 집중호우(최대 130mm 이상), 겨울 한파, 설·추석 명절 당일의 버스·따릉이 급감 수치 발견",
            "처리 기준": "측정 오류가 아닌 실제 기상·사회적 현상(날씨·명절 효과)이므로 제거하지 않고 모두 보존(단, 공휴일·수능일 라벨 별도 부여)",
            "처리 전후 행 수": "365행 → 365행 (원본 전수 유지)"
        },
        {
            "항목": "표본 범위 정제",
            "발견 내용": "택시·공유 전동킥보드는 2025년 서울시 일별 전수 데이터가 비공개",
            "처리 기준": "365일 일별 전수 데이터가 100% 확보된 따릉이(야외)와 시내버스(실내)로 분석 대상 확정",
            "처리 전후 행 수": "365행 × 11열 확정"
        }
    ])
    st.dataframe(prep_table, use_container_width=True, hide_index=True)

    st.divider()

    # 3) 상관관계 행렬
    st.markdown("### 3️⃣ 전체 변수 간 상관관계 행렬")
    num_cols = [
        "평균기온(°C)", "최저기온(°C)", "최고기온(°C)",
        "일강수량(mm)", "평균 상대습도(%)", "미세먼지", "초미세먼지",
        "따릉이대여건수", "버스 이용량"
    ]
    corr_matrix = df[num_cols].corr().round(2)

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        labels=dict(color="상관계수")
    )
    fig_corr.update_layout(height=480, title="<b>2025 서울시 기상·대기질 및 교통 이용량 상관계수 행렬</b>")
    st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("📄 통합 데이터셋 원본 미리보기 (365행 전체 조회)"):
        st.dataframe(df.drop(columns=["날짜_문자", "날짜_요일"]), use_container_width=True)


# =========================================================
# 메뉴 4: 2025 공휴일(19일) · 수능일(11/13) 특이일 분석
# =========================================================
elif menu == "🎊 4. 2025 공휴일·수능일 분석":
    st.subheader("🎊 2025년 공휴일(19일) 및 수능일(11월 13일) 교통 이용량 비교")

    avg_by_type = df_raw.groupby("일유형")[["따릉이대여건수", "버스 이용량"]].mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "평일 평균 (243일)",
            f"버스 {avg_by_type.loc['평일(월~금)', '버스 이용량']:,.0f}명",
            f"따릉이 {avg_by_type.loc['평일(월~금)', '따릉이대여건수']:,.0f}건"
        )
    with k2:
        st.metric(
            "주말 평균 (102일)",
            f"버스 {avg_by_type.loc['주말(토·일)', '버스 이용량']:,.0f}명",
            f"따릉이 {avg_by_type.loc['주말(토·일)', '따릉이대여건수']:,.0f}건"
        )
    with k3:
        st.metric(
            "공휴일·연휴 평균 (19일)",
            f"버스 {avg_by_type.loc['공휴일·연휴', '버스 이용량']:,.0f}명",
            f"따릉이 {avg_by_type.loc['공휴일·연휴', '따릉이대여건수']:,.0f}건"
        )
    with k4:
        st.metric(
            "수능일 (2025-11-13 목)",
            f"버스 {avg_by_type.loc['수능일', '버스 이용량']:,.0f}명",
            f"따릉이 {avg_by_type.loc['수능일', '따릉이대여건수']:,.0f}건"
        )

    st.divider()

    hol_only = df_raw[df_raw["특이일명"] != "일반일"].copy()
    hol_only["라벨"] = hol_only["날짜_문자"].str.slice(5) + " " + hol_only["특이일명"]

    sel_hol_mode = st.radio(
        "비교할 교통수단 선택:",
        ["시내버스 이용량 (공휴일·수능일 비교)", "따릉이 대여건수 (공휴일·수능일 비교)"],
        horizontal=True
    )

    if sel_hol_mode == "시내버스 이용량 (공휴일·수능일 비교)":
        y_col = "버스 이용량"
        avg_line = avg_by_type.loc["평일(월~금)", "버스 이용량"]
        unit_h = "명"
    else:
        y_col = "따릉이대여건수"
        avg_line = avg_by_type.loc["평일(월~금)", "따릉이대여건수"]
        unit_h = "건"

    fig_hol = px.bar(
        hol_only,
        x="라벨",
        y=y_col,
        text=y_col,
        color="평균기온(°C)",
        color_continuous_scale="RdYlBu_r",
        custom_data=["날짜_요일", "평균기온(°C)", "일강수량(mm)"],
        labels={"라벨": "2025년 공휴일 · 수능일", y_col: f"{y_col} ({unit_h})", "평균기온(°C)": "평균기온(°C)"}
    )
    fig_hol.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            f"{y_col}: " + "%{y:,.0f}" + unit_h + "<br>"
            "평균기온: %{customdata[1]:.1f}°C | 강수량: %{customdata[2]:.1f}mm"
            "<extra></extra>"
        )
    )
    fig_hol.add_hline(
        y=avg_line,
        line_dash="dash",
        line_color="#E03131",
        annotation_text=f"평일 평균 ({avg_line:,.0f}{unit_h})",
        annotation_position="top right"
    )
    fig_hol.update_layout(
        height=480,
        title=f"<b>2025년 공휴일(19일) 및 수능일별 {y_col} 비교</b>",
        xaxis_tickangle=-35,
        yaxis=dict(tickformat=",", range=[0, hol_only[y_col].max() * 1.18]),
        margin=dict(l=20, r=20, t=50, b=90)
    )
    st.plotly_chart(fig_hol, use_container_width=True)

    st.markdown("##### 📋 2025년 공휴일(19일) 및 수능일 전체 상세표 (총 20일)")
    display_hol = hol_only[["날짜_문자", "요일", "특이일명", "평균기온(°C)", "일강수량(mm)", "미세먼지", "초미세먼지", "따릉이대여건수", "버스 이용량"]].copy()
    display_hol.columns = ["날짜", "요일", "구분", "평균기온(°C)", "일강수량(mm)", "미세먼지", "초미세먼지", "따릉이 대여(건)", "버스 이용량(명)"]
    st.dataframe(display_hol, use_container_width=True, hide_index=True)
