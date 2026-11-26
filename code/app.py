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
    df["날짜_문자"] = df["일시"].dt.strftime("%Y-%m-%d")
    df["날짜_요일"] = df["날짜_문자"] + " (" + df["요일"] + ")"
    df["주말구분"] = df["요일"].apply(lambda x: "주말(토·일)" if x in ["토요일", "일요일"] else "평일(월~금)")
    return df

df_raw = load_data()
if df_raw is None:
    st.error("🚨 데이터 파일을 찾을 수 없습니다. 깃허브의 app.py와 같은 폴더에 데이터 파일을 올려주세요!")
    st.stop()

# ---------------------------------------------------------
# 3. 왼쪽 사이드바 메뉴 (제미나이 스타일 내비게이션)
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
            "📋 3. 상관관계 및 데이터 전처리"
        ]
    )
    
    st.divider()
    st.markdown("##### 📅 기간 · 요일 필터")
    day_filter = st.selectbox(
        "요일 조건 선택:",
        ["전체 (365일)", "평일만 보기 (월~금)", "주말만 보기 (토·일)"]
    )
    
    if day_filter == "평일만 보기 (월~금)":
        df = df_raw[df_raw["주말구분"] == "평일(월~금)"].copy()
    elif day_filter == "주말만 보기 (토·일)":
        df = df_raw[df_raw["주말구분"] == "주말(토·일)"].copy()
    else:
        df = df_raw.copy()

    st.divider()
    st.info(
        f"**현재 선택된 데이터:** {len(df)}일\n\n"
        "• 기상: 서울 종관기상관측(108)\n"
        "• 교통: 따릉이 · 시내버스 전수"
    )

# ---------------------------------------------------------
# 4. 상단 공통 헤더 및 핵심 지표 (KPI)
# ---------------------------------------------------------
st.title("🌤️ 2025 서울시 날씨·대기질과 교통수단(따릉이·버스) 이용량 분석")
st.markdown(
    """
    **2025년 서울시의 기상(기온·강수량·습도) 및 대기질(미세먼지·초미세먼지)** 변화가 
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
# 메뉴 1: 일별 시계열 추이 (따릉이·버스 개별 및 동시 비교)
# =========================================================
if menu == "📈 1. 일별 시계열 추이 비교":
    st.subheader("📈 일별 교통 이용량 및 기상·대기질 변화 추이")
    st.caption("💡 그래프 위에 마우스를 올리면 **날짜와 요일**, 세부 수치가 함께 표시됩니다.")

    view_mode = st.radio(
        "표시할 교통수단 선택:",
        ["따릉이 + 버스 함께 보기 (변화율 비교)", "따릉이대여건수만 보기", "버스 이용량만 보기"],
        horizontal=True
    )

    weather_options = {
        "평균기온(°C)": {"color": "#FF6B6B", "default": True},
        "최고기온(°C)": {"color": "#FFA94D", "default": False},
        "최저기온(°C)": {"color": "#4DABF7", "default": False},
        "일강수량(mm)": {"color": "#339AF0", "default": True},
        "평균 상대습도(%)": {"color": "#20C997", "default": False},
        "미세먼지": {"color": "#868E96", "default": False},
        "초미세먼지": {"color": "#495057", "default": False},
    }

    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])

    # 1) 교통수단 선 추가 (함께 보기 vs 단독 보기)
    if view_mode == "따릉이 + 버스 함께 보기 (변화율 비교)":
        st.info("ℹ️ 따릉이(약 10만 건)와 버스(약 460만 명)는 단위 크기가 크게 다르므로, 각 수단의 **평균 대비 이용량 비율(평균=100%)**로 환산하여 두 교통수단의 움직임을 한눈에 비교합니다.")
        bike_idx = (df["따릉이대여건수"] / df["따릉이대여건수"].mean()) * 100
        bus_idx = (df["버스 이용량"] / df["버스 이용량"].mean()) * 100

        fig_ts.add_trace(
            go.Scatter(
                x=df["일시"],
                y=bike_idx,
                name="따릉이 (평균=100%)",
                mode="lines",
                line=dict(color="#2B8A3E", width=2.2),
                customdata=np.stack([df["날짜_요일"], df["따릉이대여건수"]], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>따릉이: %{y:.1f}% (실제 %{customdata[1]:,.0f}건)<extra></extra>"
            ),
            secondary_y=False
        )
        fig_ts.add_trace(
            go.Scatter(
                x=df["일시"],
                y=bus_idx,
                name="시내버스 (평균=100%)",
                mode="lines",
                line=dict(color="#1864AB", width=2.2),
                customdata=np.stack([df["날짜_요일"], df["버스 이용량"]], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>시내버스: %{y:.1f}% (실제 %{customdata[1]:,.0f}명)<extra></extra>"
            ),
            secondary_y=False
        )
        left_y_title = "<b>평균 대비 이용량 지수 (평균 = 100%)</b>"
    else:
        target_ts = "따릉이대여건수" if view_mode == "따릉이대여건수만 보기" else "버스 이용량"
        unit_str = "건" if target_ts == "따릉이대여건수" else "명"
        line_color = "#2B8A3E" if target_ts == "따릉이대여건수" else "#1864AB"

        fig_ts.add_trace(
            go.Scatter(
                x=df["일시"],
                y=df[target_ts],
                name=target_ts,
                mode="lines",
                line=dict(color=line_color, width=2.3),
                customdata=df[["날짜_요일"]],
                hovertemplate="<b>%{customdata[0]}</b><br>" + f"{target_ts}: " + "%{y:,.0f}" + unit_str + "<extra></extra>"
            ),
            secondary_y=False
        )
        left_y_title = f"<b>{target_ts} ({unit_str})</b>"

    # 그래프가 들어갈 자리 예약 (체크박스를 그래프 아래에 두기 위함)
    chart_placeholder = st.empty()

    # 그래프 아래 체크박스 UI
    st.markdown("##### 🎛️ 그래프에 함께 표시할 기상·대기질 변수 선택 (체크박스로 켜고 끄기)")
    chk_cols = st.columns(len(weather_options))
    selected_vars = []

    for idx, (var_name, props) in enumerate(weather_options.items()):
        with chk_cols[idx]:
            is_checked = st.checkbox(var_name, value=props["default"], key=f"ts_chk_{var_name}")
            if is_checked:
                selected_vars.append(var_name)

    for var_name in selected_vars:
        props = weather_options[var_name]
        if var_name == "일강수량(mm)":
            fig_ts.add_trace(
                go.Bar(
                    x=df["일시"],
                    y=df[var_name],
                    name=var_name,
                    marker_color=props["color"],
                    opacity=0.45,
                    customdata=df[["날짜_요일"]],
                    hovertemplate="<b>%{customdata[0]}</b><br>" + f"{var_name}: " + "%{y:.1f}<extra></extra>"
                ),
                secondary_y=True
            )
        else:
            fig_ts.add_trace(
                go.Scatter(
                    x=df["일시"],
                    y=df[var_name],
                    name=var_name,
                    mode="lines",
                    line=dict(color=props["color"], width=1.6, dash="dot"),
                    customdata=df[["날짜_요일"]],
                    hovertemplate="<b>%{customdata[0]}</b><br>" + f"{var_name}: " + "%{y:.1f}<extra></extra>"
                ),
                secondary_y=True
            )

    # X축 날짜 한글 포맷 설정 (Jan 2025 -> 2025년 01월)
    fig_ts.update_layout(
        height=480,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            title="<b>날짜 (2025년)</b>",
            tickformat="%Y년 %m월",
            dtick="M1"
        )
    )
    fig_ts.update_yaxes(title_text=left_y_title, secondary_y=False, tickformat=",")
    fig_ts.update_yaxes(title_text="<b>기상 · 대기질 수치</b>", secondary_y=True)

    chart_placeholder.plotly_chart(fig_ts, use_container_width=True)


# =========================================================
# 메뉴 2: 선형 회귀 분석
# =========================================================
elif menu == "📐 2. 날씨별 선형 회귀 분석":
    st.subheader("📐 기상·대기질 변수에 따른 교통 이용량 선형 회귀 분석")
    st.caption("💡 점 위에 마우스를 올리면 해당 날짜의 **날짜·요일**과 실제 관측치를 확인할 수 있습니다.")

    reg_chart_placeholder = st.empty()

    st.markdown("##### 🎛️ 회귀 분석 변수 선택")
    c1, c2, c3 = st.columns([2, 2, 1.5])
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
        st.write("")
        st.write("")
        split_weekend = st.checkbox("평일 / 주말 색상 구분", value=True)

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

    if split_weekend:
        fig_reg = px.scatter(
            df,
            x=x_var,
            y=y_var,
            color="주말구분",
            color_discrete_map={"평일(월~금)": "#1C7ED6", "주말(토·일)": "#F76707"},
            custom_data=["날짜_요일", "평균기온(°C)", "일강수량(mm)", "미세먼지"],
            opacity=0.75
        )
    else:
        fig_reg = px.scatter(
            df,
            x=x_var,
            y=y_var,
            color_discrete_sequence=["#2B8A3E"],
            custom_data=["날짜_요일", "평균기온(°C)", "일강수량(mm)", "미세먼지"],
            opacity=0.75
        )

    unit_text = "건" if y_var == "따릉이대여건수" else "명"

    fig_reg.update_traces(
        marker=dict(size=8),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            f"{x_var}: " + "%{x:.1f}<br>"
            f"{y_var}: " + "%{y:,.0f}" + unit_text + "<br>"
            "평균기온: %{customdata[1]:.1f}°C | 강수량: %{customdata[2]:.1f}mm | 미세먼지: %{customdata[3]:.1f}"
            "<extra></extra>"
        )
    )

    fig_reg.add_trace(
        go.Scatter(
            x=x_range,
            y=y_pred_line,
            mode="lines",
            name=f"선형 회귀선 (결정계수 R²={r2:.3f})",
            line=dict(color="#E03131", width=3)
        )
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
# 메뉴 3: 상관관계 및 데이터 전처리 개요
# =========================================================
elif menu == "📋 3. 상관관계 및 데이터 전처리":
    st.subheader("📋 전체 변수 간 상관관계 행렬 및 데이터 전처리 보고서")

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

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            ### 🔍 핵심 연구 결론
            1. **야외 수단(따릉이)의 높은 기상 민감도:** 최고기온과 따릉이 대여건수는 강한 양의 상관관계($r = +0.66, R^2 = 0.435$)를 보였으며, 강수량 증가 시 이용량이 뚜렷하게 감소($r = -0.33$)했습니다.
            2. **대중교통(시내버스)의 비탄력성:** 시내버스는 기온($R^2 = 0.015$)과 강수량($R^2 = 0.001$)의 영향을 거의 받지 않았으며, 기상 변화보다 **평일·주말(요일) 요인**에 의해 이용량이 결정되는 필수 교통수단임을 확인했습니다.
            3. **미세먼지의 제한적 영향:** 2025년 일평균 기준 미세먼지($r = +0.04$)와 초미세먼지($r = -0.02$)는 단독 선형 모델에서 교통 이용량에 미치는 설명력이 낮았습니다.
            """
        )
    with col_b:
        st.markdown(
            """
            ### 🛠️ 데이터 출처 및 전처리·연구 한계
            - **데이터 출처:** 기상청 기상자료개방포털(종관기상관측 ASOS 서울 108지점), 서울열린데이터광장(공공자전거 일별 대여건수, 버스노선별 승하차 인원, 대기오염도 측정 정보).
            - **전처리 과정:** 2025년 365일(2025-01-01 ~ 2025-12-31) 기준 일별 병합, 무강수일 결측치(8일) `0.0mm` 대체, 서울시 측정소별 대기질 일평균 산출 및 달력 요일 매핑 검증 완료.
            - **연구의 한계 및 표본 범위:** 택시 및 민간 공유 전기자전거는 2025년 일별 서울시 전체 이용 데이터가 공개되지 않아 분석 대상에서 제외하고, 완전한 전수 데이터가 확보된 **따릉이**와 **시내버스**로 비교 범위를 명확히 설정했습니다.
            """
        )

    with st.expander("📄 통합 데이터셋 원본 미리보기 (전체 행 조회)"):
        st.dataframe(df.drop(columns=["날짜_문자", "날짜_요일"]), use_container_width=True)
