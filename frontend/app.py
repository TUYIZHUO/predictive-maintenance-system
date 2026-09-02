# -*- coding: utf-8 -*-
"""前端看板：Streamlit。视觉规范（机能工业风：奶杏 + 深海灰 + 火星橙）由启动命令 --theme.* 参数配置（见 启动.bat / 启动说明.txt）。

启动（在项目根目录）：
    streamlit run frontend/app.py --theme.primaryColor "#EB6127" --theme.backgroundColor "#152639" --theme.secondaryBackgroundColor "#1C2F44" --theme.textColor "#F1DDBC"
"""
import matplotlib
import pandas as pd
import requests
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="设备预测性维护看板", layout="wide")

# ---------------------------------------------------------------------------
# 全局样式：状态徽章 + 品牌页头 + 背景层次（其余由 config.toml 主题统一）
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp{background:linear-gradient(180deg,#1C2F44 0%,#152639 420px) !important;}
    div[data-testid="stVerticalBlockBorderWrapper"]{
        border-radius:10px;background:#1C2F44;
        box-shadow:0 2px 6px rgba(0,0,0,0.28);}
    .badge{display:inline-block;padding:3px 12px;border-radius:999px;font-size:13px;
           font-weight:600;line-height:1.5;}
    .badge-success{background:rgba(34,197,94,0.16);color:#4ADE80;border:1px solid rgba(74,222,128,0.45);}
    .badge-warning{background:rgba(235,97,39,0.16);color:#F98E5A;border:1px solid rgba(235,97,39,0.5);}
    .badge-danger{background:rgba(220,38,38,0.16);color:#F87171;border:1px solid rgba(248,113,113,0.45);}
    .badge-info{background:rgba(241,221,188,0.10);color:#F1DDBC;border:1px solid rgba(241,221,188,0.32);}
    .brand-title{font-size:24px;font-weight:700;color:#F1DDBC;
                 border-left:4px solid #EB6127;padding-left:12px;}
    .brand-sub{font-size:13px;color:#C9B896;margin-top:4px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def badge(text: str, kind: str = "info") -> str:
    """生成一个语义色状态徽章。kind: success / warning / danger / info。"""
    return f'<span class="badge badge-{kind}">{text}</span>'


# ---------------------------------------------------------------------------
# 后端连接状态
# ---------------------------------------------------------------------------
def check_backend() -> bool:
    try:
        return requests.get(f"{API}/", timeout=1.5).status_code == 200
    except Exception:
        return False


if "backend_ok" not in st.session_state:
    st.session_state.backend_ok = check_backend()

status_badge = (
    badge("● 后端已连接", "success") if st.session_state.backend_ok
    else badge("● 后端未连接", "danger")
)

# ---------------------------------------------------------------------------
# 品牌页头
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding-bottom:14px;border-bottom:1px solid #2A4058;margin-bottom:20px;">
      <div>
        <div class="brand-title">设备预测性维护看板</div>
        <div class="brand-sub">面向铣床设备的故障预测 · 异常检测 · 维护排程</div>
      </div>
      <div>{status_badge}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 数据列约定（Excel 表头）
# ---------------------------------------------------------------------------
FEATURE_COLS = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear", "type"]
DEVICE_COLS = ["device_id", "risk", "downtime_cost", "resource", "maintenance_time"]
FEATURE_RENAME = {"air_temperature": "空气温度", "process_temperature": "工艺温度",
                  "rotational_speed": "转速", "torque": "扭矩",
                  "tool_wear": "刀具磨损", "type": "类型"}
DEVICE_RENAME = {"device_id": "设备编号", "risk": "故障风险",
                 "downtime_cost": "停机损失(元/时)", "resource": "资源(工时)",
                 "maintenance_time": "维护耗时(时)"}


def read_excel(uploaded, cols):
    """读取上传的 Excel，校验列名并转为 list[dict]，返回 (记录, 错误)。"""
    df = pd.read_excel(uploaded, engine="openpyxl")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None, f"缺少列：{missing}，请按 {cols} 设置表头"
    recs = []
    for _, r in df.iterrows():
        rec = {}
        for c in cols:
            v = r[c]
            rec[c] = str(v) if c in ("type", "device_id") else float(v)
        recs.append(rec)
    return recs, None


def backend_error():
    st.error("后端调用失败，请先启动后端服务（uvicorn backend.main:app）")


# 风险等级配色（与页面状态徽章一致：绿=低、橙=中、红=高）
RISK_COLORS = {0: "#4ADE80", 1: "#F98E5A", 2: "#F87171"}
RISK_LABELS = {0: "低风险", 1: "中风险", 2: "高风险"}


def draw_gantt(plan):
    """把维护计划画成甘特图：横轴为维护批次（天），纵轴为设备，颜色代表风险等级。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    rows = sorted(plan, key=lambda p: p["priority"])   # 优先级 1→N
    n = len(rows)
    max_day = max(p["batch_day"] for p in rows)

    fig, ax = plt.subplots(figsize=(8.5, max(3.0, n * 0.55)))
    fig.patch.set_facecolor("#152639")
    ax.set_facecolor("#152639")

    for i, p in enumerate(rows):
        y = n - 1 - i                      # 优先级 1 排在最上面
        ax.barh(y, 0.9, left=p["batch_day"], height=0.62,
                color=RISK_COLORS.get(p["risk_level"], "#F98E5A"),
                edgecolor="#152639", zorder=3)

    ax.set_yticks(range(n))
    ax.set_yticklabels([p["device_id"] for p in reversed(rows)],
                       color="#F1DDBC", fontsize=9)
    ax.set_xlim(-0.05, max_day + 1)
    ax.set_xticks([d + 0.5 for d in range(max_day + 1)])
    ax.set_xticklabels([f"第 {d + 1} 天" for d in range(max_day + 1)],
                       color="#F1DDBC", fontsize=9)
    ax.set_xlabel("维护批次", color="#F1DDBC", fontsize=10)
    ax.set_title("维护排程甘特图", color="#F1DDBC", fontsize=13,
                 fontweight="bold", pad=10)

    ax.grid(axis="x", color="#2A4058", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#2A4058")
    ax.tick_params(colors="#F1DDBC", length=0)

    handles = [Patch(facecolor=RISK_COLORS[k], edgecolor="#152639",
                     label=RISK_LABELS[k]) for k in (0, 1, 2)]
    ax.legend(handles=handles, loc="upper right", frameon=True,
              facecolor="#1C2F44", edgecolor="#2A4058",
              labelcolor="#F1DDBC", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 顶部导航（分段控件，支持总览页点击跳转）
# ---------------------------------------------------------------------------
PAGES = ["总览", "故障预测", "异常检测", "维护排程"]

if "page" not in st.session_state:
    st.session_state.page = "总览"
    st.session_state.nav_epoch = 0


def _on_nav_change():
    """用户手动点击分段控件时，同步当前页。"""
    st.session_state.page = st.session_state[f"nav_{st.session_state.nav_epoch}"]


def _goto(page_name: str):
    """程序化跳转：更新目标页并重建导航控件以同步高亮。"""
    st.session_state.page = page_name
    st.session_state.nav_epoch += 1
    st.rerun()


st.segmented_control(
    "", PAGES,
    key=f"nav_{st.session_state.nav_epoch}",
    default=st.session_state.page,
    on_change=_on_nav_change,
    label_visibility="collapsed",
)

page = st.session_state.page

# ===================== 总览 =====================
if page == "总览":
    c1, c2 = st.columns([3, 1])
    c1.markdown("**服务状态**")
    c1.markdown(status_badge, unsafe_allow_html=True)
    if c2.button("刷新状态", key="refresh_backend"):
        st.session_state.backend_ok = check_backend()
        st.rerun()

    st.markdown("### 功能模块")
    m1, m2, m3 = st.columns(3)
    with m1:
        with st.container(border=True):
            st.markdown("#### 故障预测")
            st.caption("随机森林 · 集成学习")
            st.write("输入设备运行特征，输出故障概率与故障类型。")
            if st.button("进入故障预测", key="goto_predict", use_container_width=True):
                _goto("故障预测")
    with m2:
        with st.container(border=True):
            st.markdown("#### 异常检测")
            st.caption("LSTM 自编码器 · 无监督学习")
            st.write("基于重构误差，识别设备早期退化异常。")
            if st.button("进入异常检测", key="goto_anomaly", use_container_width=True):
                _goto("异常检测")
    with m3:
        with st.container(border=True):
            st.markdown("#### 维护排程")
            st.caption("遗传算法 · 智能优化")
            st.write("对多台设备优化维护顺序，最小化停机损失。")
            if st.button("进入维护排程", key="goto_schedule", use_container_width=True):
                _goto("维护排程")

    with st.container(border=True):
        st.markdown("**使用流程**")
        st.markdown("1. 启动后端服务　→　2. 上传或输入数据　→　3. 点击对应操作　→　4. 查看结果")

# ===================== 故障预测 =====================
elif page == "故障预测":
    with st.container(border=True):
        st.markdown("**Excel 批量预测**")
        st.caption("表头列名：`air_temperature, process_temperature, rotational_speed, torque, tool_wear, type`")
        up = st.file_uploader("上传特征表", type=["xlsx"], key="predict_xlsx")
        if up is not None:
            recs, err = read_excel(up, FEATURE_COLS)
            if err:
                st.error(err)
            else:
                st.markdown("**导入数据预览**")
                st.dataframe(pd.DataFrame(recs).rename(columns=FEATURE_RENAME),
                             use_container_width=True, hide_index=True)
                rows = []
                for rec in recs:
                    r = requests.post(f"{API}/predict", json=rec)
                    if r.status_code != 200:
                        backend_error()
                        break
                    d = r.json()
                    rows.append({**rec, "故障概率": f"{d['failure_prob']:.1%}",
                                 "是否故障": "故障" if d["is_failure"] else "正常",
                                 "故障类型": d["failure_type"]})
                if rows:
                    st.dataframe(pd.DataFrame(rows).rename(columns=FEATURE_RENAME),
                                 use_container_width=True, hide_index=True)

    st.write("")

    with st.container(border=True):
        st.markdown("**单条预测**")
        c1, c2, c3 = st.columns(3)
        air = c1.number_input("空气温度 (K)", 250.0, 400.0, 300.0)
        proc = c1.number_input("工艺温度 (K)", 250.0, 400.0, 310.0)
        speed = c2.number_input("转速 (rpm)", 1000.0, 3000.0, 1500.0)
        torque = c2.number_input("扭矩 (Nm)", 0.0, 100.0, 40.0)
        wear = c3.number_input("刀具磨损 (min)", 0.0, 300.0, 100.0)
        type_ = c3.selectbox("产品类型", ["L", "M", "H"])

        if st.button("预测", type="primary", key="predict_btn"):
            payload = {"air_temperature": air, "process_temperature": proc,
                       "rotational_speed": speed, "torque": torque,
                       "tool_wear": wear, "type": type_}
            r = requests.post(f"{API}/predict", json=payload)
            if r.status_code == 200:
                d = r.json()
                is_fail = d["is_failure"]
                with st.container(border=True):
                    m1, m2, m3 = st.columns(3)
                    m1.metric("故障概率", f"{d['failure_prob']:.1%}")
                    m2.metric("判定结果", "故障" if is_fail else "正常")
                    m3.metric("故障类型", d["failure_type"])
                    st.markdown(
                        badge("⚠ 预测为故障", "danger") if is_fail
                        else badge("✓ 预测为正常", "success"),
                        unsafe_allow_html=True,
                    )
                    st.markdown("**各类别概率**")
                    probs = d["class_probs"]
                    df_p = (pd.DataFrame({"故障类型": list(probs.keys()),
                                          "概率": list(probs.values())})
                            .sort_values("概率", ascending=False).reset_index(drop=True))
                    df_p["概率"] = df_p["概率"].map(lambda x: f"{x:.2%}")
                    st.dataframe(df_p, use_container_width=True, hide_index=True)
            else:
                backend_error()

# ===================== 异常检测 =====================
elif page == "异常检测":
    with st.container(border=True):
        st.markdown("**Excel 导入序列**")
        st.caption("每行一个时间步（建议 ≥10 行），列：`air_temperature, process_temperature, rotational_speed, torque, tool_wear, type`")
        up = st.file_uploader("导入运行序列", type=["xlsx"], key="anomaly_xlsx")
        if up is not None:
            recs, err = read_excel(up, FEATURE_COLS)
            if err:
                st.error(err)
            else:
                st.markdown("**导入数据预览**")
                st.dataframe(pd.DataFrame(recs).rename(columns=FEATURE_RENAME),
                             use_container_width=True, hide_index=True)
                r = requests.post(f"{API}/anomaly", json={"sequence": recs})
                if r.status_code == 200:
                    d = r.json()
                    with st.container(border=True):
                        m1, m2 = st.columns(2)
                        m1.metric("重构误差", d["reconstruction_error"])
                        m2.metric("判定阈值", d["threshold"])
                        st.markdown(
                            badge("⚠ 异常", "danger") if d["is_anomaly"]
                            else badge("✓ 正常", "success"),
                            unsafe_allow_html=True,
                        )
                else:
                    backend_error()

    st.write("")

    with st.container(border=True):
        st.markdown("**手动粘贴序列**")
        st.caption("每行：空气温度,工艺温度,转速,扭矩,刀具磨损,类型")
        seq_text = st.text_area(
            "运行序列",
            "300,310,1500,40,100,L\n300,310,1500,40,101,L\n300,310,1500,40,102,L",
            height=120,
        )
        if st.button("检测", type="primary", key="anomaly_btn"):
            sequence, ok = [], True
            for ln in seq_text.strip().splitlines():
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) != 6:
                    st.error(f"格式错误：{ln}")
                    ok = False
                    break
                air, proc, speed, torque, wear, t = parts
                sequence.append({"air_temperature": float(air), "process_temperature": float(proc),
                                 "rotational_speed": float(speed), "torque": float(torque),
                                 "tool_wear": float(wear), "type": t})
            if ok:
                r = requests.post(f"{API}/anomaly", json={"sequence": sequence})
                if r.status_code == 200:
                    d = r.json()
                    with st.container(border=True):
                        m1, m2 = st.columns(2)
                        m1.metric("重构误差", d["reconstruction_error"])
                        m2.metric("判定阈值", d["threshold"])
                        st.markdown(
                            badge("⚠ 异常", "danger") if d["is_anomaly"]
                            else badge("✓ 正常", "success"),
                            unsafe_allow_html=True,
                        )
                else:
                    backend_error()

# ===================== 维护排程 =====================
elif page == "维护排程":
    with st.container(border=True):
        st.markdown("**Excel 导入设备清单**")
        st.caption("表头列名：`device_id, risk, downtime_cost, resource, maintenance_time`")
        up = st.file_uploader("上传设备清单", type=["xlsx"], key="schedule_xlsx")
        if up is not None:
            recs, err = read_excel(up, DEVICE_COLS)
            if err:
                st.error(err)
            else:
                st.markdown("**导入数据预览**")
                st.dataframe(pd.DataFrame(recs).rename(columns=DEVICE_RENAME),
                             use_container_width=True, hide_index=True)
                capacity_x = st.number_input("每日资源上限", 1.0, 100.0, 6.0, key="cap_excel")
                if st.button("求解", type="primary", key="schedule_excel_btn"):
                    r = requests.post(f"{API}/schedule",
                                      json={"devices": recs, "resource_capacity": capacity_x})
                    if r.status_code == 200:
                        d = r.json()
                        with st.container(border=True):
                            st.metric("最小总损失", d["total_loss"])
                            st.markdown("**维护排程甘特图**")
                            st.pyplot(draw_gantt(d["plan"]))
                            st.markdown("**维护排程计划**")
                            st.dataframe(pd.DataFrame(d["plan"]),
                                         use_container_width=True, hide_index=True)
                    else:
                        backend_error()

    st.write("")

    with st.container(border=True):
        st.markdown("**手动添加设备**")
        n = st.number_input("设备数量", 1, 20, 5)
        devices = []
        for i in range(int(n)):
            with st.expander(f"设备 {i + 1}"):
                c = st.columns(5)
                did = c[0].text_input("设备编号", f"dev-{i + 1}", key=f"id_{i}")
                risk = c[1].number_input("故障风险", 0.0, 1.0, 0.5, key=f"risk_{i}")
                cost = c[2].number_input("停机损失(元/时)", 0.0, 1000.0, 100.0, key=f"cost_{i}")
                res = c[3].number_input("资源(工时)", 0.0, 20.0, 3.0, key=f"res_{i}")
                mtime = c[4].number_input("维护耗时(时)", 0.0, 20.0, 2.0, key=f"mtime_{i}")
                devices.append({"device_id": did, "risk": risk, "downtime_cost": cost,
                                "resource": res, "maintenance_time": mtime})
        capacity = st.number_input("每日资源上限", 1.0, 100.0, 6.0, key="cap_manual")
        if st.button("求解", type="primary", key="schedule_btn"):
            r = requests.post(f"{API}/schedule",
                              json={"devices": devices, "resource_capacity": capacity})
            if r.status_code == 200:
                d = r.json()
                with st.container(border=True):
                    st.metric("最小总损失", d["total_loss"])
                    st.markdown("**维护排程甘特图**")
                    st.pyplot(draw_gantt(d["plan"]))
                    st.markdown("**维护排程计划**")
                    st.dataframe(pd.DataFrame(d["plan"]),
                                 use_container_width=True, hide_index=True)
            else:
                backend_error()
