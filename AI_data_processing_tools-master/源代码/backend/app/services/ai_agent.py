# python
# File: `backend/app/services/ai_agent.py`
from io import BytesIO
import base64
import sqlite3
import re
import os
import json
from typing import Optional, Dict, Any, List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

from langchain_openai import ChatOpenAI
from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

# 简单进程内缓存：以 db_path 为键，保存最近一次查询结果与摘要
RESULT_CACHE: Dict[str, Dict[str, Any]] = {}

API_KEY = DEEPSEEK_API_KEY or os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = DEEPSEEK_BASE_URL or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")


def _make_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        temperature=temperature,
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=API_KEY,
        base_url=BASE_URL,
    )


def _extract_schema(db_path: str) -> Dict[str, List[str]]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [r[0] for r in cur.fetchall()]
        schema: Dict[str, List[str]] = {}
        for t in tables:
            try:
                cur.execute(f"PRAGMA table_info('{t}')")
                cols = [r[1] for r in cur.fetchall()]
                schema[t] = cols
            except Exception:
                schema[t] = []
        return schema
    finally:
        conn.close()


def _run_sql(db_path: str, sql: str) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
        cols = [c[0] for c in cur.description] if cur.description else []
        return {"columns": cols, "rows": rows}
    finally:
        conn.close()


def _ensure_chinese_font():
    preferred = [
        "Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "SimSun",
        "WenQuanYi Zen Hei", "Arial Unicode MS", "Noto Sans CJK JP"
    ]
    try:
        # 尝试在系统字体列表中找到首选字体
        found = None
        for f in font_manager.fontManager.ttflist:
            name = getattr(f, "name", None)
            if name in preferred:
                found = f.fname
                break
        if found:
            font_manager.fontManager.addfont(found)
            rcParams['font.family'] = font_manager.FontProperties(fname=found).get_name()
        else:
            # fallback to default sans-serif
            rcParams['font.family'] = ['sans-serif']
    except Exception:
        rcParams['font.family'] = ['sans-serif']
    rcParams['axes.unicode_minus'] = False


def _plan_with_llm(llm: ChatOpenAI, user_question: str, schema_text: str, recent_summary: str = "") -> Dict[str, Any]:
    """
    向 LLM 请求「计划决策」，严格返回一个 JSON 对象，结构示例：
    {
      "action":"query"|"use_history",
      "sql":"SELECT ...",
      "chart": {
         "needed": true|false,
         "type": "bar"|"line"|"pie"|"auto",
         "x": "列名或空",
         "y": "列名或空",
         "title": "建议的短标题或空"
      },
      "reason":"..."
    }
    """
    prompt = (
        "你是智能决策模块。基于用户问题、数据库结构和最近缓存的数据摘要，判断是否需要访问数据库，并决定是否生成图表及图表细节。\n"
        "严格返回一个 JSON 对象，不要额外解释。必须包含字段：action, sql, chart, reason。\n"
        "chart 应该是一个对象，包含: needed (bool), type ('bar'|'line'|'pie'|'auto'), x (横轴列名或空), y (纵轴列名或空), title (建议标题或空)。\n\n"
        f"用户问题: {user_question}\n数据库结构:\n{schema_text}\n最近缓存摘要:\n{recent_summary}\n\n"
        "规则：\n"
        " - 若问题可以仅凭历史缓存回答或分析，返回 action=use_history 并把 sql 置空；\n"
        " - 若需要新数据或更精确结果，返回 action=query 并尽可能提供可执行的 SQLite SQL；\n"
        " - chart.needed 表示是否建议生成图表；若不建议，请设置为 false；\n"
        " - 只返回 JSON，不要任何额外文本。"
    )
    try:
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", str(resp))
        parsed = {}
        try:
            parsed = json.loads(text)
        except Exception:
            m = re.search(r"(\{[\s\S]*\})", text)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except Exception:
                    parsed = {}
        if isinstance(parsed, dict):
            # 规范化 chart 字段为 dict（兼容旧格式）
            chart = parsed.get("chart", {})
            if isinstance(chart, (str, list, bool)) or chart is None:
                parsed["chart"] = {"needed": True, "type": chart if isinstance(chart, str) else "auto", "x": "", "y": "", "title": ""}
            else:
                # 填充缺失键
                parsed["chart"] = {
                    "needed": bool(chart.get("needed", True)),
                    "type": chart.get("type", "auto") if isinstance(chart.get("type", "auto"), str) else "auto",
                    "x": chart.get("x", "") or "",
                    "y": chart.get("y", "") or "",
                    "title": chart.get("title", "") or ""
                }
            # 保证 action/sql/reason 有值
            parsed["action"] = parsed.get("action", "query")
            parsed["sql"] = parsed.get("sql", "") or ""
            parsed["reason"] = parsed.get("reason", "") or ""
            return parsed
    except Exception:
        pass
    return {"action": "query", "sql": "", "chart": {"needed": True, "type": "auto", "x": "", "y": "", "title": ""}, "reason": "默认回退为查询/生成图表"}


def _get_friendly_labels(llm: ChatOpenAI, x_col: str, y_col: str, user_question: str) -> Dict[str, str]:
    prompt = (
        "请将以下数据库字段名映射为中文可读标签，严格返回 JSON 对象，仅包含需要的键："
        "例如 {\"x\":\"月份\",\"y\":\"销售额(元)\"}，不要额外解释。\n\n"
        f"用户问题: {user_question}\n字段: x={x_col}, y={y_col}\n\n"
        "如果无法判断，可返回空字符串作为值。"
    )
    try:
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", str(resp))
        parsed = {}
        try:
            parsed = json.loads(text)
        except Exception:
            m = re.search(r"(\{[\s\S]*\})", text)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except Exception:
                    parsed = {}
        if isinstance(parsed, dict):
            return {"x": parsed.get("x", "") or "", "y": parsed.get("y", "") or ""}
    except Exception:
        return {}
    return {}


def _make_chart_title(user_question: str, plan_title: Optional[str] = None, labels: Optional[Dict[str, str]] = None, x_col: Optional[str] = None, y_col: Optional[str] = None, max_len: int = 60) -> str:
    """
    生成简短的图表标题：优先使用 plan_title，再使用 labels/y按x，否则从 user_question 摘要。
    """
    try:
        if plan_title and plan_title.strip():
            t = plan_title.strip()
            return t if len(t) <= max_len else t[:max_len].rstrip() + "…"
        if labels and labels.get("y") and labels.get("x"):
            return f"{labels.get('y')} 按 {labels.get('x')}"
        if y_col and x_col:
            return f"{y_col} 按 {x_col}"
        if user_question:
            sep = re.split(r"[。；;,\n，]", user_question.strip())
            first = sep[0].strip() if sep else user_question.strip()
            first = re.sub(r"['\"`].*?['\"`]", "", first)
            first = re.sub(r"\s+", " ", first).strip()
            if not first:
                return "数据可视化"
            if len(first) <= max_len:
                return first
            return first[:max_len].rstrip() + "…"
    except Exception:
        pass
    return "数据可视化"


def _generate_chart_from_df(df: pd.DataFrame, title: str = "", labels: Optional[Dict[str, str]] = None, chart_type: str = "auto") -> Optional[Dict[str, str]]:
    """
    支持 chart_type: bar|line|pie|auto
    若 auto，则根据行数与数值列简单判定。
    返回: {"image": "data:image/png;base64,...", "caption": "...", "type": "bar"}
    """
    if df.empty:
        return None
    numeric = df.select_dtypes(include=["number"])
    plt.switch_backend("Agg")
    _ensure_chinese_font()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    try:
        x_col = df.columns[0]
        # 判断真正使用的 y 列
        y_col = None
        if numeric.shape[1] >= 1:
            # 先按常用字段优先选择
            for cand in ["sales_amount", "transaction_amount", "amount", "value", "total"]:
                if cand in df.columns:
                    y_col = cand
                    break
            if not y_col:
                # 选第一个数值列
                y_col = numeric.columns[0]
        # 决定最终图类型
        final_type = chart_type or "auto"
        if final_type == "auto":
            if y_col is None:
                final_type = "bar"
            else:
                # 若行数小且存在 y 列，选 pie 或 bar
                if df.shape[0] <= 10:
                    final_type = "pie" if y_col else "bar"
                else:
                    final_type = "line" if df.shape[0] > 20 and df[x_col].dtype.kind in 'iouf' else "bar"
        # 绘图分支
        if final_type == "bar" and y_col:
            try:
                plot_df = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(20)
                ax.bar(plot_df.index.astype(str), plot_df.values, color="#3b82f6")
            except Exception:
                ax.text(0.5, 0.5, "无法绘制柱状图", ha='center', va='center')
        elif final_type == "line" and y_col:
            try:
                ax.plot(df[x_col].astype(str), df[y_col], marker='o', color="#06b6d4")
            except Exception:
                ax.text(0.5, 0.5, "无法绘制折线图", ha='center', va='center')
        elif final_type == "pie" and y_col:
            try:
                plot_df = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(10)
                ax.pie(plot_df.values, labels=plot_df.index.astype(str), autopct='%1.1f%%', textprops={'fontsize': 9})
            except Exception:
                ax.text(0.5, 0.5, "无法绘制饼图", ha='center', va='center')
        else:
            # 兜底：展示表格形式的第一列样例
            ax.axis('off')
            ax.text(0.01, 0.99, df.head(10).to_string(), fontsize=8, va='top', family=rcParams.get('font.family', ['sans-serif'])[0])

        xlabel = (labels.get("x") if labels and labels.get("x") else str(x_col))
        ylabel = (labels.get("y") if labels and labels.get("y") else (str(y_col) if y_col else "值"))
        if final_type != "pie":
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        ax.set_title(title or f"{ylabel} 按 {xlabel} 的 {final_type} 图")
        if final_type != "pie":
            plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")

        caption_parts = []
        if labels:
            if labels.get("x"):
                caption_parts.append(f"横轴：{labels.get('x')}")
            if labels.get("y"):
                caption_parts.append(f"纵轴：{labels.get('y')}")
        if not caption_parts:
            if y_col:
                caption_parts.append(f"纵轴：{y_col}")
            caption_parts.append(f"横轴：{x_col}")
        caption = f"图：{title or '数据可视化'}（类型：{final_type}）。 " + "；".join(caption_parts)
        return {"image": f"data:image/png;base64,{b64}", "caption": caption, "type": final_type}
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
        return None


def _extract_sql_from_text(text: str) -> Optional[str]:
    m = re.search(r"(?is)(select\b.*?;)", text)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(?is)(select\b.*)", text)
    if m2:
        return m2.group(1).strip()
    return None


def run_nl_query(db_path: str, user_question: str) -> Dict[str, Any]:
    """
    流程说明见模块注释。
    """
    schema = _extract_schema(db_path)
    schema_lines = [f"{t}: {', '.join(cols)}" for t, cols in schema.items()]
    schema_text = "\n".join(schema_lines) if schema_lines else "无可用表"

    llm = _make_llm(temperature=0.0)
    recent = RESULT_CACHE.get(db_path)
    recent_summary = recent.get("summary", "") if recent else ""

    # 1. 请求 LLM 做出计划/决策
    plan = _plan_with_llm(llm, user_question, schema_text, recent_summary)
    action = plan.get("action", "query")
    chart_plan = plan.get("chart", {"needed": True, "type": "auto", "x": "", "y": "", "title": ""})

    # 如果模型建议使用历史缓存
    if action == "use_history" and recent:
        # 直接根据缓存摘要响应，并在需要时基于缓存数据生成图表
        answer = recent.get("summary", "使用历史缓存回答。")
        chart = None
        try:
            cached_df = recent.get("df")
            if chart_plan.get("needed", True) and isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
                x_col = chart_plan.get("x") or (cached_df.columns[0] if cached_df.shape[1] >= 1 else "")
                numeric_cols = cached_df.select_dtypes(include=["number"]).columns.tolist()
                y_col = chart_plan.get("y") or (numeric_cols[0] if numeric_cols else "")
                labels = {}
                try:
                    if x_col and y_col:
                        labels = _get_friendly_labels(llm, x_col, y_col, user_question) or {}
                except Exception:
                    labels = {}
                chart_title = _make_chart_title(user_question, plan_title=chart_plan.get("title", ""), labels=labels, x_col=x_col, y_col=y_col)
                chart = _generate_chart_from_df(cached_df.head(100), title=chart_title, labels=labels, chart_type=chart_plan.get("type", "auto"))
        except Exception:
            chart = None
        return {"answer": answer, "sql": "", "query_result": {"columns": recent.get("columns", []), "rows": recent.get("rows", []), "chart": chart}}

    # 否则按需查询数据库（先尝试使用 plan 中给出的 SQL）
    sql = plan.get("sql", "").strip()
    if not sql:
        # 请求 LLM 生成 SQL（保留原流程的保底 prompt）
        prompt = (
            "请把下面的用户问题翻译为可在 SQLite 上执行的 SQL（只返回 SQL 或包裹在代码块的 SQL):\n\n"
            f"用户问题: {user_question}\n数据库结构:\n{schema_text}\n\n"
            "请尽可能给出明确可运行的 SELECT 语句。"
        )
        try:
            resp = llm.invoke(prompt)
            text = getattr(resp, "content", str(resp))
        except Exception as e:
            return {"error": f"调用 LLM 生成 SQL 失败: {e}", "sql": "", "query_result": {"columns": [], "rows": [], "chart": None}}
        sql = _extract_sql_from_text(text) or text.strip()

    if not re.search(r"(?i)select\b", sql):
        return {"answer": sql or "LLM 未返回可执行 SQL", "sql": "", "query_result": {"columns": [], "rows": [], "chart": None}}

    try:
        res = _run_sql(db_path, sql)
        df = pd.DataFrame(res["rows"], columns=res["columns"]) if res["rows"] else pd.DataFrame(columns=res["columns"])
        # 更新缓存（存储前 500 行用于快速响应）
        try:
            RESULT_CACHE[db_path] = {
                "df": df.head(500),
                "columns": res.get("columns", []),
                "rows": res.get("rows", [])[:500],
                "summary": f"最近一次查询返回 {len(df)} 条记录。"
            }
        except Exception:
            pass
    except Exception as e:
        return {"error": f"执行 SQL 出错: {e}", "sql": sql}

    # 生成 chart（若有数据并且模型建议）
    chart = None
    try:
        if not df.empty and chart_plan.get("needed", True):
            x_col = chart_plan.get("x") or (df.columns[0] if df.shape[1] >= 1 else "")
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            y_col = chart_plan.get("y") or (numeric_cols[0] if numeric_cols else "")
            labels = {}
            try:
                if x_col and y_col:
                    labels = _get_friendly_labels(llm, x_col, y_col, user_question) or {}
            except Exception:
                labels = {}
            chart_title = _make_chart_title(user_question, plan_title=chart_plan.get("title", ""), labels=labels, x_col=x_col, y_col=y_col)
            chart = _generate_chart_from_df(df.head(100), title=chart_title, labels=labels, chart_type=chart_plan.get("type", "auto"))
    except Exception:
        chart = None

    # 简短摘要返回
    row_count = len(df)
    summary_parts = []
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        prefer = None
        for cand in ["sales_amount", "transaction_amount", "amount", "value", "total"]:
            if cand in df.columns:
                prefer = cand
                break
        if not prefer:
            prefer = numeric_cols[0]
        total = df[prefer].sum()
        try:
            top = df.groupby(df.columns[0])[prefer].sum().sort_values(ascending=False).head(3)
            top_list = [f"{idx}: {val}" for idx, val in top.items()]
        except Exception:
            top_list = []
        summary_parts.append(f"共返回 {row_count} 条记录；字段 `{prefer}` 总和为 {total}。")
        if top_list:
            summary_parts.append("按横轴分组的前几项： " + "; ".join(top_list))
    else:
        summary_parts.append(f"共返回 {row_count} 条记录（无数值列可用于汇总）。")

    try:
        first_vals = df.iloc[:, 0].astype(str)
        min_v, max_v = first_vals.min(), first_vals.max()
        summary_parts.append(f"横轴示例范围：从 {min_v} 到 {max_v}。")
    except Exception:
        pass

    answer = "已执行 SQL 并返回数据。\n" + "\n".join(summary_parts) + "\n\n如果需要，我可以：\n- 生成并解释图表；\n- 基于当前结果生成决策要点；\n- 修改 SQL（例如改变时间窗口或过滤条件）。"

    return {"answer": answer, "sql": sql, "query_result": {"columns": res["columns"], "rows": res["rows"], "chart": chart}}


def generate_report(db_path: str, prompt: str) -> Dict[str, Any]:
    """
    基于数据库与用户提示生成中文决策报告，返回 report 文本与可选 chart。
    返回: {"report": "...", "chart": {"image": "...", "caption": "..."}}
    """
    if not prompt:
        prompt = "请基于当前数据生成一份简短的决策分析报告，包含关键发现与建议（中文）。"

    schema = _extract_schema(db_path)
    schema_text = "\n".join([f"{t}: {', '.join(cols)}" for t, cols in schema.items()]) or "无表"

    llm = _make_llm(temperature=0.0)

    probe_prompt = (
        "你是数据分析师。数据库结构:\n" + schema_text
        + "\n请基于用户目标提供最多 2 条用于分析的 SQLite SQL（只给 SQL），用于获取支持结论的数据。"
        + "\n用户目标：" + prompt
    )
    try:
        probe_resp = llm.invoke(probe_prompt)
        probe_text = getattr(probe_resp, "content", str(probe_resp))
    except Exception as e:
        return {"report": f"调用 LLM 失败: {e}"}

    sql = _extract_sql_from_text(probe_text) or ""
    data_sample = {"columns": [], "rows": []}
    chart = None
    if sql:
        try:
            data_sample = _run_sql(db_path, sql)
            df = pd.DataFrame(data_sample["rows"], columns=data_sample["columns"]) if data_sample["rows"] else pd.DataFrame(columns=data_sample["columns"])
            if not df.empty:
                # 让模型决定是否需要图表（简单请求）
                try:
                    chart_plan = _plan_with_llm(llm, prompt, schema_text, "")
                    cp = chart_plan.get("chart", {"needed": True, "type": "auto", "x": "", "y": "", "title": ""})
                except Exception:
                    cp = {"needed": True, "type": "auto", "x": "", "y": "", "title": ""}
                if cp.get("needed", True):
                    x_col = cp.get("x") or (df.columns[0] if df.shape[1] >= 1 else "")
                    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                    y_col = cp.get("y") or (numeric_cols[0] if numeric_cols else "")
                    labels = {}
                    try:
                        if x_col and y_col:
                            labels = _get_friendly_labels(llm, x_col, y_col, prompt) or {}
                    except Exception:
                        labels = {}
                    chart_title = _make_chart_title(prompt, plan_title=cp.get("title", ""), labels=labels, x_col=x_col, y_col=y_col)
                    chart = _generate_chart_from_df(df.head(100), title=chart_title, labels=labels, chart_type=cp.get("type", "auto"))
        except Exception as e:
            data_sample = {"error": f"执行 SQL 出错: {e}"}

    sample_text = f"数据样本列: {data_sample.get('columns', [])}\n前几行: {data_sample.get('rows', [])[:5]}"
    final_prompt = (
        "请用中文写一份决策分析报告，基于以下信息（条理清晰）：\n"
        f"用户目标: {prompt}\n"
        f"数据库结构: {schema_text}\n"
        f"数据样本: {sample_text}\n"
        "请包含：关键发现、可能的原因、建议的下一步操作（简洁）。"
    )
    try:
        final_resp = llm.invoke(final_prompt)
        final_text = getattr(final_resp, "content", str(final_resp))
    except Exception as e:
        final_text = f"生成报告出错: {e}"

    result = {"report": final_text}
    if chart:
        result["chart"] = chart
    return result
