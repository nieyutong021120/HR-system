"""
语料构建脚本 - 从真实法规JSON中筛选劳动用工相关法规，解析"编/章/节/条"，
输出带 frontmatter 的结构化 Markdown 到 data/laws/

用法:
    python scripts/build_corpus.py            # 构建语料
    python scripts/build_corpus.py --inspect  # 只打印首条匹配法规的原始结构，用于核对字段/正文格式

数据源: twang2218/law-datasets (国家法律法规数据库 flk.npc.gov.cn)
"""

import re
import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_JSON = PROJECT_ROOT / "data" / "raw" / "laws.json"
OUT_DIR = PROJECT_ROOT / "data" / "laws"

# 只保留现行有效(数据中实际取值为 '有效'/'尚未生效'；'已修改'/'已废止' 排除)
VALID_STATUS = {"有效", "现行有效", "尚未生效"}
# 效力级别白名单
VALID_TYPES = {"法律", "行政法规", "部门规章", "司法解释", "监察法规"}

# 标题命中即视为劳动用工相关(子串匹配)
TITLE_KEYWORDS = [
    "劳动法", "劳动合同法", "就业促进", "就业", "社会保险", "社保", "工伤保险", "工伤",
    "失业保险", "生育保险", "医疗保险", "医疗保障", "养老保险", "年休假", "带薪年休假",
    "女职工", "未成年工", "妇女权益", "工资", "最低工资", "工时", "工作时间", "休息休假",
    "探亲假", "婚假", "产假", "劳动争议", "人事争议", "职业病", "劳务派遣", "集体合同",
    "工会", "退休", "职业培训", "职业技能", "劳动保障监察", "人事争议", "事业单位人事",
    "职工福利", "雇佣", "聘用",
]

# 核心法规(保证优先入选)
CORE_TITLES = [
    "中华人民共和国劳动法",
    "中华人民共和国劳动合同法",
    "中华人民共和国劳动合同法实施条例",
    "中华人民共和国社会保险法",
    "中华人民共和国就业促进法",
    "中华人民共和国劳动争议调解仲裁法",
    "工伤保险条例",
    "失业保险条例",
    "女职工劳动保护特别规定",
    "职工带薪年休假条例",
    "中华人民共和国职业病防治法",
    "劳务派遣暂行规定",
    "工资支付暂行规定",
    "中华人民共和国工会法",
]

# 子领域归类：以"标题"为主信号(整部法规的主题由标题决定)，按序匹配，标题无命中再回退正文
DOMAIN_RULES: Dict[str, List[str]] = {
    "工伤职业病": ["工伤", "职业病", "因工"],
    "女职工未成年工保护": ["女职工", "未成年工", "妇女权益", "产假", "哺乳"],
    "劳务派遣": ["劳务派遣"],
    "劳动争议": ["劳动争议", "人事争议", "劳动监察", "劳动保障监察"],
    "退休养老": ["退休", "离休"],
    "工时休假": ["工时", "工作时间", "休息休假", "年休假", "休假", "加班", "探亲假", "婚假"],
    "工资福利": ["工资", "薪酬", "福利", "津贴", "报酬"],
    "工会集体合同": ["工会", "集体合同", "职工代表大会"],
    "就业促进": ["就业", "职业培训", "职业技能", "招聘", "职业介绍"],
    "社会保险": ["社会保险", "社保", "养老", "医疗", "失业", "生育", "住房公积金"],
    "劳动合同": ["劳动合同", "劳动法", "试用期", "聘用", "雇佣", "解除", "终止"],
}

MAX_LAWS = 40   # 语料规模上限(核心优先，其余按效力级别补足)

CN_NUM = "一二三四五六七八九十百千零〇\\d"
# 行首锚定：避免把条文内部的交叉引用(如"依照本法第四十七条")误判为新条文/新章
RE_ARTICLE = re.compile(rf"(?m)^[ 　]*(第[{CN_NUM}]+条)")
RE_CHAPTER = re.compile(rf"(?m)^[ 　]*(第[{CN_NUM}]+[编章节])[ 　]*([^\n第]{{0,40}})")


def load_laws() -> List[Dict[str, Any]]:
    if not RAW_JSON.exists():
        raise FileNotFoundError(f"未找到 {RAW_JSON}，请先运行 scripts/download_data.py")
    print(f"正在加载 {RAW_JSON} ...")
    with open(RAW_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # 兼容 {"data": [...]} 之类包裹结构
        for k in ("data", "laws", "result", "items"):
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
    print(f"共加载 {len(data)} 条记录")
    return data


def get_content(entry: Dict[str, Any]) -> str:
    """稳健提取全文字段"""
    for key in ("content", "body", "text", "full_text", "detail", "content_text"):
        v = entry.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, list):
            # 有些实现把正文按段落存成 list
            return "\n".join(str(x) for x in v)
    return ""


def title_match(title: str) -> bool:
    return any(kw in title for kw in TITLE_KEYWORDS)


def classify_domain(title: str, content: str) -> str:
    # 标题优先：整部法规的主题由标题决定，避免正文噪声(如总则里的"工作时间")误判
    for domain, kws in DOMAIN_RULES.items():
        if any(kw in title for kw in kws):
            return domain
    # 标题无命中再回退正文前段
    head = content[:300]
    for domain, kws in DOMAIN_RULES.items():
        if any(kw in head for kw in kws):
            return domain
    return "综合"


def normalize_title(title: str) -> str:
    """去掉括号里的修正/修订年份，用于同名归并"""
    return re.sub(r"[（(].*?[)）]", "", title).strip()


def parse_structure(content: str) -> List[Tuple[str, str, str]]:
    """
    把法规正文解析为 [(章标题, 条号, 条文内容), ...]
    对无章节结构的法规，章标题为空。
    """
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    parts = RE_ARTICLE.split(content)
    # parts = [前言, '第一条', 正文1, '第二条', 正文2, ...]
    preamble = parts[0] if parts else ""

    def last_chapter(text: str) -> str:
        matches = RE_CHAPTER.findall(text)
        if not matches:
            return ""
        marker, name = matches[-1]
        return f"{marker} {name.strip()}".strip()

    current_chapter = last_chapter(preamble)
    articles: List[Tuple[str, str, str]] = []
    i = 1
    while i < len(parts):
        art_no = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        # body 末尾可能带"下一章"的标题(属于下一条文之前)，需剥离
        ch_matches = list(RE_CHAPTER.finditer(body))
        if ch_matches:
            last = ch_matches[-1]
            art_content = body[:last.start()].strip()
            current_chapter = f"{last.group(1)} {last.group(2).strip()}".strip()
        else:
            art_content = body.strip()
        if art_content:
            articles.append((current_chapter, art_no, art_content))
        i += 2
    return articles


def to_markdown(entry: Dict[str, Any], domain: str) -> Optional[str]:
    title = (entry.get("title") or "").strip()
    content = get_content(entry)
    if not title or not content:
        return None

    law_type = entry.get("type") or "其他"
    office = entry.get("office") or ""
    publish = (entry.get("publish") or "")[:10]
    effective = (entry.get("effective") or entry.get("expiry") or "")[:10]
    status_raw = str(entry.get("status") or "")
    status = "现行有效" if status_raw in VALID_STATUS else status_raw
    url = entry.get("url") or ""
    law_id = entry.get("id") or ""

    articles = parse_structure(content)
    if not articles:
        return None

    fm = [
        "---",
        f"title: {title}",
        f"law_id: {law_id}",
        f"law_type: {law_type}",
        f"office: {office}",
        f"publish_date: {publish}",
        f"effective_date: {effective}",
        f"status: {status}",
        f"domain: {domain}",
        f"source_url: {url}",
        "---",
    ]
    body = [f"# {title}", ""]
    last_chapter = None
    for chapter, art_no, art_content in articles:
        if chapter and chapter != last_chapter:
            body.append(f"## {chapter}")
            body.append("")
            last_chapter = chapter
        body.append(f"### {art_no}")
        body.append(art_content)
        body.append("")
    return "\n".join(fm) + "\n" + "\n".join(body)


def safe_filename(title: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", title)[:80]


def select_laws(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """筛选：现行有效 + 效力级别白名单 + 标题命中；同名归并取最新"""
    candidates: Dict[str, Dict[str, Any]] = {}
    for entry in data:
        title = (entry.get("title") or "").strip()
        if not title or not title_match(title):
            continue
        law_type = entry.get("type") or ""
        if law_type not in VALID_TYPES:
            continue
        status_raw = str(entry.get("status") or "")
        if status_raw not in VALID_STATUS:
            continue
        if not get_content(entry):
            continue

        norm = normalize_title(title)
        old = candidates.get(norm)
        # 同名取发布日期更新的
        if old is None or (entry.get("publish") or "") > (old.get("publish") or ""):
            candidates[norm] = entry

    laws = list(candidates.values())

    # 排序：核心法规优先 -> 效力级别 -> 标题
    type_priority = {"法律": 4, "行政法规": 3, "监察法规": 3, "部门规章": 2, "司法解释": 2}

    def sort_key(e: Dict[str, Any]):
        title = e.get("title", "")
        is_core = 0 if any(c in title for c in CORE_TITLES) else 1
        return (is_core, -type_priority.get(e.get("type", ""), 0), title)

    laws.sort(key=sort_key)
    return laws[:MAX_LAWS]


def inspect(data: List[Dict[str, Any]]):
    for entry in data:
        title = entry.get("title", "")
        if "劳动合同法" in title and get_content(entry):
            print("=== 匹配到样例法规 ===")
            print("字段keys:", list(entry.keys()))
            for k, v in entry.items():
                if k in ("content", "body", "text"):
                    print(f"\n--- {k} (前1200字) ---\n{str(v)[:1200]}")
                else:
                    print(f"{k}: {v}")
            arts = parse_structure(get_content(entry))
            print(f"\n=== 解析出 {len(arts)} 条 ===")
            for ch, no, txt in arts[:5]:
                print(f"[{ch}] {no}: {txt[:60]}...")
            return
    print("未找到劳动合同法拉取样例；打印首条记录字段：")
    if data:
        print("keys:", list(data[0].keys()))
        print(json.dumps({k: (str(v)[:200]) for k, v in data[0].items()}, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="只核对数据结构，不构建语料")
    args = ap.parse_args()

    data = load_laws()

    if args.inspect:
        inspect(data)
        return

    laws = select_laws(data)
    print(f"筛选出 {len(laws)} 部劳动用工相关法规")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for entry in laws:
        title = entry.get("title", "")
        domain = classify_domain(title, get_content(entry))
        md = to_markdown(entry, domain)
        if not md:
            print(f"  跳过(无正文或无法解析条): {title}")
            continue
        out = OUT_DIR / f"{safe_filename(normalize_title(title))}.md"
        with open(out, "w", encoding="utf-8") as f:
            f.write(md)
        written += 1

    print(f"✅ 已写入 {written} 部法规到 {OUT_DIR}")


if __name__ == "__main__":
    main()
