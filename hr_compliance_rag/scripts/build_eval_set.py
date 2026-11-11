"""
HR合规RAG —— P1 smoke 评测集(20题)

三层评测：
  A. ground-truth 存在性：期望法条是否真实存在于语料(防止评测集本身写错)
  B. 检索召回：期望法条是否进入 top-k 混合检索结果
  C. 引用准确率：期望法条是否出现在生成答案的"依据条款"引用脚注中

用法：
  # 仅检索层(无需LLM，秒级)
  PYTHONIOENCODING=utf-8 python scripts/build_eval_set.py --retrieval-only
  # 全量(含LLM生成，约10分钟)
  PYTHONIOENCODING=utf-8 python scripts/build_eval_set.py
"""

import re
import sys
import time
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from main import HRComplianceSystem

# 每条 ground-truth 均已对照 data/laws 语料人工核验(2026-11-01)
# expect: (标题关键词, 法条号)；命中任一即算通过
EVAL_SET = [
    {"q": "试用期最长多久?", "role": "employee",
     "expect": [("劳动合同法", "第十九条"), ("劳动法", "第二十一条")]},
    {"q": "经济补偿金怎么计算?", "role": "employee",
     "expect": [("劳动合同法", "第四十七条")]},
    {"q": "年休假有几天?", "role": "employee",
     "expect": [("职工带薪年休假条例", "第三条")]},
    {"q": "工伤如何认定?", "role": "hr",
     "expect": [("工伤保险条例", "第十四条"), ("工伤保险条例", "第十七条"), ("工伤保险条例", "第二十条")]},
    {"q": "女职工产假多少天?", "role": "employee",
     "expect": [("女职工劳动保护特别规定", "第七条"), ("劳动法", "第六十二条")]},
    {"q": "加班费怎么算?", "role": "employee",
     "expect": [("劳动法", "第四十四条"), ("劳动合同法", "第三十一条")]},
    {"q": "公司不签劳动合同要付双倍工资吗?", "role": "employee",
     "expect": [("劳动合同法", "第八十二条")]},
    {"q": "员工辞职要提前多久通知?", "role": "employee",
     "expect": [("劳动合同法", "第三十七条")]},
    {"q": "竞业限制期限最长多久?", "role": "legal",
     "expect": [("劳动合同法", "第二十四条")]},
    {"q": "什么情况下应当订立无固定期限劳动合同?", "role": "hr",
     "expect": [("劳动合同法", "第十四条")]},
    {"q": "用人单位可以单方解除劳动合同的情形有哪些?", "role": "hr",
     "expect": [("劳动合同法", "第三十九条"), ("劳动合同法", "第四十条")]},
    {"q": "最低工资标准由谁规定?", "role": "employee",
     "expect": [("劳动法", "第四十八条")]},
    {"q": "每天工作时间不超过几小时?", "role": "employee",
     "expect": [("劳动法", "第三十六条"), ("国务院关于职工工作时间的规定", "第三条")]},
    {"q": "领取失业保险金需要什么条件?", "role": "employee",
     "expect": [("失业保险条例", "第十四条")]},
    {"q": "劳动争议申请仲裁的时效是多久?", "role": "legal",
     "expect": [("劳动争议调解仲裁法", "第二十七条")]},
    {"q": "用人单位必须为员工缴纳社保吗?", "role": "hr",
     "expect": [("社会保险法", "第五十八条"), ("社会保险法", "第六十条")]},
    {"q": "劳务派遣只能用在什么岗位?", "role": "hr",
     "expect": [("劳动合同法", "第六十六条")]},
    {"q": "用人单位单方解除劳动合同需要通知工会吗?", "role": "legal",
     "expect": [("劳动合同法", "第四十三条")]},
    {"q": "哪些情形下用人单位不得解除劳动合同?", "role": "hr",
     "expect": [("劳动合同法", "第四十二条"), ("劳动法", "第二十九条")]},
    {"q": "工伤认定申请要在多长时间内提出?", "role": "hr",
     "expect": [("工伤保险条例", "第十七条"), ("工伤保险条例", "第二十条")]},
]

# 越界问题(应被 guardrail 拦截，不检索)
OUT_OF_SCOPE = [
    "帮我写一首关于春天的诗",
    "公司股权激励的税务筹划怎么做?",
]

CITE_RE = re.compile(r"《(.+?)》\s*(第[一二三四五六七八九十百千零〇\d]+条)")


def build_chunk_index(system):
    """构建 (标题, 法条号) 集合，用于 ground-truth 存在性校验"""
    idx = set()
    for c in system.ingestion.chunks:
        idx.add((c.metadata.get("title", ""), c.metadata.get("article_no", "")))
    return idx


def expected_exists(expect, chunk_idx):
    for kw, art in expect:
        for title, article in chunk_idx:
            if kw in title and article == art:
                return True
    return False


def retrieval_hit(expect, docs):
    for kw, art in expect:
        for d in docs:
            if kw in d.metadata.get("title", "") and d.metadata.get("article_no", "") == art:
                return (kw, art)
    return None


def parse_citations(answer):
    return CITE_RE.findall(answer)


def citation_hit(expect, answer):
    cites = parse_citations(answer)
    for kw, art in expect:
        for title, article in cites:
            if kw in title and article == art:
                return (kw, art)
    return None


def ask_with_retry(system, q, role, max_tries=6, base_wait=20):
    """调用 ask_question，遇 429/限流则指数退避重试"""
    for attempt in range(max_tries):
        try:
            return system.ask_question(q, role=role, stream=False)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower() or "RPM" in msg:
                wait = base_wait * (attempt + 1)
                print(f"    [限流] 第{attempt+1}次退避 {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            raise
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true", help="仅评测检索层，跳过LLM生成")
    ap.add_argument("--throttle", type=int, default=20, help="每题之间的节流秒数(默认20，应对低RPM)")
    args = ap.parse_args()

    system = HRComplianceSystem()
    system.initialize_system()
    system.build_knowledge_base()
    chunk_idx = build_chunk_index(system)

    print("\n" + "=" * 72)
    print("P1 SMOKE 评测 —— 20 题 (ground-truth 存在性 / 检索召回 / 引用准确率)")
    print("=" * 72)

    gt_ok = ret_ok = cite_ok = cite_total = 0
    rows = []
    n = len(EVAL_SET)
    for i, item in enumerate(EVAL_SET, 1):
        q, role, expect = item["q"], item["role"], item["expect"]

        # A. ground-truth 存在性
        gt = expected_exists(expect, chunk_idx)
        gt_ok += int(gt)

        # B. 检索召回
        docs = system.retrieval.hybrid_search(q, top_k=system.config.top_k)
        rhit = retrieval_hit(expect, docs)
        ret_ok += int(rhit is not None)

        # C. 引用准确率(可选)
        chit = None
        if not args.retrieval_only:
            answer = ask_with_retry(system, q, role)
            if isinstance(answer, str):
                chit = citation_hit(expect, answer)
                cite_total += 1
                cite_ok += int(chit is not None)

        rows.append((i, q, gt, rhit, chit))
        status = "GT✓" if gt else "GT✗"
        rstatus = f"检索✓({rhit[1]})" if rhit else "检索✗"
        cstatus = "" if args.retrieval_only else (f" 引用✓({chit[1]})" if chit else " 引用✗")
        print(f"[{i:>2}] {status} {rstatus}{cstatus}  —— {q}", flush=True)

        if not args.retrieval_only and i < n:
            time.sleep(args.throttle)

    n = len(EVAL_SET)
    print("\n" + "-" * 72)
    print(f"ground-truth 存在性: {gt_ok}/{n}  ({gt_ok/n*100:.0f}%)")
    print(f"检索召回 (top-{system.config.top_k}):   {ret_ok}/{n}  ({ret_ok/n*100:.0f}%)")
    if not args.retrieval_only:
        print(f"引用准确率:         {cite_ok}/{cite_total}  ({cite_ok/max(cite_total,1)*100:.0f}%)")

    # 越界 guardrail 检查
    if not args.retrieval_only:
        print("\n" + "-" * 72)
        print("越界 guardrail 检查:")
        for j, q in enumerate(OUT_OF_SCOPE):
            if j > 0:
                time.sleep(args.throttle)
            route = system.dialogue.query_router(q)
            ok = route == "out_of_scope"
            print(f"  [{'拦截✓' if ok else '未拦截✗'}] route={route}  —— {q}", flush=True)

    print("=" * 72)


if __name__ == "__main__":
    main()
