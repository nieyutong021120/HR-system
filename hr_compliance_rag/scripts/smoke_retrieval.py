"""
检索冒烟测试 - 不依赖LLM，验证 语料->索引->混合检索->引用 链路
运行: python scripts/smoke_retrieval.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG
from kb_modules import IngestionModule, IndexingModule, RetrievalModule

TEST_QUERIES = [
    "试用期最长多久",
    "经济补偿金怎么计算",
    "年休假有几天",
    "工伤如何认定",
    "女职工产假多少天",
    "加班费怎么算",
]


def main():
    cfg = DEFAULT_CONFIG
    ing = IngestionModule(cfg.laws_md_path)
    ing.load_documents()
    chunks = ing.chunk_documents()

    idx = IndexingModule(cfg.embedding_model, cfg.index_save_path)
    vs = idx.load_index()
    if vs is None:
        vs = idx.build_vector_index(chunks)
        idx.save_index()

    ret = RetrievalModule(vs, chunks, retriever_k=cfg.retriever_k)

    print("\n" + "=" * 70)
    for q in TEST_QUERIES:
        hits = ret.hybrid_search(q, top_k=cfg.top_k)
        print(f"\nQ: {q}")
        if not hits:
            print("   (无命中)")
            continue
        for i, d in enumerate(hits, 1):
            m = d.metadata
            print(f"   {i}. 《{m.get('title')}》{m.get('article_no')} "
                  f"[{m.get('law_type')}/{m.get('domain')}] rrf={m.get('rrf_score', 0):.4f}")
        # 父法规回溯验证
        parents = ing.get_parent_documents(hits)
        print(f"   -> 回溯父法规: {[p.metadata.get('title') for p in parents]}")


if __name__ == "__main__":
    main()
