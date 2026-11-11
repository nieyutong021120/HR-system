"""
检索优化模块 - 混合检索(向量 + jieba-BM25) + RRF重排 + 效力级别优先 + 元数据过滤

改自 C8/rag_modules/retrieval_optimization.py，关键升级：
1. BM25 注入 jieba 中文分词(修复 C8 中文按字切分的召回短板)
2. RRF 去重键改用 chunk_id(修复 C8 用内容MD5可能碰撞的问题)
3. RRF 分数叠加"效力级别"加权(法律 > 行政法规 > 部门规章 > 司法解释)
4. 元数据过滤改为向量侧前置过滤(FAISS filter) + BM25侧后置过滤兜底
"""

import logging
import hashlib
from typing import List, Dict, Any, Callable

import jieba
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .ingestion import IngestionModule

logger = logging.getLogger(__name__)


def _jieba_tokenize(text: str) -> List[str]:
    """中文分词(BM25 预处理)"""
    return [w for w in jieba.cut(text) if w.strip()]


class RetrievalModule:
    """检索优化模块 - 混合检索与过滤"""

    def __init__(self, vectorstore: FAISS, chunks: List[Document], retriever_k: int = 10):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.retriever_k = retriever_k
        self.setup_retrievers()

    def setup_retrievers(self):
        logger.info("正在设置检索器...")
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.retriever_k}
        )
        # BM25 使用 jieba 分词，显著改善中文关键词召回
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=self.retriever_k,
            preprocess_func=_jieba_tokenize,
        )
        logger.info("检索器设置完成")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _match_filters(meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """判断一条元数据是否满足全部过滤条件"""
        for key, value in filters.items():
            if key not in meta:
                return False
            if isinstance(value, list):
                if meta[key] not in value:
                    return False
            elif meta[key] != value:
                return False
        return True

    def _vector_search(self, query: str, k: int, filters: Dict[str, Any] = None) -> List[Document]:
        """向量检索，可选前置元数据过滤"""
        if filters:
            predicate: Callable[[Dict[str, Any]], bool] = lambda meta: self._match_filters(meta, filters)
            return self.vectorstore.similarity_search(query, k=k, filter=predicate)
        return self.vector_retriever.invoke(query)

    # ------------------------------------------------------------------ #
    def hybrid_search(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> List[Document]:
        """
        混合检索：向量(可前置过滤) + BM25(后置过滤兜底) -> RRF融合 -> 效力级别加权 -> top_k
        """
        vector_docs = self._vector_search(query, self.retriever_k, filters)

        bm25_docs = self.bm25_retriever.invoke(query)
        if filters:
            bm25_docs = [d for d in bm25_docs if self._match_filters(d.metadata, filters)]

        reranked = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked[:top_k]

    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """带元数据过滤的检索(保留 C8 同名接口，内部走 hybrid_search)"""
        return self.hybrid_search(query, top_k=top_k, filters=filters)

    # ------------------------------------------------------------------ #
    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """
        RRF(Reciprocal Rank Fusion) 重排，并叠加效力级别加权。
        去重键优先使用 chunk_id(子块唯一)，回退到内容MD5。
        """
        doc_scores: Dict[str, float] = {}
        doc_objects: Dict[str, Document] = {}

        def _key(doc: Document) -> str:
            # 修复 C8：优先用 chunk_id，避免内容相同的块发生哈希碰撞
            cid = doc.metadata.get("chunk_id")
            if cid:
                return f"id::{cid}"
            return "md5::" + hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()

        def _accumulate(docs: List[Document], tag: str):
            for rank, doc in enumerate(docs):
                key = _key(doc)
                doc_objects[key] = doc
                rrf = 1.0 / (k + rank + 1)
                doc_scores[key] = doc_scores.get(key, 0.0) + rrf
                logger.debug(f"{tag} - rank{rank + 1}: RRF={rrf:.4f}")

        _accumulate(vector_docs, "向量检索")
        _accumulate(bm25_docs, "BM25检索")

        # 效力级别加权：final = rrf * (1 + 0.05 * priority)，高位的法规适度优先
        scored = []
        for key, rrf in doc_scores.items():
            law_type = doc_objects[key].metadata.get("law_type", "其他")
            priority = IngestionModule.get_law_type_priority(law_type)
            final = rrf * (1 + 0.05 * priority)
            doc_objects[key].metadata['rrf_score'] = final
            scored.append((final, key))

        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [doc_objects[key] for _, key in scored]

        logger.info(
            f"RRF重排完成: 向量{len(vector_docs)} + BM25{len(bm25_docs)} -> 合并{len(reranked)}个法条块"
        )
        return reranked
