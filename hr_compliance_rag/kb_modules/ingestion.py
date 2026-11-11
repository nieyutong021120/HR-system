"""
数据准备模块(法规语料) - 负责法规Markdown加载、按"条"结构化分块、父子映射与元数据

改自 C8/rag_modules/data_preparation.py，主要差异：
- 父文档 = 一部法规；子块 = 一条法条
- 元数据: title/law_type(效力级别)/domain/status(时效性)/article_no/source_url
- get_parent_documents 用预建字典做 O(1) 回溯(修复 C8 的 O(N^2) 查找)
"""

import logging
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class IngestionModule:
    """数据准备模块 - 法规加载 / 结构化分块 / 父子映射"""

    # 劳动用工子领域标签(供查询过滤复用，避免关键词重复定义)
    DOMAIN_LABELS = [
        '劳动合同', '社会保险', '工时休假', '工资福利', '劳动争议',
        '工伤职业病', '女职工未成年工保护', '劳务派遣', '就业促进',
        '退休养老', '工会集体合同', '综合'
    ]

    # 效力级别 -> 检索优先级(数字越大位阶越高，检索/排序时优先)
    LAW_TYPE_PRIORITY = {
        '宪法': 5,
        '法律': 4,
        '行政法规': 3,
        '监察法规': 3,
        '部门规章': 2,
        '司法解释': 2,
        '地方性法规': 1,
        '其他': 0,
    }

    def __init__(self, laws_md_path: str):
        """
        Args:
            laws_md_path: 清洗后的法规Markdown目录
        """
        self.laws_md_path = laws_md_path
        self.documents: List[Document] = []        # 父文档(整部法规)
        self.chunks: List[Document] = []           # 子块(单条法条)
        self.parent_child_map: Dict[str, str] = {} # child_id -> law_id
        # 预建 law_id -> 父文档 的索引，供 O(1) 回溯(修复 C8 的 O(N^2))
        self._parent_index: Dict[str, Document] = {}

    # ------------------------------------------------------------------ #
    # frontmatter 解析(不引入 pyyaml，保持依赖精简)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
        """
        解析形如:
            ---
            key: value
            ---
            正文...
        返回 (元数据字典, 正文)
        """
        meta: Dict[str, str] = {}
        body = text
        if text.startswith('---'):
            # 第二个 '---' 之后为正文
            end = text.find('\n---', 3)
            if end != -1:
                header = text[3:end].strip('\n')
                body = text[end + 4:].lstrip('\n')
                for line in header.splitlines():
                    if ':' in line:
                        k, v = line.split(':', 1)
                        meta[k.strip()] = v.strip()
        return meta, body

    @classmethod
    def get_supported_domains(cls) -> List[str]:
        return cls.DOMAIN_LABELS

    @classmethod
    def get_law_type_priority(cls, law_type: str) -> int:
        return cls.LAW_TYPE_PRIORITY.get(law_type, 0)

    # ------------------------------------------------------------------ #
    # 加载父文档
    # ------------------------------------------------------------------ #
    def load_documents(self) -> List[Document]:
        """加载 data/laws/*.md 作为父文档"""
        logger.info(f"正在从 {self.laws_md_path} 加载法规文档...")

        documents: List[Document] = []
        base = Path(self.laws_md_path)
        if not base.exists():
            raise FileNotFoundError(f"法规目录不存在: {self.laws_md_path}，请先运行 scripts/build_corpus.py")

        for md_file in sorted(base.rglob("*.md")):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    raw = f.read()

                meta, body = self._parse_frontmatter(raw)

                # 确定性 law_id：优先用 frontmatter 里的原始法规id，否则用相对路径MD5
                law_id = meta.get('law_id') or hashlib.md5(
                    md_file.resolve().relative_to(base.resolve()).as_posix().encode('utf-8')
                ).hexdigest()

                doc = Document(
                    page_content=body,
                    metadata={
                        "source": str(md_file),
                        "law_id": law_id,
                        "title": meta.get('title', md_file.stem),
                        "law_type": meta.get('law_type', '其他'),
                        "office": meta.get('office', ''),
                        "publish_date": meta.get('publish_date', ''),
                        "effective_date": meta.get('effective_date', ''),
                        "status": meta.get('status', '现行有效'),
                        "domain": meta.get('domain', '综合'),
                        "source_url": meta.get('source_url', ''),
                        "doc_type": "parent",
                    }
                )
                documents.append(doc)
            except Exception as e:
                logger.warning(f"读取法规 {md_file} 失败: {e}")

        self.documents = documents
        # 预建父文档索引，供 O(1) 回溯
        self._parent_index = {d.metadata["law_id"]: d for d in documents}
        logger.info(f"成功加载 {len(documents)} 部法规")
        return documents

    # ------------------------------------------------------------------ #
    # 按"条"结构化分块
    # ------------------------------------------------------------------ #
    def chunk_documents(self) -> List[Document]:
        """将每部法规按 章/条 结构切分为子块(每条一个块)"""
        logger.info("正在按法条结构分块...")
        if not self.documents:
            raise ValueError("请先加载文档")

        headers_to_split_on = [
            ("#", "法名"),
            ("##", "章"),
            ("###", "条"),
        ]
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,   # 保留标题，便于理解上下文
        )

        all_chunks: List[Document] = []
        for doc in self.documents:
            law_id = doc.metadata["law_id"]
            try:
                md_chunks = splitter.split_text(doc.page_content)
            except Exception as e:
                logger.warning(f"法规 {doc.metadata.get('title')} 分块失败: {e}，整篇作为单块")
                md_chunks = [Document(page_content=doc.page_content)]

            if len(md_chunks) <= 1:
                logger.warning(f"法规 {doc.metadata.get('title')} 未能按条切分，可能缺少标题结构")

            for i, chunk in enumerate(md_chunks):
                # 继承父文档元数据
                chunk.metadata.update(doc.metadata)
                # 提取条号(来自 ### 标题，如 "第十九条")
                article_no = self._extract_article_no(chunk)
                # 确定性 chunk_id：由 law_id+条号+序号 派生，保证持久化索引与内存分块的ID一致
                # (若用随机uuid，跨进程/跨次运行的ID不一致会导致RRF无法去重)
                child_id = hashlib.md5(f"{law_id}|{article_no}|{i}".encode('utf-8')).hexdigest()
                chunk.metadata.update({
                    "chunk_id": child_id,
                    "law_id": law_id,
                    "parent_id": law_id,       # 与 C8 保持字段名兼容
                    "doc_type": "child",
                    "chunk_index": i,
                    "article_no": article_no,
                    "chapter": chunk.metadata.get("章", ""),
                    "chunk_size": len(chunk.page_content),
                })
                self.parent_child_map[child_id] = law_id
                all_chunks.append(chunk)

        self.chunks = all_chunks
        logger.info(f"分块完成，共生成 {len(all_chunks)} 个法条块")
        return all_chunks

    @staticmethod
    def _extract_article_no(chunk: Document) -> str:
        """从块标题或正文首部提取"第X条"作为条号"""
        header = chunk.metadata.get("条", "")
        m = re.search(r'第[一二三四五六七八九十百千零〇\d]+条', header)
        if m:
            return m.group()
        # 回退：从正文首部找
        m = re.search(r'第[一二三四五六七八九十百千零〇\d]+条', chunk.page_content[:30])
        return m.group() if m else ""

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    def get_statistics(self) -> Dict[str, Any]:
        if not self.documents:
            return {}
        domains: Dict[str, int] = {}
        law_types: Dict[str, int] = {}
        for doc in self.documents:
            d = doc.metadata.get('domain', '综合')
            domains[d] = domains.get(d, 0) + 1
            t = doc.metadata.get('law_type', '其他')
            law_types[t] = law_types.get(t, 0) + 1
        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'domains': domains,
            'law_types': law_types,
            'avg_chunk_size': (sum(c.metadata.get('chunk_size', 0) for c in self.chunks) / len(self.chunks)) if self.chunks else 0,
        }

    # ------------------------------------------------------------------ #
    # 父文档回溯(O(1) 查表，修复 C8 的 O(N^2))
    # ------------------------------------------------------------------ #
    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据命中的法条子块回溯整部法规(去重，按命中条数排序)
        """
        parent_hits: Dict[str, int] = {}
        for chunk in child_chunks:
            law_id = chunk.metadata.get("parent_id") or chunk.metadata.get("law_id")
            if law_id:
                parent_hits[law_id] = parent_hits.get(law_id, 0) + 1

        # 命中条数多的法规排前面；同数时按效力级别高者优先
        sorted_ids = sorted(
            parent_hits.keys(),
            key=lambda x: (parent_hits[x], self.get_law_type_priority(
                self._parent_index[x].metadata.get("law_type", "其他") if x in self._parent_index else "其他")),
            reverse=True,
        )

        parent_docs = [self._parent_index[pid] for pid in sorted_ids if pid in self._parent_index]

        info = [f"{d.metadata.get('title')}({parent_hits.get(d.metadata.get('law_id'), 0)}条)" for d in parent_docs]
        logger.info(f"从 {len(child_chunks)} 个法条块回溯到 {len(parent_docs)} 部法规: {', '.join(info)}")
        return parent_docs
