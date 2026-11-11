"""
生成集成模块 - LLM合规问答生成 + 法条引用溯源

改自 C8/generation_integration.py，关键差异：
- 生成直接基于检索到的"法条子块"(而非整部法规)，因为法律问题需精确到条，且整部法规过长会撑爆上下文
- 上下文每条带 [《法规名》第X条 | 效力级别 | 时效性] 标注
- 回答末尾由代码"确定性"拼接依据条款(标题+条号+官方链接)与免责声明，不依赖LLM自觉
- 支持角色视角(员工/HR/法务)
"""

import os
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

ROLE_DESC = {
    "employee": "普通员工(需要通俗解释、关注自身权益)",
    "hr": "HR从业者(需要准确的执行口径、合规要点)",
    "legal": "法务(需要严谨的法条适用与边界分析)",
}

DISCLAIMER = "\n\n---\n⚠️ 本回答基于国家法律法规数据库检索生成，仅供参考，具体适用请以官方正式文本及专业法律意见为准。"


class GenerationModule:
    """生成集成模块 - 合规问答与引用溯源"""

    def __init__(self, model_name: str = "kimi-k2-0711-preview", temperature: float = 0.2, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()

    def setup_llm(self):
        logger.info(f"正在初始化生成LLM: {self.model_name}")
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")
        self.llm = MoonshotChat(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=5,
            moonshot_api_key=api_key,
        )
        logger.info("生成LLM初始化完成")

    # ------------------------------------------------------------------ #
    # 上下文与引用构建
    # ------------------------------------------------------------------ #
    def _build_context(self, chunks: List[Document], max_length: int = 3000) -> str:
        if not chunks:
            return "暂无相关法规信息。"
        parts, cur = [], 0
        for i, doc in enumerate(chunks, 1):
            m = doc.metadata
            header = f"【依据{i}】《{m.get('title', '未知法规')}》{m.get('article_no', '')}"
            header += f" | {m.get('law_type', '')} | {m.get('status', '')}"
            text = f"{header}\n{doc.page_content.strip()}\n"
            if cur + len(text) > max_length:
                break
            parts.append(text)
            cur += len(text)
        divider = "\n" + "=" * 50 + "\n"
        return divider + divider.join(parts)

    def _build_citations(self, chunks: List[Document]) -> str:
        """确定性拼接依据条款(去重)，保证溯源不依赖LLM"""
        seen, lines = set(), []
        idx = 1
        for doc in chunks:
            m = doc.metadata
            key = (m.get('title', ''), m.get('article_no', ''))
            if key in seen or not m.get('title'):
                continue
            seen.add(key)
            cite = f"{idx}. 《{m.get('title')}》{m.get('article_no', '')}（{m.get('law_type', '')}·{m.get('status', '')}）"
            if m.get('source_url'):
                cite += f"\n   原文: {m['source_url']}"
            lines.append(cite)
            idx += 1
        if not lines:
            return ""
        return "\n\n📎 依据条款：\n" + "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 生成：详细/一般问答
    # ------------------------------------------------------------------ #
    def _answer_prompt(self, role: str) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template("""
你是企业HR合规助手。请严格依据下方检索到的法规条文回答问题，不得编造法条或超出条文范围。

提问者角色: {role_desc}

用户问题: {question}

检索到的法规条文:
{context}

要求:
- 直接给出结论，再结合条文说明依据
- 引用时点明出自哪部法规的第几条
- 条文未覆盖的部分，如实说明"现有检索条文未涉及"
- 语言贴合提问者角色

回答:""")

    def generate_answer(self, query: str, chunks: List[Document], role: str = "employee") -> str:
        context = self._build_context(chunks)
        chain = (
            {"question": RunnablePassthrough(),
             "context": lambda _: context,
             "role_desc": lambda _: ROLE_DESC.get(role, ROLE_DESC["employee"])}
            | self._answer_prompt(role) | self.llm | StrOutputParser()
        )
        answer = chain.invoke(query)
        return answer + self._build_citations(chunks) + DISCLAIMER

    def generate_answer_stream(self, query: str, chunks: List[Document], role: str = "employee"):
        context = self._build_context(chunks)
        chain = (
            {"question": RunnablePassthrough(),
             "context": lambda _: context,
             "role_desc": lambda _: ROLE_DESC.get(role, ROLE_DESC["employee"])}
            | self._answer_prompt(role) | self.llm | StrOutputParser()
        )
        for chunk in chain.stream(query):
            yield chunk
        # 流式结束后追加确定性引用与免责声明
        yield self._build_citations(chunks)
        yield DISCLAIMER

    # ------------------------------------------------------------------ #
    # 生成：清单类(list)
    # ------------------------------------------------------------------ #
    def generate_list_answer(self, query: str, chunks: List[Document], role: str = "employee") -> str:
        if not chunks:
            return "抱歉，没有检索到相关法规条款。"
        lines = []
        seen = set()
        for doc in chunks:
            m = doc.metadata
            key = (m.get('title'), m.get('article_no'))
            if key in seen:
                continue
            seen.add(key)
            summary = doc.page_content.strip().replace("\n", " ")[:60]
            lines.append(f"- 《{m.get('title')}》{m.get('article_no', '')}：{summary}…")
        body = f"为您检索到以下相关法规条款：\n" + "\n".join(lines)
        return body + self._build_citations(chunks) + DISCLAIMER
