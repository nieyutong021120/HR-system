"""
对话理解模块 - 意图路由 + 查询重写 + 越界兜底

query_router/query_rewrite 搬自 C8/generation_integration.py，
新增 out_of_scope 分类(guardrail)，并把重写规则改为劳动用工场景。
多轮上下文改写(P2)预留接口。
"""

import os
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class DialogueModule:
    """对话理解模块 - 意图路由与查询重写"""

    def __init__(self, model_name: str = "kimi-k2-0711-preview", temperature: float = 0.2):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        self.setup_llm()

    def setup_llm(self):
        logger.info(f"正在初始化对话LLM: {self.model_name}")
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")
        self.llm = MoonshotChat(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=512,
            max_retries=5,
            moonshot_api_key=api_key,
        )
        logger.info("对话LLM初始化完成")

    # ------------------------------------------------------------------ #
    def query_router(self, query: str) -> str:
        """
        意图路由：
          list        - 想要法规/条款清单或有哪些规定
          detail      - 想要具体标准/如何操作，需要精确引用法条
          general     - 概念解释类
          out_of_scope- 与劳动用工无关或需人工处理(兜底，防幻觉)
        """
        prompt = ChatPromptTemplate.from_template("""
你是企业HR合规助手的意图分类器。请将用户问题归为以下四类之一：

1. 'list' - 想要清单/有哪些规定。例：关于试用期有哪些规定？涉及社保的法规有哪些？
2. 'detail' - 想要具体标准或操作方法，需精确引用法条。例：试用期最长多久？辞退补偿怎么算？年假几天？
3. 'general' - 概念解释。例：什么是无固定期限劳动合同？什么算工伤？
4. 'out_of_scope' - 满足以下任一即归此类：(a)与劳动用工/HR无关(如闲聊、天气、写诗)；(b)需要人工介入的个案裁决(如"我和公司的官司能不能赢")；(c)虽与HR沾边但本知识库(仅劳动法律法规)未覆盖的主题，如税务筹划、股权激励方案设计、财务/会计、商业保险、具体薪酬数字测算等。例：今天天气如何？帮我算下我和公司的官司能不能赢？公司股权激励的税务筹划怎么做？

请只返回一个词：list、detail、general 或 out_of_scope

用户问题: {query}

分类结果:""")
        chain = {"query": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()
        result = chain.invoke(query).strip().lower()
        if result in ['list', 'detail', 'general', 'out_of_scope']:
            return result
        return 'general'

    # ------------------------------------------------------------------ #
    def query_rewrite(self, query: str) -> str:
        """劳动用工场景的智能查询重写(具体明确的查询保持原样)"""
        prompt = PromptTemplate(
            template="""
你是劳动用工领域的查询分析助手。判断是否需要重写用户查询以提升法规检索效果。

原始查询: {query}

规则：
1. 具体明确(直接返回原查询)：含明确法律概念或问题，如"试用期最长多久""经济补偿金怎么计算""工伤认定流程"
2. 模糊口语(需重写为规范法律术语)：如"被开了有没有赔偿"→"用人单位解除劳动合同的经济补偿"，"上班多久有年假"→"职工带薪年休假天数规定"

重写原则：保持原意、补充规范法律术语、简洁。

请输出最终查询(无需重写就返回原查询):""",
            input_variables=["query"]
        )
        chain = {"query": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()
        response = chain.invoke(query).strip()
        # 模型偶发返回空串(或仅标点)，会导致后续检索用空查询 —— 回退原查询
        if not response or len(response) < 2:
            logger.warning(f"查询重写返回空/异常('{response}')，回退原查询: '{query}'")
            return query
        if response != query:
            logger.info(f"查询已重写: '{query}' -> '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")
        return response

    # ------------------------------------------------------------------ #
    def rewrite_with_history(self, query: str, history: List[str]) -> str:
        """
        [P2 预留] 结合多轮对话历史做指代消解改写。
        P1 暂直接返回原查询。
        """
        # TODO(P2): 用历史上下文补全指代，如"那劳务派遣呢？"
        return query
