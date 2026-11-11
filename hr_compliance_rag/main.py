"""
HR合规问答RAG系统 - 主程序

改自 C8/main.py。流程：意图路由 -> 查询重写 -> 元数据过滤 -> 混合检索 -> 引用溯源生成
"""

import os
import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, HRComplianceConfig
from kb_modules import (
    IngestionModule,
    IndexingModule,
    RetrievalModule,
    DialogueModule,
    GenerationModule,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HRComplianceSystem:
    """企业HR合规智能问答系统"""

    def __init__(self, config: HRComplianceConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.ingestion: IngestionModule = None
        self.indexing: IndexingModule = None
        self.retrieval: RetrievalModule = None
        self.dialogue: DialogueModule = None
        self.generation: GenerationModule = None

        if not Path(self.config.laws_md_path).exists():
            raise FileNotFoundError(
                f"法规目录不存在: {self.config.laws_md_path}，请先运行 scripts/build_corpus.py"
            )
        if not os.getenv("MOONSHOT_API_KEY"):
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

    def initialize_system(self):
        print("🚀 正在初始化HR合规RAG系统...")
        self.ingestion = IngestionModule(self.config.laws_md_path)
        self.indexing = IndexingModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path,
        )
        self.dialogue = DialogueModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
        )
        self.generation = GenerationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        print("✅ 系统初始化完成！")

    def build_knowledge_base(self):
        print("\n正在构建知识库...")
        # 无论是否已有索引，都需要文档与分块(供BM25与父文档回溯)
        print("加载法规文档...")
        self.ingestion.load_documents()
        print("按法条结构分块...")
        chunks = self.ingestion.chunk_documents()

        vectorstore = self.indexing.load_index()
        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
        else:
            print("未找到已保存的索引，开始构建新索引...")
            vectorstore = self.indexing.build_vector_index(chunks)
            self.indexing.save_index()

        print("初始化检索优化...")
        self.retrieval = RetrievalModule(vectorstore, chunks, retriever_k=self.config.retriever_k)

        stats = self.ingestion.get_statistics()
        print("\n📊 知识库统计:")
        print(f"   法规总数: {stats['total_documents']}")
        print(f"   法条块数: {stats['total_chunks']}")
        print(f"   子领域分布: {stats['domains']}")
        print(f"   效力级别分布: {stats['law_types']}")
        print(f"   平均块长: {stats['avg_chunk_size']:.0f} 字")
        print("✅ 知识库构建完成！")

    def _extract_filters_from_query(self, query: str) -> dict:
        """从问题中提取元数据过滤条件(子领域 / 效力级别)"""
        filters = {}
        for domain in IngestionModule.get_supported_domains():
            if domain in query:
                filters['domain'] = domain
                break
        for law_type in ['法律', '行政法规', '部门规章', '司法解释']:
            if law_type in query:
                filters['law_type'] = law_type
                break
        return filters

    def ask_question(self, question: str, role: str = "employee", stream: bool = False):
        if not all([self.retrieval, self.generation, self.dialogue]):
            raise ValueError("请先构建知识库")

        print(f"\n❓ 用户问题: {question}  [角色: {role}]")

        # 1. 意图路由
        route = self.dialogue.query_router(question)
        print(f"🎯 意图类型: {route}")

        # 2. 越界兜底(guardrail)
        if route == 'out_of_scope':
            return ("这个问题超出了劳动用工合规知识库的范围，或涉及需要人工判断的个案。"
                    "建议联系公司法务或HR获取专门意见。")

        # 3. 查询重写(list 保持原样)
        if route == 'list':
            rewritten = question
        else:
            print("🤖 智能分析查询...")
            rewritten = self.dialogue.query_rewrite(question)

        # 4. 检索(自动应用元数据过滤)
        filters = self._extract_filters_from_query(question)
        if filters:
            print(f"应用过滤条件: {filters}")
            chunks = self.retrieval.metadata_filtered_search(rewritten, filters, top_k=self.config.top_k)
            # 过滤后为空(如问题含"劳务派遣"但无该domain标签的法规)时回退无过滤检索，避免漏答
            if not chunks:
                print("⚠️ 过滤后无命中，回退到全库混合检索")
                chunks = self.retrieval.hybrid_search(rewritten, top_k=self.config.top_k)
        else:
            chunks = self.retrieval.hybrid_search(rewritten, top_k=self.config.top_k)

        # 显示命中的法条
        if chunks:
            info = [f"《{c.metadata.get('title')}》{c.metadata.get('article_no', '')}" for c in chunks]
            print(f"🔍 命中 {len(chunks)} 条法规: {', '.join(info)}")
        else:
            print("🔍 未命中相关法条")
            return "抱歉，没有检索到相关法规条款。请尝试更换关键词。"

        # 5. 生成
        if route == 'list':
            print("📋 生成条款清单...")
            return self.generation.generate_list_answer(question, chunks, role=role)
        else:
            print("✍️ 生成合规回答...")
            if stream:
                return self.generation.generate_answer_stream(question, chunks, role=role)
            return self.generation.generate_answer(question, chunks, role=role)

    def run_interactive(self):
        print("=" * 60)
        print("⚖️  企业HR合规智能问答系统  ⚖️")
        print("=" * 60)
        print("💡 基于真实国家法律法规，回答劳动用工合规问题，答案附法条溯源")

        self.initialize_system()
        self.build_knowledge_base()

        role = self.config.default_role
        print(f"\n当前角色视角: {role} (输入 '角色 hr/employee/legal' 切换)")
        print("交互式问答 (输入'退出'结束):")

        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break
                if user_input.startswith('角色'):
                    parts = user_input.split()
                    if len(parts) > 1 and parts[1] in ['hr', 'employee', 'legal']:
                        role = parts[1]
                        print(f"已切换角色视角: {role}")
                    continue

                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                if use_stream:
                    for chunk in self.ask_question(user_input, role=role, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    print(self.ask_question(user_input, role=role, stream=False))
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")

        print("\n感谢使用HR合规问答系统！")


def main():
    try:
        HRComplianceSystem().run_interactive()
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")


if __name__ == "__main__":
    main()
