"""
HR合规问答RAG系统配置文件
"""

from dataclasses import asdict, dataclass
from typing import Dict, Any


@dataclass
class HRComplianceConfig:
    """HR合规问答RAG系统配置类"""

    # 路径配置
    raw_data_path: str = "./data/raw/laws.json"   # 原始法规JSON(下载解压后)
    laws_md_path: str = "./data/laws"             # 清洗后的结构化法规Markdown目录
    index_save_path: str = "./vector_index"       # 向量索引持久化目录

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    # 沿用 C8 的 Moonshot(kimi) 通道；如账号不可用该模型名，改此处即可
    llm_model: str = "kimi-k2.6"

    # 检索配置
    top_k: int = 5          # 最终返回的法条(子块)数量
    retriever_k: int = 10   # 单路(向量/BM25)召回候选数，供RRF融合

    # 生成配置
    # 合规问答要求事实准确、可溯源，故温度显著低于 C8 食谱场景(temp=1)
    temperature: float = 1
    max_tokens: int = 2048

    # 角色视角(P1/P2 用"角色视角"替代真实RBAC，P3 混入内部文档再启用真RBAC)
    default_role: str = "employee"   # employee / hr / legal

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HRComplianceConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


# 默认配置实例
DEFAULT_CONFIG = HRComplianceConfig()
