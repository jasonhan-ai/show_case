"""
向量索引管理模块。
"""
from typing import List, Dict, Any, Optional
from .embeddings import TextEmbedding
from .operations import QdrantOperations

class TextIndexer:
    """文本索引管理器"""
    
    def __init__(
        self,
        embedding_model: TextEmbedding,
        qdrant_ops: QdrantOperations,
        collection_name: str
    ):
        """
        初始化索引管理器。
        
        参数：
            embedding_model: 向量生成模型
            qdrant_ops: Qdrant操作实例
            collection_name: 集合名称
        """
        self.embedding_model = embedding_model
        self.qdrant_ops = qdrant_ops
        self.collection_name = collection_name
    
    def create_index(self, force: bool = False) -> bool:
        """
        创建索引。
        
        参数：
            force: 如果为True，则强制重新创建索引
            
        返回：
            bool: 成功返回True
        """
        if force:
            self.qdrant_ops.delete_collection(self.collection_name)
        
        return self.qdrant_ops.create_collection(
            collection_name=self.collection_name,
            vector_size=self.embedding_model.dimension
        )
    
    def add_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        添加文本到索引。
        
        参数：
            texts: 文本列表
            ids: 可选的ID列表
            metadata: 可选的元数据列表
            
        返回：
            bool: 成功返回True
        """
        vectors = self.embedding_model.encode(texts)
        return self.qdrant_ops.upsert_points(
            collection_name=self.collection_name,
            vectors=vectors,
            ids=ids,
            payload=metadata
        )
    
    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似文本。
        
        参数：
            query: 查询文本
            limit: 最大返回结果数
            score_threshold: 最小相似度阈值
            
        返回：
            List[Dict[str, Any]]: 搜索结果列表，每个结果包含score和payload
        """
        query_vector = self.embedding_model.encode([query])[0]
        results = self.qdrant_ops.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold
        )
        
        return [
            {
                "score": res.score,
                "payload": res.payload
            }
            for res in results
        ] 