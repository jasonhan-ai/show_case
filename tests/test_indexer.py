"""
索引管理器模块的单元测试。
"""
import unittest
import numpy as np
from qdrant_client import QdrantClient
from src.qdrant_utils.embeddings import BGEEmbedding
from src.qdrant_utils.indexer import TextIndexer
from src.qdrant_utils.operations import QdrantOperations

class TestIndexer(unittest.TestCase):
    """测试索引管理器类"""
    
    def setUp(self):
        """测试前准备"""
        self.client = QdrantClient()
        self.qdrant_ops = QdrantOperations(self.client)
        self.embedding_model = BGEEmbedding()
        self.collection_name = "test_indexer"
        self.indexer = TextIndexer(
            self.embedding_model,
            self.qdrant_ops,
            self.collection_name
        )
        
        # 测试数据
        self.texts = [
            "重生之都市修仙",
            "我在修仙界开网店",
            "修真聊天群",
            "斗破苍穹",
            "完美世界"
        ]
    
    def tearDown(self):
        """测试后清理"""
        self.client.delete_collection(self.collection_name)
    
    def test_create_index(self):
        """测试创建索引"""
        # 测试正常创建
        success = self.indexer.create_index()
        self.assertTrue(success)
        
        # 测试重复创建（应该返回False）
        success = self.indexer.create_index()
        self.assertFalse(success)
        
        # 测试强制重建
        success = self.indexer.create_index(force=True)
        self.assertTrue(success)
    
    def test_add_texts(self):
        """测试添加文本"""
        # 创建索引
        self.indexer.create_index()
        
        # 测试添加单个文本
        success = self.indexer.add_texts([self.texts[0]])
        self.assertTrue(success)
        
        # 测试添加多个文本
        success = self.indexer.add_texts(self.texts[1:])
        self.assertTrue(success)
        
        # 测试添加重复文本
        success = self.indexer.add_texts([self.texts[0]])
        self.assertTrue(success)  # 应该允许更新
    
    def test_search(self):
        """测试搜索功能"""
        # 准备数据
        self.indexer.create_index()
        self.indexer.add_texts(self.texts)
        
        # 测试单个查询
        query = "修仙小说"
        results = self.indexer.search(query, limit=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all("score" in r for r in results))
        self.assertTrue(all("payload" in r for r in results))
        
        # 测试多个查询
        queries = ["修仙小说", "都市小说"]
        results = self.indexer.search_batch(queries, limit=3)
        self.assertEqual(len(results), len(queries))
        for query_results in results:
            self.assertLessEqual(len(query_results), 3)
    
    def test_vector_operations(self):
        """测试向量操作"""
        # 创建索引
        self.indexer.create_index()
        
        # 生成向量
        vectors = self.embedding_model.generate_vector(self.texts)
        
        # 添加向量
        success = self.indexer.add_vectors(vectors, self.texts)
        self.assertTrue(success)
        
        # 搜索向量
        query_vector = vectors[0]
        results = self.indexer.search_by_vector(query_vector, limit=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all("score" in r for r in results))
        self.assertTrue(all("payload" in r for r in results))

if __name__ == '__main__':
    unittest.main() 