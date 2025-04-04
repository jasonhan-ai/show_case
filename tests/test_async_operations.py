"""
异步操作模块的单元测试。
"""
import unittest
import asyncio
import uuid
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from src.qdrant_utils.async_operations import AsyncQdrantOperations
from src.qdrant_utils.embeddings import BGEEmbedding
from src.qdrant_utils.async_indexer import AsyncTextIndexer

class TestAsyncOperations(unittest.TestCase):
    """测试异步操作类"""
    
    def setUp(self):
        """测试前准备"""
        self.client = AsyncQdrantClient()
        self.ops = AsyncQdrantOperations(self.client)
        self.collection_name = f"test_{uuid.uuid4().hex[:8]}"
        self.embedding_model = BGEEmbedding()
        self.indexer = AsyncTextIndexer(
            self.embedding_model,
            self.ops,
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
        self.queries = ["修仙小说", "都市小说"]
    
    def tearDown(self):
        """测试后清理"""
        async def cleanup():
            await self.ops.delete_collection(self.collection_name)
        asyncio.run(cleanup())
    
    def test_batch_operations(self):
        """测试批量操作"""
        async def run_test():
            # 创建索引
            success = await self.indexer.create_index()
            self.assertTrue(success)
            
            # 批量添加文本
            success = await self.indexer.add_texts_batch(
                texts=self.texts,
                batch_size=2
            )
            self.assertTrue(success)
            
            # 批量搜索
            results = await self.indexer.search_batch(
                queries=self.queries,
                limit=3,
                batch_size=1
            )
            
            # 验证结果
            self.assertEqual(len(results), len(self.queries))
            for query_results in results:
                self.assertLessEqual(len(query_results), 3)
                for res in query_results:
                    self.assertIn("score", res)
                    self.assertIn("payload", res)
                    self.assertIn("title", res["payload"])
        
        asyncio.run(run_test())
    
    def test_concurrent_searches(self):
        """测试并发搜索"""
        async def run_test():
            # 创建索引并添加数据
            await self.indexer.create_index()
            await self.indexer.add_texts_batch(self.texts)
            
            # 并发执行多个搜索
            tasks = [
                self.indexer.search_batch([query])
                for query in self.queries * 3  # 重复查询以增加并发量
            ]
            results = await asyncio.gather(*tasks)
            
            # 验证结果
            self.assertEqual(len(results), len(self.queries) * 3)
            for query_results in results:
                self.assertEqual(len(query_results), 1)  # 每个查询一个结果列表
        
        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main() 