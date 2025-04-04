"""
向量生成模块的单元测试。
"""
import unittest
import numpy as np
from src.qdrant_utils.embeddings import BGEEmbedding, Text2VecEmbedding

class TestEmbeddings(unittest.TestCase):
    """测试向量生成类"""
    
    def setUp(self):
        """测试前准备"""
        self.texts = [
            "重生之都市修仙",
            "我在修仙界开网店",
            "修真聊天群",
            "斗破苍穹",
            "完美世界"
        ]
    
    def test_bge_embedding(self):
        """测试BGE模型向量生成"""
        model = BGEEmbedding()
        vectors = model.generate_vector(self.texts)
        
        # 检查向量数量
        self.assertEqual(len(vectors), len(self.texts))
        
        # 检查向量维度
        self.assertEqual(len(vectors[0]), model.vector_size)
        
        # 检查向量是否已归一化
        for vector in vectors:
            norm = np.linalg.norm(vector)
            self.assertAlmostEqual(norm, 1.0, places=6)
        
        # 检查相似文本的向量相似度
        similar_texts = ["修仙小说", "修真小说"]
        similar_vectors = model.generate_vector(similar_texts)
        similarity = np.dot(similar_vectors[0], similar_vectors[1])
        self.assertGreater(similarity, 0.8)
        
        # 检查不相似文本的向量相似度
        different_texts = ["修仙小说", "科幻小说"]
        different_vectors = model.generate_vector(different_texts)
        similarity = np.dot(different_vectors[0], different_vectors[1])
        self.assertLess(similarity, 0.8)
    
    def test_text2vec_embedding(self):
        """测试Text2Vec模型向量生成"""
        model = Text2VecEmbedding()
        vectors = model.generate_vector(self.texts)
        
        # 检查向量数量
        self.assertEqual(len(vectors), len(self.texts))
        
        # 检查向量维度
        self.assertEqual(len(vectors[0]), model.vector_size)
        
        # 检查向量是否已归一化
        for vector in vectors:
            norm = np.linalg.norm(vector)
            self.assertAlmostEqual(norm, 1.0, places=6)
        
        # 检查相似文本的向量相似度
        similar_texts = ["修仙小说", "修真小说"]
        similar_vectors = model.generate_vector(similar_texts)
        similarity = np.dot(similar_vectors[0], similar_vectors[1])
        self.assertGreater(similarity, 0.8)
        
        # 检查不相似文本的向量相似度
        different_texts = ["修仙小说", "科幻小说"]
        different_vectors = model.generate_vector(different_texts)
        similarity = np.dot(different_vectors[0], different_vectors[1])
        self.assertLess(similarity, 0.8)

if __name__ == '__main__':
    unittest.main() 