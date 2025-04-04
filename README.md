# Qdrant Vector Database Utils

这是一个基于 Qdrant 向量数据库的工具包，提供了文本向量化、索引管理和向量检索等功能。该工具包支持同步和异步操作，并集成了多种文本向量模型。

## 功能特点

- 支持多种文本向量模型（BGE、Text2Vec）
- 提供同步和异步操作接口
- 支持批量文本处理和向量检索
- 内置向量相似度计算和归一化处理
- 完整的异常处理和错误恢复机制

## 安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd show_case
```

2. 创建并激活虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 配置

1. 创建 `.env` 文件并设置 Qdrant 配置：
```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your_api_key  # 如果需要
```

## 使用示例

### 同步操作

```python
from qdrant_utils import TextIndexer, BGEEmbedding, QdrantOperations
from qdrant_client import QdrantClient

# 初始化组件
client = QdrantClient()
ops = QdrantOperations(client)
model = BGEEmbedding()
indexer = TextIndexer(model, ops, "my_collection")

# 创建索引
indexer.create_index()

# 添加文本
texts = ["重生之都市修仙", "我在修仙界开网店", "修真聊天群"]
indexer.add_texts(texts)

# 搜索相似文本
results = indexer.search("修仙小说", limit=2)
for r in results:
    print(f"- {r['payload']['title']} (相似度: {r['score']:.4f})")
```

### 异步操作

```python
import asyncio
from qdrant_utils import AsyncTextIndexer, BGEEmbedding, AsyncQdrantOperations
from qdrant_client.async_qdrant_client import AsyncQdrantClient

async def main():
    # 初始化组件
    client = AsyncQdrantClient()
    ops = AsyncQdrantOperations(client)
    model = BGEEmbedding()
    indexer = AsyncTextIndexer(model, ops, "my_collection")

    # 创建索引
    await indexer.create_index()

    # 批量添加文本
    texts = ["重生之都市修仙", "我在修仙界开网店", "修真聊天群"]
    await indexer.add_texts_batch(texts)

    # 批量搜索
    queries = ["修仙小说", "都市小说"]
    results = await indexer.search_batch(queries, limit=2)
    for query, query_results in zip(queries, results):
        print(f"\n查询: {query}")
        for r in query_results:
            print(f"- {r['payload']['title']} (相似度: {r['score']:.4f})")

asyncio.run(main())
```

## 测试

运行单元测试：
```bash
python -m pytest tests/ -v
```

## 项目结构

```
src/qdrant_utils/
├── __init__.py
├── client.py          # Qdrant 客户端配置
├── embeddings.py      # 文本向量模型
├── indexer.py         # 同步索引管理器
├── operations.py      # 同步向量操作
├── async_indexer.py   # 异步索引管理器
└── async_operations.py # 异步向量操作

tests/
├── test_embeddings.py
├── test_indexer.py
└── test_async_operations.py
```

## 依赖

- Python 3.8+
- qdrant-client
- numpy
- transformers
- torch
- python-dotenv

## 许可证

MIT License
