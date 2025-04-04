# Qdrant向量数据库工具

本项目提供了用于操作Qdrant向量数据库的工具，包括客户端配置、数据导入和向量检索等功能。

## 安装

1. 克隆仓库
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 配置

在项目根目录创建 `.env` 文件，包含以下变量：
```
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your_api_key  # 可选
```

## 使用示例

```python
from src.qdrant_utils import QdrantClientConfig, QdrantOperations

# 初始化客户端
config = QdrantClientConfig()
client = config.get_client()

# 创建操作实例
ops = QdrantOperations(client)

# 创建集合
ops.create_collection("my_collection", vector_size=768)

# 上传向量
vectors = [[0.1, 0.2, ...], [0.3, 0.4, ...]]  # 你的向量数据
ops.upsert_points("my_collection", vectors)

# 搜索
results = ops.search("my_collection", query_vector=[0.1, 0.2, ...], limit=5)
```

## 功能特点

- 通过环境变量轻松配置客户端
- 支持自定义参数创建集合
- 支持批量上传向量数据和元数据
- 支持可配置参数的向量相似度搜索
