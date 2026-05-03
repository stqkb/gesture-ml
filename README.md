# 🤖 Gesture ML — 手势数字识别

基于 MediaPipe + PyTorch 的手势数字识别系统。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练（首次会自动生成 demo 数据）
python -m src.train

# 3. 预测
python -m src.predict

# 4. 启动 API 服务
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## 项目结构

```
gesture-ml/
├── configs/config.yaml    # 超参数配置
├── data/
│   ├── raw/               # 原始数据
│   └── processed/         # 处理后数据（自动生成）
├── models/                # 训练好的模型
├── src/
│   ├── utils.py           # 工具函数
│   ├── data.py            # 数据加载 & 预处理
│   ├── model.py           # 模型定义
│   ├── train.py           # 训练引擎
│   ├── predict.py         # 推理
│   ├── extract_landmarks.py  # MediaPipe 关键点提取
│   ├── api.py             # FastAPI 服务
│   └── visualize.py       # 可视化
├── tests/                 # 单元测试
├── Makefile               # 快捷命令
└── requirements.txt
```

## 使用真实数据

将你的手势图片放入 `data/raw/`，然后运行关键点提取：

```python
from src.extract_landmarks import extract_landmarks_from_image
landmarks = extract_landmarks_from_image("data/raw/gesture_0.jpg")
```

或使用摄像头实时采集：

```python
from src.extract_landmarks import extract_landmarks_from_camera
extract_landmarks_from_camera(callback=your_handler)
```

## API 接口

启动后访问 http://localhost:8000/docs 查看交互文档。

```bash
# 单个预测
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 0.3, ...]}'  # 63 个浮点数
```

## 配置

编辑 `configs/config.yaml` 调整超参数：

- `model.hidden_dims`: 网络层数和宽度
- `train.epochs`: 训练轮数
- `train.lr`: 学习率
- `train.batch_size`: 批大小
