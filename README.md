# 🤖 Gesture ML 完整使用教程

> 从零开始：安装 Python → 注册 GitHub → 克隆仓库 → 训练模型 → API 调用 → 摄像头实时识别

---

## 📑 目录

- [第一章：环境准备](#第一章环境准备)
  - [1.1 安装 Python](#11-安装-python)
  - [1.2 安装 Git](#12-安装-git)
  - [1.3 注册 GitHub 账号](#13-注册-github-账号)
- [第二章：获取代码](#第二章获取代码)
  - [2.1 克隆仓库](#21-克隆仓库)
  - [2.2 项目结构一览](#22-项目结构一览)
- [第三章：安装依赖](#第三章安装依赖)
  - [3.1 创建虚拟环境（推荐）](#31-创建虚拟环境推荐)
  - [3.2 安装项目依赖](#32-安装项目依赖)
- [第四章：理解项目](#第四章理解项目)
  - [4.1 它在做什么？](#41-它在做什么)
  - [4.2 核心概念](#42-核心概念)
  - [4.3 配置文件详解](#43-配置文件详解)
- [第五章：训练模型](#第五章训练模型)
  - [5.1 使用内置 Demo 数据训练](#51-使用内置-demo-数据训练)
  - [5.2 使用摄像头采集真实数据](#52-使用摄像头采集真实数据)
  - [5.3 用真实数据重新训练](#53-用真实数据重新训练)
- [第六章：模型预测](#第六章模型预测)
  - [6.1 命令行预测](#61-命令行预测)
  - [6.2 Python 代码调用](#62-python-代码调用)
- [第七章：API 服务](#第七章api-服务)
  - [7.1 启动 API 服务](#71-启动-api-服务)
  - [7.2 API 接口文档](#72-api-接口文档)
  - [7.3 PowerShell 调用示例](#73-powershell-调用示例)
  - [7.4 Python 调用 API](#74-python-调用-api)
  - [7.5 JavaScript 调用 API](#75-javascript-调用-api)
- [第八章：摄像头实时识别](#第八章摄像头实时识别)
- [第九章：自定义与进阶](#第九章自定义与进阶)
  - [9.1 修改超参数](#91-修改超参数)
  - [9.2 使用 Makefile 快捷命令](#92-使用-makefile-快捷命令)
  - [9.3 运行测试](#93-运行测试)
- [附录：常见问题](#附录常见问题)

---

## 第一章：环境准备

### 1.1 安装 Python

本项目需要 **Python 3.10 或更高版本**。

#### Windows

1. 打开浏览器，访问 https://www.python.org/downloads/
2. 点击 **"Download Python 3.x.x"** 按钮（下载最新版）
3. 运行安装程序，**⚠️ 勾选 "Add Python to PATH"**（最重要的一步！）
4. 点击 "Install Now"
5. 安装完成后，打开 **PowerShell**，输入以下命令验证：

```powershell
python --version
# 应该输出类似: Python 3.12.x

pip --version
# 应该输出类似: pip 24.x.x from ...
```

> **如果 `python` 命令不生效**，尝试 `python3` 或 `py`。如果都不行，说明安装时没勾选 "Add Python to PATH"，需要重新安装。

#### macOS

```bash
# 方法一：使用 Homebrew（推荐）
brew install python

# 方法二：从官网下载
# 访问 https://www.python.org/downloads/ 下载安装
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

---

### 1.2 安装 Git

Git 是版本控制工具，用来下载和管理代码。

#### Windows

1. 访问 https://git-scm.com/download/win
2. 下载并运行安装程序，一路点 "Next"（默认选项即可）
3. 安装完成后，打开 PowerShell 验证：

```powershell
git --version
# 应该输出: git version 2.x.x
```

#### macOS

```bash
# macOS 通常自带 Git，如果没有：
xcode-select --install
```

#### Linux

```bash
sudo apt install git
```

---

### 1.3 注册 GitHub 账号

1. 打开 https://github.com
2. 点击右上角 **"Sign up"**
3. 输入邮箱、密码、用户名
4. 完成邮箱验证
5. 注册成功后，你就可以 Fork、Star 和克隆项目了

#### （可选）配置 Git 身份

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"
```

---

## 第二章：获取代码

### 2.1 克隆仓库

打开 PowerShell（Windows）或终端（macOS/Linux），执行：

```bash
# 进入你想存放项目的目录
cd ~/Desktop    # 或者任何你喜欢的位置

# 克隆仓库
git clone https://github.com/stqkb/gesture-ml.git

# 进入项目目录
cd gesture-ml
```

> **网络问题？** 如果 GitHub 克隆太慢，可以使用镜像或代理：
> ```bash
> # 使用代理（如果你有的话）
> git config --global http.proxy http://127.0.0.1:7890
> 
> # 或者用 gitee 镜像（如果有的话）
> git clone https://gitee.com/mirrors/gesture-ml.git
> ```

### 2.2 项目结构一览

```
gesture-ml/
├── configs/
│   └── config.yaml          # ⚙️ 超参数配置文件
├── data/
│   ├── raw/                  # 📁 原始手势数据（0-9 每个数字一个文件夹）
│   │   ├── 0/                #   数字 0 的样本 (.npy 文件)
│   │   ├── 1/                #   数字 1 的样本
│   │   └── ...               #   ...
│   └── processed/            # 📊 处理后的数据（自动生成）
├── models/
│   └── best_model.pt         # 🧠 训练好的模型文件
├── src/
│   ├── utils.py              #   工具函数（配置加载、随机种子等）
│   ├── data.py               #   数据加载 & 预处理
│   ├── model.py              #   神经网络模型定义
│   ├── train.py              #   🏋️ 训练引擎
│   ├── predict.py            #   🔮 推理/预测模块
│   ├── collect_data.py       #   📷 摄像头数据采集
│   ├── extract_landmarks.py  #   ✋ MediaPipe 手部关键点提取
│   ├── camera_predict.py     #   🎥 摄像头实时识别
│   ├── api.py                #   🌐 FastAPI 服务
│   └── visualize.py          #   📈 可视化工具
├── tests/
│   └── test_pipeline.py      #   ✅ 单元测试
├── Makefile                  #   快捷命令
├── requirements.txt          #   Python 依赖列表
└── README.md                 #   项目说明
```

---

## 第三章：安装依赖

### 3.1 创建虚拟环境（推荐）

虚拟环境可以避免不同项目的依赖冲突，**强烈推荐使用**。

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows CMD:
venv\Scripts\activate.bat

# macOS/Linux:
source venv/bin/activate

# 激活后，命令行前面会出现 (venv) 标志
```

> **PowerShell 执行策略报错？**
> 如果看到"无法加载文件，因为在此系统上禁止运行脚本"，执行：
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3.2 安装项目依赖

```bash
pip install -r requirements.txt
```

这会安装以下主要库：

| 库 | 用途 |
|---|---|
| `torch` | PyTorch 深度学习框架 |
| `mediapipe` | Google 的手部关键点检测 |
| `opencv-python` | 摄像头和图像处理 |
| `fastapi` + `uvicorn` | Web API 服务 |
| `numpy` | 数值计算 |
| `scikit-learn` | 数据拆分和评估 |
| `matplotlib` | 可视化绘图 |
| `pyyaml` | 配置文件解析 |
| `tqdm` | 训练进度条 |

> **安装 PyTorch 很慢？** 如果你有 NVIDIA GPU，可以安装 CUDA 版本加速训练：
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 第四章：理解项目

### 4.1 它在做什么？

这个项目实现了一个**手势数字识别系统**：

```
摄像头画面 → MediaPipe 提取 21 个手部关键点 → 63 维特征向量 → 神经网络 → 识别出 0-9 的数字
```

具体流程：

1. **MediaPipe** 检测画面中的手，定位 21 个关键点（每个点有 x, y, z 三个坐标 → 共 63 个数字）
2. **归一化**：以手腕为原点居中，缩放到单位范围
3. **MLP 神经网络**：输入 63 维向量，输出 10 个类别的概率
4. **预测结果**：概率最高的那个数字就是识别结果

```
手部 21 个关键点示意：

        8   12  16  20
        |   |   |   |
    4   7   11  15  19
    |   |   |   |   |
    3   6   10  14  18
    |   |   |   |   |
    2   5   9   13  17
     \  |   |   |  /
      \ |   |   | /
        1-----------
        |
        0 (手腕)
```

### 4.2 核心概念

| 概念 | 说明 |
|---|---|
| **Landmarks** | 手部 21 个关键点的坐标，每个点有 (x, y, z)，共 63 维 |
| **MLP** | 多层感知机（Multi-Layer Perceptron），一种基础的神经网络 |
| **Epoch** | 完整遍历一次训练数据集称为一个 epoch |
| **Batch Size** | 每次送入模型的样本数量 |
| **Learning Rate** | 学习率，控制参数更新的步长 |
| **Validation** | 验证集，用来监控模型是否过拟合 |

### 4.3 配置文件详解

编辑 `configs/config.yaml` 可以调整所有超参数：

```yaml
data:
  raw_path: "data/raw"           # 原始数据路径
  processed_path: "data/processed" # 处理后数据路径
  test_size: 0.2                 # 测试集占比 20%
  val_size: 0.1                  # 验证集占比 10%
  random_state: 42               # 随机种子（保证可复现）

model:
  name: "mlp"
  input_dim: 63                  # 输入维度（21 个点 × 3 个坐标）
  num_classes: 10                # 输出类别数（数字 0-9）
  hidden_dims: [128, 64]         # 隐藏层结构：两层，128 和 64 个神经元
  dropout: 0.3                   # Dropout 比例（防过拟合）

train:
  epochs: 50                     # 训练轮数
  batch_size: 32                 # 批大小
  lr: 0.001                      # 学习率
  weight_decay: 0.0001           # 权重衰减（L2 正则化）
  device: "auto"                 # auto / cpu / cuda
  seed: 42                       # 随机种子

output:
  model_path: "models/best_model.pt" # 模型保存路径
  log_interval: 10               # 每 N 个 batch 打印一次日志
```

---

## 第五章：训练模型

### 5.1 使用内置 Demo 数据训练

项目内置了**自动生成 Demo 数据**的功能，不需要摄像头就能跑通整个流程。

```bash
# 确保在项目根目录 gesture-ml/ 下
python -m src.train
```

你会看到类似输出：

```
⚡ No real data found, generating demo dataset...
   Saved demo data to data/processed/ (2000 samples)
📊 Data split: train=1400 | val=200 | test=400
Model: 18,858 trainable parameters

Training for 50 epochs...

  Epoch 001 | Train Loss: 2.1543 Acc: 0.234 | Val Loss: 1.8765 Acc: 0.350 | LR: 0.001000
  Epoch 002 | Train Loss: 1.6543 Acc: 0.412 | Val Loss: 1.4321 Acc: 0.525 | LR: 0.001000
  ...
  Epoch 050 | Train Loss: 0.1234 Acc: 0.985 | Val Loss: 0.2345 Acc: 0.940 | LR: 0.000500
  Saved best model -> models/best_model.pt (val_acc=0.940)

Training completed in 12.3s
Best Val Accuracy: 0.940

Final test evaluation...
   Test Loss: 0.2100 | Test Accuracy: 0.935
```

> **Demo 数据 vs 真实数据**：Demo 数据是程序生成的模拟数据，用来验证代码能跑通。要真正识别手势，需要用摄像头采集真实数据（见下一节）。

### 5.2 使用摄像头采集真实数据

这是最有趣的部分——用你自己的手来训练模型！

```bash
python -m src.collect_data
```

启动后：

1. 摄像头窗口会弹出
2. 对着摄像头比一个数字手势（比如比个 "3"）
3. 按键盘上的 **数字键 0-9** 保存当前帧的手部关键点
4. 每个数字建议采集 **20-50 个样本**（变换角度、距离、光线）
5. 采集完所有数字后，按 **q** 退出

```
操作方法：
  - 对着镜头比一个数字手势
  - 按键盘 0-9 保存当前帧的关键点
  - 每个数字建议采集 20-50 个样本（不同角度）
  - 按 q 退出

摄像头已启动
  已保存数字 3 的样本 (共 1 个)
  已保存数字 3 的样本 (共 2 个)
  已保存数字 7 的样本 (共 1 个)
  ...
```

采集的数据保存在 `data/raw/` 目录下：

```
data/raw/
├── 0/          # 数字 0 的样本
│   ├── 0000.npy
│   ├── 0001.npy
│   └── ...
├── 1/          # 数字 1 的样本
│   └── ...
└── ...
```

> **💡 采集技巧**：
> - 每个数字变换不同角度和距离
> - 左右手都采集一些
> - 光线要充足
> - 背景尽量简洁
> - 每个数字至少 20 个样本，越多越好

### 5.3 用真实数据重新训练

采集完数据后，重新训练模型：

```bash
# 删除旧的处理数据和模型（可选）
# Windows PowerShell:
Remove-Item -Recurse data/processed -ErrorAction SilentlyContinue
Remove-Item models/best_model.pt -ErrorAction SilentlyContinue

# macOS/Linux:
rm -rf data/processed models/best_model.pt

# 重新训练
python -m src.train
```

这次会自动检测到 `data/raw/` 里的真实数据：

```
📊 Loaded 350 real collected samples
📊 Data split: train=245 | val=35 | test=70
Model: 18,858 trainable parameters

Training for 50 epochs...
  ...
```

---

## 第六章：模型预测

### 6.1 命令行预测

```bash
python -m src.predict
```

输出示例：

```
Model loaded (val_acc=0.940) on cpu

Predicted digit: 7
Class probabilities:
   7: XXXXXXXXXXXXXXXXXXXXXXXXXX 0.923
   1: XXX 0.034
   9: XX 0.021
   4: X 0.012
   ...
```

### 6.2 Python 代码调用

```python
from src.predict import GesturePredictor
import numpy as np

# 加载模型（只需加载一次）
predictor = GesturePredictor()

# 方式一：直接用 63 维向量预测
features = np.random.randn(63).astype(np.float32)  # 替换成真实数据
digit = predictor.predict(features)
print(f"识别结果: {digit}")

# 方式二：获取各类别概率
proba = predictor.predict_proba(features)
print(f"数字: {digit}, 置信度: {proba[str(digit)]:.1%}")

# 方式三：批量预测
batch = np.random.randn(10, 63).astype(np.float32)  # 10 个样本
digits = predictor.predict_batch(batch)
print(f"批量结果: {digits}")
```

### 6.3 从图片提取关键点并预测

```python
from src.extract_landmarks import extract_landmarks_from_image
from src.predict import GesturePredictor

# 从图片提取手部关键点
landmarks = extract_landmarks_from_image("path/to/hand_photo.jpg")
print(f"关键点维度: {landmarks.shape}")  # (63,)

# 用模型预测
predictor = GesturePredictor()
digit = predictor.predict(landmarks)
print(f"识别结果: {digit}")
```

---

## 第七章：API 服务

### 7.1 启动 API 服务

```bash
# 方式一：直接启动
uvicorn src.api:app --host 0.0.0.0 --port 8000

# 方式二：带热重载（开发模式，修改代码自动重启）
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# 方式三：使用 Makefile
make api
```

启动后你会看到：

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

打开浏览器访问 **http://localhost:8000/docs** 可以看到交互式 API 文档（Swagger UI）。

### 7.2 API 接口文档

#### `GET /` — 健康检查

```bash
curl http://localhost:8000/
```

响应：
```json
{"status": "ok", "model_loaded": true}
```

#### `POST /predict` — 单个预测

请求 63 个浮点数（21 个手部关键点 × 3 个坐标），返回识别的数字。

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"features\": [0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.2, 0.4, 0.6, 0.1, 0.3, 0.5]}"
```

响应：
```json
{
  "digit": 7,
  "confidence": 0.923,
  "probabilities": {
    "0": 0.001,
    "1": 0.034,
    "2": 0.002,
    "3": 0.005,
    "4": 0.012,
    "5": 0.001,
    "6": 0.002,
    "7": 0.923,
    "8": 0.010,
    "9": 0.010
  }
}
```

#### `POST /predict/batch` — 批量预测

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d "{\"samples\": [[0.5, 0.3, ...(63个数)...], [0.1, 0.2, ...(63个数)...]]}"
```

#### `GET /health` — 健康详情

```bash
curl http://localhost:8000/health
```

响应：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "val_acc": 0.940
}
```

### 7.3 PowerShell 调用示例

在 Windows PowerShell 中调用 API：

```powershell
# 单个预测
$body = @{
    features = @(0.5) * 63   # 63 个 0.5，实际使用时替换为真实数据
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://localhost:8000/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body

# 输出结果
Write-Host "识别结果: $($response.digit)"
Write-Host "置信度: $($response.confidence)"

# 查看所有类别概率
$response.probabilities | Format-Table
```

```powershell
# 批量预测
$batchBody = @{
    samples = @(
        @(0.5) * 63,
        @(0.3) * 63
    )
} | ConvertTo-Json -Depth 3

$batchResponse = Invoke-RestMethod -Uri "http://localhost:8000/predict/batch" `
    -Method POST `
    -ContentType "application/json" `
    -Body $batchBody

$batchResponse.predictions | ForEach-Object {
    Write-Host "数字: $($_.digit), 置信度: $($_.confidence)"
}
```

```powershell
# 健康检查
$health = Invoke-RestMethod -Uri "http://localhost:8000/health"
Write-Host "状态: $($health.status)"
Write-Host "模型已加载: $($health.model_loaded)"
Write-Host "验证准确率: $($health.val_acc)"
```

### 7.4 Python 调用 API

```python
import requests

# 单个预测
features = [0.5] * 63  # 替换为真实的手部关键点数据
resp = requests.post("http://localhost:8000/predict",
                     json={"features": features})
result = resp.json()
print(f"识别结果: {result['digit']}, 置信度: {result['confidence']:.1%}")

# 批量预测
resp = requests.post("http://localhost:8000/predict/batch",
                     json={"samples": [features, [0.3]*63]})
for pred in resp.json()["predictions"]:
    print(f"数字: {pred['digit']}, 置信度: {pred['confidence']:.1%}")
```

### 7.5 JavaScript 调用 API

```javascript
// 单个预测
const features = new Array(63).fill(0.5); // 替换为真实数据

const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features })
});

const result = await response.json();
console.log(`识别结果: ${result.digit}, 置信度: ${(result.confidence * 100).toFixed(1)}%`);
```

---

## 第八章：摄像头实时识别

这是最终的完整体验——用摄像头实时识别手势数字。

```bash
python -m src.camera_predict
```

启动后：

1. 摄像头窗口弹出
2. 对着摄像头比一个数字手势
3. 画面上会显示：
   - 手部骨骼连线（MediaPipe 绘制）
   - 识别出的数字和置信度
4. 按 **q** 退出

```
📷 摄像头已启动，按 'q' 退出
```

> **要求**：必须先完成训练（第五章），确保 `models/best_model.pt` 存在。

---

## 第九章：自定义与进阶

### 9.1 修改超参数

编辑 `configs/config.yaml`：

```yaml
# 想训练更多轮？
train:
  epochs: 100    # 从 50 改到 100

# 想让模型更大？
model:
  hidden_dims: [256, 128, 64]   # 三层，更宽
  dropout: 0.2                   # 降低 dropout

# 学习率调小一点？
train:
  lr: 0.0005
```

修改后重新训练即可。

### 9.2 使用 Makefile 快捷命令

项目提供了 Makefile，可以用更短的命令操作：

```bash
make install   # 安装依赖
make train     # 训练模型
make predict   # 运行预测
make api       # 启动 API 服务
make test      # 运行测试
make clean     # 清理生成的文件
make help      # 查看所有命令
```

> **Windows 用户**：Makefile 需要安装 `make` 工具。可以用 `choco install make`（需要先安装 Chocolatey），或者直接用对应的 Python 命令。

### 9.3 运行测试

```bash
python -m pytest tests/ -v

# 或者
make test
```

---

## 附录：常见问题

### Q: `python` 命令不识别？

Windows 上试试 `python3` 或 `py`。如果都不行，说明安装 Python 时没勾选 "Add Python to PATH"，需要重新安装。

### Q: `pip install` 太慢？

使用国内镜像源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: MediaPipe 安装失败？

MediaPipe 对 Python 版本有要求。确保使用 Python 3.10-3.12。Python 3.13+ 可能不支持。

### Q: 摄像头打不开？

1. 确认摄像头没被其他程序占用
2. Windows：检查隐私设置 → 摄像头权限
3. 试试修改代码中的 `cv2.VideoCapture(0)` 为 `cv2.VideoCapture(1)`（如果有多摄像头）

### Q: 训练时 GPU 没被使用？

```python
import torch
print(torch.cuda.is_available())  # 应该输出 True
```

如果输出 `False`，说明没装 CUDA 版本的 PyTorch，或者没有 NVIDIA 显卡。用 CPU 训练也可以，只是慢一些。

### Q: `models/best_model.pt` 不存在？

说明还没训练过模型。先运行 `python -m src.train`。

### Q: API 返回 400 错误？

确认你发送的 `features` 数组正好是 **63 个浮点数**（21 个关键点 × 3 个坐标）。

### Q: 如何提高识别准确率？

1. 采集更多真实数据（每个数字 50+ 个样本）
2. 变换角度、距离、光线采集
3. 增加训练轮数（`epochs`）
4. 调整模型结构（`hidden_dims`）
5. 数据增强已经内置（随机缩放、噪声、翻转）

---

> **📌 项目地址**: https://github.com/stqkb/gesture-ml
>
> **📧 有问题？** 在 GitHub 上提 Issue：https://github.com/stqkb/gesture-ml/issues
