[English](README.md)

# Galaxy AI

Galaxy AI 是一个本地优先的 AI Lab，用于在自己的机器上运行、训练和探索多种 AI 任务。项目目标是利用本地算力，例如 CUDA、Apple GPU/MPS 或 CPU，而不是依赖云端推理。

## 愿景

应用将提供一个可视化星系界面，每个星球代表一种 AI 任务。首批计划任务包括：

- 文本生成
- 图像生成
- 视觉理解
- 语音转写

模型训练和微调将作为可扩展任务能力，允许用户基于 Hugging Face 预训练模型，使用自建数据集继续训练，并将训练后的模型持久化到本地。

## 计划技术栈

- 前端：Vite、React、TypeScript、Tailwind CSS
- 3D GUI：React Three Fiber 和 Three.js
- 后端：Python 和 FastAPI
- AI 运行时：PyTorch，并按 CUDA、MPS、CPU 顺序回退
- AI 生态：Hugging Face `transformers`、`diffusers`、`safetensors`
- 存储：SQLite 元数据加本地文件系统产物
- 依赖工具：前端使用 `npm`，Python 使用 `uv`

## 本地优先安全原则

- 应用默认完全本地运行。
- 后端服务默认绑定 `127.0.0.1`。
- 不允许遥测。
- 不允许在线推理 API。
- Hugging Face 下载必须由用户显式触发。
- 推理、训练、评估、checkpoint 和训练后模型持久化都在本地完成。

`models/`、`outputs/`、`datasets/`、`checkpoints/` 只是默认兜底目录。实际工作目录必须能在应用 Settings 界面中配置。

## 本地启动

克隆仓库后，或依赖发生变化时，先运行初始化脚本：

```bash
./init.sh
```

初始化完成后，日常启动本地后端和前端：

```bash
./start.sh
```

Windows PowerShell 用户先运行 `./init.ps1`，再运行 `./start.ps1`。

## 仓库状态

当前仓库只包含初始项目规则和元信息。React 与 FastAPI 脚手架会在后续 `develop` 分支上创建。

## 分支流程

- `main`：稳定仓库规则、发布和经过审查的集成点
- `develop`：活跃开发分支
