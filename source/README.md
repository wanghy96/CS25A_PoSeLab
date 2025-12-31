# PoseLab-2D（Source Code / 开发者版）

本仓库包含 **PoseLab-2D** 姿态分析系统的完整源码，基于 **OpenMMLab / MMPose**，实现了从视频输入到姿态分析、评分、可视化输出的完整流水线。

> 📌 **本 README 面向对象**
>
> - 课程助教 / 任课老师  
> - 项目合作者  
> - 未来维护项目的你自己  
> - 希望复现实验流程的开发者  

目标是确保：  
**任何一个具备 Python / CV 基础的人，拿到源码后都能顺利运行、理解并复现结果。**

---

## 一、项目整体结构

```text
PoseLab-2D/
├─ app_gui.py              # 图形界面入口（最终打包主入口）
├─ app_cli.py              # 命令行入口（开发 / 批处理 / 复现实验）
├─ pose_video_ios_app.py   # 视频级姿态分析与可视化主流程
├─ pose_system_v4_ch.py    # 核心姿态指标与评分算法（算法核心）
├─ plotting.py             # CSV 导出与统计图绘制
├─ resources.py            # 资源路径管理（兼容 PyInstaller）
├─ assets/                 # 模型、配置、字体等静态资源
│   ├─ mmpose_config.py
│   ├─ mmpose_checkpoint.pth
│   └─ fonts/
│       └─ NotoSansSC-Regular.ttf
├─ output/                 # 分析结果输出目录
├─ PoseLab.spec            # PyInstaller 打包配置文件
└─ README_source.md        # 本文件（源码说明）
```

---

## 二、运行环境要求

### 1️⃣ 操作系统

- **Windows 10 / 11（已完整测试）**
- macOS（开发阶段使用过，路径逻辑已兼容）

---

### 2️⃣ Python 环境

- **Python 3.8（强烈推荐）**
- 使用 **conda 环境**（避免依赖冲突）

```bash
conda create -n openmmlab python=3.8 -y
conda activate openmmlab
```

---

### 3️⃣ PyTorch 安装

#### ▶ CPU 版本（最稳妥，推荐）

```bash
pip install torch torchvision torchaudio
```

#### ▶ CUDA 版本（有 NVIDIA GPU 时）

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

### 4️⃣ OpenMMLab / MMPose 依赖

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmpose
```

---

### 5️⃣ 其他依赖库

```bash
pip install opencv-python pillow matplotlib pandas reportlab
```

---

## 三、核心模块说明（重点）

### 🔹 1. `pose_system_v4_ch.py` —— 核心算法模块

**职责：**

- 姿态几何关系计算  
- 姿态指标定义与评分逻辑  
- 与 UI / 视频 / 模型完全解耦  

**主要内容：**

- 姿态指标：
  - FHA（头前伸）
  - FHP（头部前倾）
  - SBA（肩部不平衡）
  - PPA（骨盆前倾）
  - PPT（骨盆侧倾）
  - TKA（膝关节角）

- 指标平滑：
  - `MetricSmoother`

- 区域与综合评分：
  - `compute_region_scores()`

- 姿态分析主接口：
  - `analyze_posture_v3(kpts)`

📌 **这是整个 PoseLab 的“理论与算法核心”**

---

### 🔹 2. `pose_video_ios_app.py` —— 视频分析主流程

**职责：**

- 视频逐帧读取  
- 调用 MMPose 进行关键点推理  
- 调用姿态算法进行分析  
- 完成可视化与结果输出  

**功能包括：**

- MMPose Top-down 模型初始化  
- 关键点推理  
- 姿态问题等级判定  
- 骨架绘制与问题高亮  
- iOS 风格信息面板合成  

**输出内容：**

- 标注后的视频  
- CSV 数据文件  
- 姿态指标随时间变化图  

📌 **这是系统的“工程核心模块”**

---

### 🔹 3. `app_gui.py` —— 图形界面入口

**职责：**

- 提供 Windows 图形界面  
- 选择输入视频文件  
- 调用 `process_video_app()` 启动完整流程  

**特点：**

- 不包含任何算法实现  
- 作为 PyInstaller 打包的主入口  
- 面向最终用户  

📌 **这是最终产品级入口**

---

### 🔹 4. `app_cli.py` —— 命令行入口

**职责：**

- 提供无 GUI 的命令行接口  
- 用于批量处理、调试与实验复现  

**示例：**

```bash
python app_cli.py --video input.mp4 --out output/
```

---

### 🔹 5. `plotting.py` —— 数据导出与可视化

**职责：**

- CSV 文件写入  
- 姿态指标随时间变化图绘制  
- 使用 matplotlib（Agg 后端，适合打包）

---

### 🔹 6. `resources.py` —— 资源路径管理

**职责：**

- 统一管理 `assets/` 路径  
- 同时兼容：
  - 源码运行  
  - PyInstaller 打包后运行  

📌 **这是程序可被成功打包的重要保障模块**

---

## 四、源码运行方式

### ▶ GUI 模式（推荐）

```bash
python app_gui.py
```

---

### ▶ CLI 模式（复现实验）

```bash
python app_cli.py --video your_video.mp4 --out output/
```

---

## 五、模型与资源说明

### 📦 `assets/` 目录

- `mmpose_config.py`  
  → MMPose 模型配置文件  

- `mmpose_checkpoint.pth`  
  → 对应模型权重文件  

- `fonts/NotoSansSC-Regular.ttf`  
  → 中文字体（用于信息面板与报告）

⚠️ **assets 目录必须完整保留，否则程序无法运行**

---

## 六、关于打包（简要说明）

本项目使用 **PyInstaller（onedir 模式）** 打包：

```bash
python -m PyInstaller app_gui.py --onedir --windowed
```

生成结果：

```text
dist/PoseLab/
├─ PoseLab.exe
├─ _internal/
```

`PoseLab.exe` 与 `_internal/` **必须同时存在**。

---

## 七、声明

- 本项目为 **教学 / 科研 / 工程原型**
- 姿态评分不构成医学诊断
- 算法基于经验与几何建模，不代表临床标准

---

## 八、作者

**PoseLab-2D**  
© 2025
