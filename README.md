# 生成编码 (Generative Encoding)

基于最小描述长度（MDL）原则的数据表征新范式

## 概述

**生成编码**是一种创新的数据表征框架，它将程序化生成、神经隐式表示与MDL原则在数学上统一起来。该框架利用可计算序列驱动层级可加的隐式场，实现从宏观到微观的统一表征。

### 核心特性

- ✨ **MDL优势**：在具有规律性的数据上实现显著的描述长度压缩（可达100x以上）
- 🔄 **层级表征**：残差金字塔模型从粗到细逐步细化表示
- 📐 **理论保证**：提供收敛性、稳定性和渐进细化的数学证明
- 🎯 **无限分辨率**：基于隐式场，支持任意坐标查询
- 🧮 **可计算性**：编码信息为可验证的生成规律

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```python
from generative_encoding import GenerativeEncodingFramework
import numpy as np

# 创建框架
framework = GenerativeEncodingFramework(dimension=2)

# 准备数据
coordinates = np.random.rand(100, 2)
values = np.sin(2 * np.pi * coordinates[:, 0])

# 编码
model, mdl_bound = framework.encode(coordinates, values, n_scales=4)
print(f"压缩比: {mdl_bound.advantage:.2f}x")

# 解码
new_coords = np.random.rand(50, 2)
predicted = framework.decode(new_coords)
```

### 运行演示

```bash
# 基本演示
python generative_encoding.py

# 完整示例集
python examples.py
```

## 文档

- 📖 [在线文档](https://zhenpeng1979.github.io) - 完整的框架介绍和可视化
- 📝 [理论基础](theory.md) - 数学形式化和定理证明
- 💻 [代码示例](examples.py) - 5个实践演示

## 核心组件

1. **可计算序列（Computable Sequences）**
   - 提供算法可描述的模式生成
   - 支持傅里叶、多项式等多种序列类型

2. **隐式场（Implicit Fields）**
   - 连续函数表示，无限分辨率
   - 由可计算序列驱动，参数高效

3. **分段权重函数（Piecewise Weights）**
   - 位序-尺度对齐
   - 确保不同层级间平滑过渡

4. **残差金字塔（Residual Pyramid）**
   - 层级加性分解
   - 保证收敛性和渐进细化

## 理论基础

### MDL优势定理

对于具有规律性的数据，生成编码的描述长度显著小于原始数据：

```
L(G) + K(G|F) ≤ K(D) + O(log n)
```

其中：
- `L(G)`: 模型编码长度
- `K(G|F)`: 条件柯尔莫哥洛夫复杂度
- `K(D)`: 数据复杂度

### 收敛性保证

残差金字塔模型保证几何级数收敛：

```
‖ε(k+1)‖ ≤ α·‖ε(k)‖, where 0 < α < 1
```

### 稳定性界

表示满足Lipschitz连续性：

```
‖R(x) - R(y)‖ ≤ L·‖x - y‖
```

## 应用场景

- 🔬 **科学计算**：物理场模拟、天文数据、分子动力学
- 🖼️ **计算机视觉**：隐式神经表示（INR）、3D重建、超分辨率
- 📊 **信号处理**：音频编码、传感器数据、时序建模
- 📈 **数据压缩**：具有规律性的大规模数据集

## 实验结果

在规律性数据上的压缩比：

| 数据类型 | 压缩比 | MDL优势 |
|---------|--------|---------|
| 高度规律（正弦） | 175x | 9943 units |
| 中度规律（多项式） | 131x | 9924 units |
| 低度规律（复杂） | 131x | 9924 units |

## 项目结构

```
.
├── generative_encoding.py  # 核心框架实现
├── examples.py             # 实践示例
├── theory.md              # 理论文档
├── index.html             # 在线文档
├── requirements.txt       # Python依赖
└── README.md             # 本文件
```

## 许可证

MIT License

## 引用

如果您在研究中使用本框架，请引用：

```bibtex
@misc{generative_encoding_2025,
  title={Generative Encoding: A Novel Data Representation Paradigm based on MDL Principle},
  author={Zhenpeng},
  year={2025},
  url={https://github.com/Zhenpeng1979/Zhenpeng1979.github.io}
}
```

## 联系方式

- GitHub: [@Zhenpeng1979](https://github.com/Zhenpeng1979)
- 网站: [zhenpeng1979.github.io](https://zhenpeng1979.github.io)

---

*生成编码 - 将信息编码为规律，而非数据点*
