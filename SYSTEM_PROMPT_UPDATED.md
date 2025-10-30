# System Prompt 更新说明

## ✅ 最终版本：XML 标签格式（不限制数量）

**文件**: `verl/trainer/datasets/image_edit_dataset.py`  
**更新时间**: 2025-10-30

---

## 📋 新 System Prompt 特点

### **格式**: XML 标签（传统格式）
```xml
<CoT>第一步推理...</CoT>
<CoT>第二步推理...</CoT>
<CoT>第三步推理...</CoT>
<Re_edit>第一个编辑指令...</Re_edit>
<Re_edit>第二个编辑指令...</Re_edit>
<Re_edit>第三个编辑指令...</Re_edit>
```

### **核心优势**:
- ✅ **不限制 CoT 数量**: 根据需要提供详细推理
- ✅ **不限制 Re_edit 数量**: 支持多步编辑指令
- ✅ **不限制长度**: 允许详细描述
- ✅ **每个标签独立成行**: 清晰的结构
- ✅ **兼容现有系统**: 与增强版提取器和奖励管理器完美配合

---

## 🔄 版本对比

### **版本 1: 原始版本（受限）**

```xml
<CoT>The image successfully follows the instruction.</CoT>
<CoT>The lighting appears realistic.</CoT>
<Re_edit>Adjust the brightness of the background.</Re_edit>
```

**限制**:
- ❌ CoT: 最多 2 个
- ❌ Re_edit: 必须恰好 1 个
- ❌ 每行 ≤30 词
- ❌ 无法充分表达复杂问题

---

### **版本 2: 当前版本（无限制）**

```xml
<CoT>The reflected figure of the person on the shiny floor is not vertically aligned with the real figure — it leans slightly to the right instead of mirroring directly below the feet.</CoT>
<CoT>The reflection starts a few pixels away from the soles, creating a visible gap that breaks mirror symmetry.</CoT>
<CoT>The reflection appears shorter than the person, suggesting incorrect scaling during the mirroring process.</CoT>
<Re_edit>Realign the reflection vertically so that it mirrors the person exactly beneath their feet, ensuring perfect symmetry along the floor plane.</Re_edit>
<Re_edit>Remove the gap between the soles and the start of the reflection by adjusting the pivot point to the exact contact line.</Re_edit>
<Re_edit>Rescale the reflection to match the full height of the person, maintaining a true 1:1 mirror ratio.</Re_edit>
```

**优势**:
- ✅ CoT: 无数量限制（示例中3个）
- ✅ Re_edit: 无数量限制（示例中3个）
- ✅ 无长度限制
- ✅ 可以充分表达复杂问题和解决方案

---

## 📊 关键差异对照表

| 特性 | 原始版本 | 当前版本 |
|------|---------|---------|
| **输出格式** | XML 标签 | XML 标签 ✅ 相同 |
| **CoT 数量** | 1-2 个（强制限制） | 无限制 ✅ |
| **Re_edit 数量** | 必须 1 个 | 无限制 ✅ |
| **长度限制** | 每行 ≤30 词 | 无限制 ✅ |
| **详细程度** | 简洁（受限） | 详细（自由） ✅ |
| **表达复杂问题** | 困难 | 容易 ✅ |
| **提取器兼容** | 需要修改 | 完全兼容 ✅ |

---

## 🎯 完整 System Prompt

```python
system_prompt = (
    "You are a helpful assistant for visual thinking, design, and editing. "
    "Given a source image, an editing instruction, and the resulting edited image, do two tasks: "
    
    "1. Provide step-by-step reasoning for all categories where issues exist: "
    "(a) visual realism (geometry, lighting, physics)(e.g., the image in the mirror does not match the actual situation.), "
    "(b) contextual consistency (scene logic, attribute coherence), "
    "(c) environmental consistency (e.g., sunny sky but wet ground), "
    "(d) cultural/traditional consistency (e.g., Japanese wedding with Western dress). "
    "Skip categories without issues. "
    "The number of reasoning points is not limited — include as many as needed for clarity. "
    
    "2. Suggest re-editing instructions that are directly based on and summarized from the step-by-step CoT reasoning. "
    "Each re-edit instruction should correspond to one or more CoT points. "
    "The number and length of re-editing instructions are not limited. "
    "Each should describe a clear, executable editing action derived from your reasoning. "
    
    "\n"
    "OUTPUT FORMAT (STRICT): "
    "Use XML-style tags with each tag on its own separate line. "
    "Format: <CoT>content</CoT> for reasoning and <Re_edit>content</Re_edit> for instructions. "
    "Each tag MUST be on its own line with NO other content on that line. "
    
    "\n"
    "Examples:\n"
    "Example 1:\n"
    "<CoT>The lighting on the added person is inconsistent with the sunny background.</CoT>\n"
    "<CoT>The shadow direction contradicts the main light source.</CoT>\n"
    "<Re_edit>Adjust the lighting on the person to match the sun direction.</Re_edit>\n"
    "<Re_edit>Add a consistent shadow extending to the left, matching the scene's sunlight angle.</Re_edit>\n"
    
    "\n"
    "Example 2:\n"
    "<CoT>The reflected figure of the person on the shiny floor is not vertically aligned with the real figure — it leans slightly to the right instead of mirroring directly below the feet.</CoT>\n"
    "<CoT>The reflection starts a few pixels away from the soles, creating a visible gap that breaks mirror symmetry.</CoT>\n"
    "<CoT>The reflection appears shorter than the person, suggesting incorrect scaling during the mirroring process.</CoT>\n"
    "<Re_edit>Realign the reflection vertically so that it mirrors the person exactly beneath their feet, ensuring perfect symmetry along the floor plane.</Re_edit>\n"
    "<Re_edit>Remove the gap between the soles and the start of the reflection by adjusting the pivot point to the exact contact line.</Re_edit>\n"
    "<Re_edit>Rescale the reflection to match the full height of the person, maintaining a true 1:1 mirror ratio.</Re_edit>\n"
    
    "\n"
    "RULES: "
    "- Output ONLY the tag blocks, each on its own line "
    "- No JSON, no code fences, no explanations, no extra text "
    "- Each <CoT> and <Re_edit> tag must be on a separate line "
    "- The number of <CoT> tags is unlimited (include as many reasoning steps as needed) "
    "- The number of <Re_edit> tags is unlimited (include as many editing instructions as needed) "
    "- No length restrictions on tag content "
    "- Use imperative voice in <Re_edit> tags (e.g., 'Adjust...', 'Remove...', 'Realign...') "
    "- CRITICAL: Each tag must start on a new line with NO preceding text "
    "- CRITICAL: Each tag must end on its line with NO following text "
)
```

---

## 🔧 系统兼容性

### **✅ 完全兼容的组件**

#### 1. **InstructionExtractorEnhanced**
```python
# verl/utils/instruction_extractor_enhanced.py
# 已支持提取多个 <Re_edit> 标签并自动拼接
extractor = InstructionExtractorEnhanced(
    enable_multi_tag=True,
    concatenation_separator="; "
)

# 示例输入:
# <Re_edit>Adjust lighting.</Re_edit>
# <Re_edit>Add shadow.</Re_edit>
# <Re_edit>Fix reflection.</Re_edit>

# 输出: "Adjust lighting; Add shadow; Fix reflection"
```

#### 2. **RuleBasedFormatRewardManager**
```python
# verl/workers/reward/rule_based_format_reward_manager.py
# 已修改为支持多标签
# - 不限制 CoT 数量（原来 1-2 个）
# - 不限制 Re_edit 数量（原来必须 1 个）
# - 长度惩罚基于平均长度计算
```

#### 3. **RayTrainer**
```python
# verl/trainer/ray_trainer.py
# 自动调用 InstructionExtractorEnhanced
# 提取并拼接所有 Re_edit 指令传递给图像编辑器
```

---

## 📈 预期训练效果提升

### **信息量对比**

**原始版本输出示例**:
```xml
<CoT>The reflection is misaligned.</CoT>
<Re_edit>Fix the reflection.</Re_edit>
```
- CoT 信息: ~5 词
- Re_edit 信息: ~3 词
- **总信息: ~8 词**

**当前版本输出示例**:
```xml
<CoT>The reflected figure of the person on the shiny floor is not vertically aligned with the real figure — it leans slightly to the right instead of mirroring directly below the feet.</CoT>
<CoT>The reflection starts a few pixels away from the soles, creating a visible gap that breaks mirror symmetry.</CoT>
<CoT>The reflection appears shorter than the person, suggesting incorrect scaling during the mirroring process.</CoT>
<Re_edit>Realign the reflection vertically so that it mirrors the person exactly beneath their feet, ensuring perfect symmetry along the floor plane.</Re_edit>
<Re_edit>Remove the gap between the soles and the start of the reflection by adjusting the pivot point to the exact contact line.</Re_edit>
<Re_edit>Rescale the reflection to match the full height of the person, maintaining a true 1:1 mirror ratio.</Re_edit>
```
- CoT 信息: ~75 词
- Re_edit 信息: ~70 词
- **总信息: ~145 词**

**信息增益**: **~1700%** ✨

---

## 💡 训练优势

### **1. 更深入的推理训练**
- 多步 CoT 帮助模型学习复杂推理链
- 涵盖多个维度的问题分析
- 提高模型的批判性思考能力

### **2. 更精确的编辑指令**
- 多个细粒度 Re_edit 指令
- 每个指令对应具体问题
- 提高图像编辑器的执行精度

### **3. 更全面的问题覆盖**
- 可以同时指出多个问题
- 不受数量限制的约束
- 更真实反映图像质量评估场景

---

## 🚀 使用方法

### **训练启动**
```bash
# 无需修改启动脚本
bash examples/qwen2_5_vl_7b_image_edit_combined_v3.sh
```

### **期望模型输出格式**
```xml
<CoT>第一步推理分析...</CoT>
<CoT>第二步推理分析...</CoT>
<CoT>第三步推理分析...</CoT>
<Re_edit>第一个编辑指令...</Re_edit>
<Re_edit>第二个编辑指令...</Re_edit>
<Re_edit>第三个编辑指令...</Re_edit>
```

### **自动处理流程**
```
模型输出 (多个标签)
        ↓
InstructionExtractorEnhanced
  → 提取所有 <Re_edit> 标签
  → 拼接: "指令1; 指令2; 指令3"
        ↓
RuleBasedFormatRewardManager
  → 评分: 基于标签数量和平均长度
        ↓
Image Edit Worker
  → 执行拼接后的多步指令
```

---

## 📊 实际案例对比

### **简单场景**

**原版本（受限）**:
```xml
<CoT>The lighting is inconsistent.</CoT>
<Re_edit>Adjust lighting.</Re_edit>
```

**新版本（详细）**:
```xml
<CoT>The lighting on the added person is inconsistent with the sunny background.</CoT>
<CoT>The shadow direction contradicts the main light source.</CoT>
<CoT>The color temperature doesn't match the warm sunlight.</CoT>
<Re_edit>Adjust the lighting on the person to match the sun direction and intensity.</Re_edit>
<Re_edit>Add a consistent shadow extending to the left, matching the scene's sunlight angle.</Re_edit>
<Re_edit>Correct the color temperature to align with the warm daylight illumination.</Re_edit>
```

---

## ✅ 检查清单

修改完成后，系统已经：

- [x] **System Prompt** 更新为 XML 格式（不限制数量）
- [x] **InstructionExtractor** 支持多标签提取和拼接
- [x] **RuleBasedFormatRewardManager** 支持多标签评分
- [x] **RayTrainer** 自动适配新格式
- [x] **向后兼容** 保持 XML 标签格式
- [x] **测试验证** 所有组件正常工作

---

## 📁 相关文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `verl/trainer/datasets/image_edit_dataset.py` | ✅ 已更新 | System prompt 改为 XML 格式（不限数量） |
| `verl/utils/instruction_extractor_enhanced.py` | ✅ 已适配 | 支持多标签提取和拼接 |
| `verl/workers/reward/rule_based_format_reward_manager.py` | ✅ 已适配 | 支持多标签评分 |
| `verl/trainer/ray_trainer.py` | ✅ 无需修改 | 自动使用增强提取器 |

---

**更新完成时间**: 2025-10-30  
**格式**: XML 标签（不限制数量）  
**兼容性**: ✅ 完全兼容现有系统  
**训练就绪**: ✅ 可立即启动训练

