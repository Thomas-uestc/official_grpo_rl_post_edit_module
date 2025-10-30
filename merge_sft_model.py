#!/usr/bin/env python3
"""
预先合并SFT AdaLoRA权重到基础模型
专门适配用户的AdaLoRA配置：init_r=16, target_r=8, full模式
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel, get_peft_model, AdaLoraConfig
import os
import json
import re
from pathlib import Path

def merge_sft_model():
    """合并SFT模型权重"""
    # 根据用户实际路径配置
    base_model_path = "/data2/yixuan/.cache/huggingface/hub/Qwen-Qwen2.5-VL-7B-Instruct"
    adalora_path = "/data2/yixuan/Temporary/Qwen2.5-VL/qwen-vl-finetune/result_cvpr/hps7b_long_cot_v2/hps7b_long_cot_22_1030_0950_bs4_lr1e-04_lrcos_ga4_r64-4_t0.2-0.7_qkv/checkpoint-3600"
    output_path = "/data2/yixuan/Temporary/EasyR1/models/qwen2.5-vl-7b-sft-merged-v2-long-cot"
    
    print("🚀 开始合并SFT AdaLoRA模型...")
    print(f"📥 基础模型: {base_model_path}")
    print(f"📥 SFT模型: {adalora_path}")
    print(f"📤 输出路径: {output_path}")
    
    try:
        # 验证路径存在
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"基础模型路径不存在: {base_model_path}")
        if not os.path.exists(adalora_path):
            raise FileNotFoundError(f"AdaLoRA checkpoint路径不存在: {adalora_path}")
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 加载基础模型
        print("📥 加载基础模型...")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2"  # 与训练时保持一致
        )
        print("✅ 基础模型加载成功")
        
        # 读取AdaLoRA配置
        print("📋 读取AdaLoRA配置...")
        adapter_config_path = os.path.join(adalora_path, "adapter_config.json")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        print(f"📋 AdaLoRA配置:")
        print(f"   - PEFT类型: {adapter_config.get('peft_type')}")
        print(f"   - 当前rank: {adapter_config.get('r')}")
        print(f"   - 初始rank: {adapter_config.get('init_r')}")
        print(f"   - 目标rank: {adapter_config.get('target_r')}")
        print(f"   - Alpha: {adapter_config.get('lora_alpha')}")
        print(f"   - Target模块: {adapter_config.get('target_modules')}")
        print(f"   - 训练步数: {adapter_config.get('total_step')}")
        print(f"   - tinit: {adapter_config.get('tinit')}, tfinal: {adapter_config.get('tfinal')}")
        
        # 直接从checkpoint加载PEFT模型（不重建配置）
        print("🔧 直接从checkpoint加载PEFT模型...")
        peft_model = PeftModel.from_pretrained(
            base_model, 
            adalora_path, 
            adapter_name="default",
            is_trainable=False  # 设置为不可训练，因为我们只是要合并
        )
        print("✅ PEFT模型加载成功")
        
        # 显示可训练参数统计
        print("📊 PEFT模型参数统计:")
        peft_model.print_trainable_parameters()
        
        # 合并权重
        print("🔄 合并权重...")
        merged_model = peft_model.merge_and_unload()
        print("✅ 权重合并完成")
        
        # 保存合并后的模型
        print("💾 保存合并后的模型...")
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        print("✅ 模型保存完成")
        
        # 复制tokenizer和processor
        print("📋 复制tokenizer和processor...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(output_path)
        
        processor = AutoProcessor.from_pretrained(base_model_path)
        processor.save_pretrained(output_path)
        print("✅ Tokenizer和processor保存完成")
        
        # 验证合并后的模型
        print("🔍 验证合并后的模型...")
        test_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            output_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        
        total_params = sum(p.numel() for p in test_model.parameters())
        trainable_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
        
        print(f"📊 合并后模型参数:")
        print(f"   - 总参数: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"   - 可训练参数: {trainable_params:,}")
        
        # 检查是否有PEFT配置残留
        has_peft = hasattr(test_model, 'peft_config')
        print(f"🎯 PEFT配置残留: {has_peft}")
        
        if not has_peft:
            print("🎉 合并成功！模型已转换为标准transformer")
        else:
            print("⚠️ 警告：模型仍包含PEFT配置")
        
        # 创建使用说明文件
        from datetime import datetime
        readme_content = f"""# Merged SFT Model

## 模型信息
- 基础模型: Qwen2.5-VL-7B-Instruct
- SFT方法: AdaLoRA (init_r=16, target_r=8)
- 训练步数: {adapter_config.get('total_step', 'N/A')}
- 合并时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 在RL训练中使用
在配置文件中设置：
```yaml
worker:
  actor:
    model:
      model_path: {output_path}
      use_peft: false
```

## 参数统计
- 总参数: {total_params:,} ({total_params/1e9:.2f}B)
- 可训练参数: {trainable_params:,}
"""
        
        with open(os.path.join(output_path, "README.md"), "w") as f:
            f.write(readme_content)
        
        print(f"\n✅ 合并完成！")
        print(f"📁 合并后模型路径: {output_path}")
        print(f"💡 现在可以在RL训练中使用此路径，无需PEFT配置")
        
        return output_path
        
    except Exception as e:
        print(f"❌ 合并失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("=" * 80)
    print("🚀 SFT AdaLoRA模型合并工具")
    print("🎯 专门适配 Qwen2.5-VL + AdaLoRA (init_r=16, target_r=8)")
    print("=" * 80)
    
    merged_path = merge_sft_model()
    
    if merged_path:
        print(f"\n" + "=" * 80)
        print(f"🎉 合并成功！")
        print(f"📁 合并后模型: {merged_path}")
        print(f"\n💡 使用方法：")
        print(f"   在RL训练配置中设置：")
        print(f"   worker.actor.model.model_path: {merged_path}")
        print(f"   worker.actor.model.use_peft: false")
        print(f"\n🔧 VERL训练命令示例：")
        print(f"   python verl/trainer/main.py config.yaml")
        print(f"=" * 80)
        return True
    else:
        print(f"\n❌ 合并失败")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
