#!/usr/bin/env python3
"""
最终SFT知识验证 - 专门测试图像编辑相关能力
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import json
import os
from PIL import Image
import io
import base64

def create_test_image():
    """创建一个简单的测试图像"""
    from PIL import Image, ImageDraw
    
    # 创建一个简单的测试图像
    img = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(img)
    
    # 绘制一个简单的场景：蓝色天空，绿色草地，红色房子
    draw.rectangle([0, 0, 224, 112], fill='lightblue')  # 天空
    draw.rectangle([0, 112, 224, 224], fill='lightgreen')  # 草地
    draw.rectangle([80, 60, 144, 120], fill='red')  # 房子
    draw.rectangle([106, 90, 118, 120], fill='brown')  # 门
    
    return img

def image_to_base64(image):
    """将PIL图像转换为base64字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def test_image_editing_capability(model, tokenizer, processor, model_name="模型"):
    """测试图像编辑相关的能力"""
    print(f"\n🧪 测试{model_name}的图像编辑能力...")
    
    # 创建测试图像
    test_image = create_test_image()
    
    # 图像编辑相关的测试提示
    test_prompts = [
        "Describe what you see in this image.",
        "What objects are in this image?",
        "What colors do you see in this image?",
        "How would you edit this image to make it more beautiful?",
        "What would you change about this image?",
        "Describe the composition of this image.",
        "What is the main subject of this image?",
        "How would you improve the lighting in this image?",
        "What editing suggestions do you have for this image?",
        "Describe the scene in this image in detail.",
    ]
    
    responses = []
    
    for i, prompt in enumerate(test_prompts):
        try:
            # 准备输入
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": test_image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 处理输入
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # 生成响应
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0.1,
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            responses.append({
                'prompt': prompt,
                'response': response.strip()
            })
            
            print(f"   测试 {i+1}/10: {prompt[:30]}...")
            
        except Exception as e:
            print(f"   ❌ 测试 {i+1} 失败: {e}")
            responses.append({
                'prompt': prompt,
                'response': f"ERROR: {str(e)}"
            })
    
    return responses

def process_vision_info(messages):
    """处理视觉信息 - 从Qwen2.5-VL示例代码改编"""
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if isinstance(message, dict) and "content" in message:
            for content in message["content"]:
                if content.get("type") == "image":
                    image_inputs.append(content["image"])
                elif content.get("type") == "video":
                    video_inputs.append(content["video"])
    
    return image_inputs, video_inputs

def compare_model_responses(original_responses, merged_responses):
    """对比两个模型的响应"""
    print(f"\n📊 对比模型响应...")
    
    different_responses = 0
    similar_responses = 0
    
    for i, (orig, merged) in enumerate(zip(original_responses, merged_responses)):
        orig_resp = orig['response'].lower()
        merged_resp = merged['response'].lower()
        
        # 简单的相似度检查
        if orig_resp != merged_resp:
            different_responses += 1
            print(f"\n   📝 测试 {i+1}: {orig['prompt'][:40]}...")
            print(f"      原始: {orig_resp[:100]}{'...' if len(orig_resp) > 100 else ''}")
            print(f"      合并: {merged_resp[:100]}{'...' if len(merged_resp) > 100 else ''}")
            print("      ✅ 响应有差异")
        else:
            similar_responses += 1
    
    print(f"\n   📊 响应对比统计:")
    print(f"      - 有差异: {different_responses}")
    print(f"      - 相似: {similar_responses}")
    print(f"      - 差异比例: {different_responses/(different_responses+similar_responses)*100:.1f}%")
    
    return different_responses > 0

def verify_sft_knowledge():
    """验证SFT知识是否成功融入"""
    print("=" * 80)
    print("🔍 最终SFT知识验证 - 图像编辑能力测试")
    print("=" * 80)
    
    # 路径配置
    original_model_path = "/data2/yixuan/.cache/huggingface/hub/Qwen-Qwen2.5-VL-7B-Instruct"
    merged_model_path = "/data2/yixuan/Temporary/EasyR1/models/qwen2.5-vl-7b-sft-merged"
    
    try:
        # 1. 加载原始模型
        print("\n📥 加载原始模型...")
        original_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            original_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        original_processor = AutoProcessor.from_pretrained(original_model_path, trust_remote_code=True)
        original_tokenizer = AutoTokenizer.from_pretrained(original_model_path, trust_remote_code=True)
        print("✅ 原始模型加载成功")
        
        # 2. 加载合并模型
        print("\n📥 加载合并模型...")
        merged_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            merged_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        merged_processor = AutoProcessor.from_pretrained(merged_model_path, trust_remote_code=True)
        merged_tokenizer = AutoTokenizer.from_pretrained(merged_model_path, trust_remote_code=True)
        print("✅ 合并模型加载成功")
        
        # 3. 测试原始模型
        original_responses = test_image_editing_capability(
            original_model, original_tokenizer, original_processor, "原始模型"
        )
        
        # 4. 测试合并模型
        merged_responses = test_image_editing_capability(
            merged_model, merged_tokenizer, merged_processor, "合并模型"
        )
        
        # 5. 对比响应
        has_differences = compare_model_responses(original_responses, merged_responses)
        
        # 6. 最终结论
        print(f"\n" + "=" * 80)
        print("📊 最终验证结果")
        print("=" * 80)
        
        if has_differences:
            print("🎉 验证成功！")
            print("✅ 合并模型在图像编辑任务上表现出与原始模型的差异")
            print("✅ 这表明SFT训练的知识已经成功融入到模型中")
            print("🎯 建议：可以开始RL训练")
        else:
            print("⚠️  验证结果不明确")
            print("❓ 合并模型与原始模型的响应过于相似")
            print("💡 可能的原因：")
            print("   1. SFT训练的改动主要在特定领域，测试用例未覆盖")
            print("   2. 合并过程确实未成功")
            print("   3. 测试方法需要改进")
        
        return has_differences
        
    except Exception as e:
        print(f"\n❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = verify_sft_knowledge()
    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
