#!/usr/bin/env python3
"""
Direct comparison between original and enhanced instruction extractors
"""

import sys
from verl.utils.instruction_extractor import InstructionExtractor
from verl.utils.instruction_extractor_enhanced import InstructionExtractorEnhanced


def compare_single_case(case_name: str, model_output: str):
    """Compare extraction results for a single case"""
    
    print(f"\n{'='*80}")
    print(f"Case: {case_name}")
    print(f"{'='*80}")
    
    print(f"\nModel Output:")
    print("-" * 80)
    print(model_output)
    print("-" * 80)
    
    # Original extractor
    original_extractor = InstructionExtractor()
    original_result = original_extractor.extract_re_edit_instruction(model_output)
    
    # Enhanced extractor (multi-tag enabled)
    enhanced_extractor = InstructionExtractorEnhanced(
        enable_multi_tag=True,
        concatenation_separator="; "
    )
    enhanced_result = enhanced_extractor.extract_re_edit_instruction(model_output)
    
    # Enhanced extractor (list mode)
    enhanced_list = enhanced_extractor.extract_all_tags(model_output)
    
    print(f"\n📌 Original Extractor Result:")
    print(f"   {original_result}")
    
    print(f"\n✨ Enhanced Extractor Result (Concatenated):")
    print(f"   {enhanced_result}")
    
    print(f"\n📋 Enhanced Extractor Result (List Mode):")
    for i, inst in enumerate(enhanced_list, 1):
        print(f"   {i}. {inst}")
    
    # Analysis
    print(f"\n📊 Analysis:")
    original_count = model_output.count("<Re_edit>")
    print(f"   • Total <Re_edit> tags in output: {original_count}")
    print(f"   • Original extracts: 1 instruction")
    print(f"   • Enhanced extracts: {len(enhanced_list)} instructions")
    
    if len(enhanced_list) > 1:
        info_gain = ((len(enhanced_list) - 1) / 1) * 100
        print(f"   • Information gain: +{info_gain:.0f}%")
        print(f"   • Status: 🎯 Enhanced version captures MORE information")
    else:
        print(f"   • Information gain: 0%")
        print(f"   • Status: ✓ Both versions equivalent (single tag)")


def run_comparison():
    """Run comparison for multiple test cases"""
    
    print("\n" + "="*80)
    print("INSTRUCTION EXTRACTOR COMPARISON")
    print("Original vs Enhanced Version")
    print("="*80)
    
    # Test Case 1: Single tag (baseline)
    case1 = """
<CoT>分析图像后发现需要改进光照</CoT>
<Re_edit>Improve the lighting contrast and brightness</Re_edit>
"""
    compare_single_case("Single Re_edit Tag (Baseline)", case1)
    
    # Test Case 2: Two tags
    case2 = """
<CoT>图像有两个主要问题</CoT>
<Re_edit>Enhance the lighting contrast</Re_edit>
<Re_edit>Adjust the color saturation to natural levels</Re_edit>
"""
    compare_single_case("Two Re_edit Tags", case2)
    
    # Test Case 3: Three tags (complex editing)
    case3 = """
<CoT>
经过分析，这张图片需要三步改进：
1. 首先需要修正光照不均的问题
2. 然后调整色彩平衡
3. 最后增强细节清晰度
</CoT>
<Re_edit>Improve the lighting uniformity across the entire image</Re_edit>
<Re_edit>Adjust white balance and color temperature to neutral</Re_edit>
<Re_edit>Apply selective sharpening to enhance important details</Re_edit>
"""
    compare_single_case("Three Re_edit Tags (Complex Multi-Step)", case3)
    
    # Test Case 4: Four tags (very detailed)
    case4 = """
<Re_edit>Increase overall brightness by 15%</Re_edit>
<Re_edit>Enhance contrast in the midtones</Re_edit>
<Re_edit>Boost color saturation for blues and greens</Re_edit>
<Re_edit>Apply subtle noise reduction in shadow areas</Re_edit>
"""
    compare_single_case("Four Re_edit Tags (Very Detailed)", case4)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY & RECOMMENDATION")
    print("="*80)
    print("""
📌 Original Extractor Behavior:
   • Always extracts ONLY the first <Re_edit> tag
   • Ignores all subsequent tags
   • Simple and predictable

✨ Enhanced Extractor Advantages:
   • Extracts ALL <Re_edit> tags
   • Concatenates them into a single comprehensive instruction
   • Provides list mode for step-by-step processing
   • Backward compatible (can disable multi-tag)
   • Captures MORE information from model outputs

🎯 Recommendation:
   • Use ENHANCED version for training (enable_multi_tag=True)
   • Allows model to express complex, multi-step editing intentions
   • Image Edit model can benefit from richer instructions
   • ~100-300% more information captured in multi-tag cases

⚙️ Configuration:
   extractor = InstructionExtractorEnhanced(
       enable_multi_tag=True,           # Enable multi-tag extraction
       concatenation_separator="; ",    # Use "; " for clarity
       fallback_instruction="Improve the image quality"
   )
""")
    
    print("="*80)


if __name__ == "__main__":
    run_comparison()

