#!/usr/bin/env python3
"""
Test script for enhanced instruction extractor with multi-tag support
"""

from verl.utils.instruction_extractor_enhanced import InstructionExtractorEnhanced


def test_single_tag():
    """Test extraction of single Re_edit tag"""
    print("\n" + "="*80)
    print("Test 1: Single Re_edit tag")
    print("="*80)
    
    model_output = """
<CoT>
Based on the original image, I observe that the lighting is too dim.
The preliminary edit has improved the brightness, but contrast needs work.
</CoT>
<Re_edit>Enhance the lighting contrast and adjust color saturation</Re_edit>
"""
    
    extractor = InstructionExtractorEnhanced(enable_multi_tag=True)
    instruction = extractor.extract_re_edit_instruction(model_output)
    
    print(f"Input:\n{model_output}")
    print(f"\nExtracted instruction:\n{instruction}")
    print(f"\nExpected: Single instruction extracted")
    print(f"Result: ✓ PASS" if "contrast" in instruction else "✗ FAIL")


def test_multiple_tags_concatenation():
    """Test extraction and concatenation of multiple Re_edit tags"""
    print("\n" + "="*80)
    print("Test 2: Multiple Re_edit tags with concatenation")
    print("="*80)
    
    model_output = """
<CoT>
The image needs several improvements:
1. Lighting is uneven
2. Colors are oversaturated
3. Details are blurry
</CoT>
<Re_edit>Improve the lighting contrast</Re_edit>
<Re_edit>Adjust the color saturation to natural levels</Re_edit>
<Re_edit>Enhance the sharpness and details</Re_edit>
"""
    
    extractor = InstructionExtractorEnhanced(
        enable_multi_tag=True,
        concatenation_separator="; "
    )
    instruction = extractor.extract_re_edit_instruction(model_output)
    
    print(f"Input:\n{model_output}")
    print(f"\nExtracted instruction:\n{instruction}")
    print(f"\nExpected: All three instructions concatenated with '; '")
    print(f"Result: ✓ PASS" if instruction.count(";") == 2 else "✗ FAIL")


def test_multiple_tags_as_list():
    """Test extraction of multiple tags as separate list items"""
    print("\n" + "="*80)
    print("Test 3: Multiple Re_edit tags extracted as list")
    print("="*80)
    
    model_output = """
<CoT>Analysis of the image...</CoT>
<Re_edit>Improve lighting contrast</Re_edit>
<Re_edit>Adjust color saturation</Re_edit>
<Re_edit>Enhance sharpness</Re_edit>
"""
    
    extractor = InstructionExtractorEnhanced(enable_multi_tag=True)
    instructions_list = extractor.extract_all_tags(model_output)
    
    print(f"Input:\n{model_output}")
    print(f"\nExtracted instructions as list:")
    for i, inst in enumerate(instructions_list, 1):
        print(f"  {i}. {inst}")
    
    print(f"\nExpected: 3 separate instructions")
    print(f"Result: ✓ PASS" if len(instructions_list) == 3 else "✗ FAIL")


def test_multiple_tags_disabled():
    """Test behavior when multi-tag is disabled"""
    print("\n" + "="*80)
    print("Test 4: Multiple tags with multi-tag disabled (backward compatible)")
    print("="*80)
    
    model_output = """
<CoT>Analysis...</CoT>
<Re_edit>First instruction</Re_edit>
<Re_edit>Second instruction</Re_edit>
<Re_edit>Third instruction</Re_edit>
"""
    
    extractor = InstructionExtractorEnhanced(enable_multi_tag=False)
    instruction = extractor.extract_re_edit_instruction(model_output)
    
    print(f"Input:\n{model_output}")
    print(f"\nExtracted instruction:\n{instruction}")
    print(f"\nExpected: Only first instruction extracted")
    print(f"Result: ✓ PASS" if instruction == "First instruction" else "✗ FAIL")


def test_custom_separator():
    """Test custom concatenation separator"""
    print("\n" + "="*80)
    print("Test 5: Custom concatenation separator")
    print("="*80)
    
    model_output = """
<Re_edit>Improve lighting</Re_edit>
<Re_edit>Adjust colors</Re_edit>
<Re_edit>Enhance details</Re_edit>
"""
    
    # Test with different separators
    separators = [", then ", " AND ", " | "]
    
    for sep in separators:
        extractor = InstructionExtractorEnhanced(
            enable_multi_tag=True,
            concatenation_separator=sep
        )
        instruction = extractor.extract_re_edit_instruction(model_output)
        print(f"\nSeparator: '{sep}'")
        print(f"Result: {instruction}")
        print(f"Contains separator: ✓" if sep in instruction else "✗")


def test_batch_processing():
    """Test batch processing of multiple model outputs"""
    print("\n" + "="*80)
    print("Test 6: Batch processing")
    print("="*80)
    
    model_outputs = [
        "<Re_edit>Improve lighting</Re_edit>",
        "<Re_edit>First step</Re_edit><Re_edit>Second step</Re_edit>",
        "Invalid output without tags",
        "<Re_edit>A</Re_edit><Re_edit>B</Re_edit><Re_edit>C</Re_edit>"
    ]
    
    extractor = InstructionExtractorEnhanced(
        enable_multi_tag=True,
        concatenation_separator="; ",
        fallback_instruction="FALLBACK"
    )
    
    instructions = extractor.extract_batch_instructions(model_outputs)
    
    print(f"Processing {len(model_outputs)} outputs:")
    for i, (output, instruction) in enumerate(zip(model_outputs, instructions), 1):
        print(f"\n  Output {i}: {output[:50]}...")
        print(f"  Extracted: {instruction}")
    
    print(f"\nExpected: 4 instructions with appropriate handling")
    print(f"Result: ✓ PASS" if len(instructions) == 4 else "✗ FAIL")


def test_comparison_with_original():
    """Compare with original extractor behavior"""
    print("\n" + "="*80)
    print("Test 7: Comparison - Enhanced vs Original Behavior")
    print("="*80)
    
    model_output = """
<Re_edit>Improve the lighting contrast</Re_edit>
<Re_edit>Adjust the color saturation</Re_edit>
<Re_edit>Enhance the sharpness</Re_edit>
"""
    
    # Original behavior (multi-tag disabled)
    extractor_original = InstructionExtractorEnhanced(enable_multi_tag=False)
    original_result = extractor_original.extract_re_edit_instruction(model_output)
    
    # Enhanced behavior (multi-tag enabled)
    extractor_enhanced = InstructionExtractorEnhanced(
        enable_multi_tag=True,
        concatenation_separator="; "
    )
    enhanced_result = extractor_enhanced.extract_re_edit_instruction(model_output)
    
    print(f"Original behavior (only first tag):")
    print(f"  {original_result}")
    print(f"\nEnhanced behavior (all tags concatenated):")
    print(f"  {enhanced_result}")
    print(f"\nEnhanced extracts more information: ✓" if len(enhanced_result) > len(original_result) else "✗")


def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*80)
    print("ENHANCED INSTRUCTION EXTRACTOR - TEST SUITE")
    print("="*80)
    
    test_single_tag()
    test_multiple_tags_concatenation()
    test_multiple_tags_as_list()
    test_multiple_tags_disabled()
    test_custom_separator()
    test_batch_processing()
    test_comparison_with_original()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    run_all_tests()

