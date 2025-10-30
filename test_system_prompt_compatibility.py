#!/usr/bin/env python3
"""
测试新版本 System Prompt 与系统各组件的兼容性
"""

import sys
sys.path.insert(0, '/data2/yixuan/Temporary/EasyR1_upload_10_19')

from verl.utils.instruction_extractor_enhanced import InstructionExtractorEnhanced
from verl.workers.reward.rule_based_format_reward_manager import RuleBasedFormatRewardManager
from verl.workers.reward.config import RewardConfig


def test_multi_tag_extraction():
    """测试多标签提取"""
    print("\n" + "="*80)
    print("测试1: 多标签提取和拼接")
    print("="*80)
    
    # 创建增强版提取器
    extractor = InstructionExtractorEnhanced(
        enable_multi_tag=True,
        concatenation_separator="; ",
        fallback_instruction="Improve the image quality and consistency"
    )
    
    # 测试用例：新版本 system prompt 的期望输出格式
    test_cases = [
        {
            "name": "单个 Re_edit",
            "output": """<CoT>The lighting on the added person is inconsistent with the sunny background.</CoT>
<Re_edit>Adjust the lighting on the person to match the sun direction.</Re_edit>"""
        },
        {
            "name": "多个 Re_edit (2个)",
            "output": """<CoT>The lighting is inconsistent.</CoT>
<CoT>The shadow direction contradicts the main light source.</CoT>
<Re_edit>Adjust the lighting on the person to match the sun direction.</Re_edit>
<Re_edit>Add a consistent shadow extending to the left.</Re_edit>"""
        },
        {
            "name": "多个 Re_edit (3个) - 复杂案例",
            "output": """<CoT>The reflected figure of the person on the shiny floor is not vertically aligned with the real figure — it leans slightly to the right instead of mirroring directly below the feet.</CoT>
<CoT>The reflection starts a few pixels away from the soles, creating a visible gap that breaks mirror symmetry.</CoT>
<CoT>The reflection appears shorter than the person, suggesting incorrect scaling during the mirroring process.</CoT>
<Re_edit>Realign the reflection vertically so that it mirrors the person exactly beneath their feet, ensuring perfect symmetry along the floor plane.</Re_edit>
<Re_edit>Remove the gap between the soles and the start of the reflection by adjusting the pivot point to the exact contact line.</Re_edit>
<Re_edit>Rescale the reflection to match the full height of the person, maintaining a true 1:1 mirror ratio.</Re_edit>"""
        }
    ]
    
    all_passed = True
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─'*80}")
        print(f"案例 {i}: {test['name']}")
        print(f"{'─'*80}")
        
        # 提取指令
        instruction = extractor.extract_re_edit_instruction(test['output'])
        
        # 统计标签数量
        re_edit_count = test['output'].count('<Re_edit>')
        cot_count = test['output'].count('<CoT>')
        
        print(f"CoT 数量: {cot_count}")
        print(f"Re_edit 数量: {re_edit_count}")
        print(f"\n提取结果:")
        print(f">>> {instruction}")
        
        # 验证
        if re_edit_count == 1:
            expected_separator = False
        else:
            expected_separator = '; ' in instruction
        
        if re_edit_count > 1 and not expected_separator:
            print(f"\n❌ 测试失败: 多个Re_edit未被正确拼接")
            all_passed = False
        elif instruction == extractor.fallback_instruction and re_edit_count > 0:
            print(f"\n❌ 测试失败: 提取失败，返回了fallback")
            all_passed = False
        else:
            print(f"\n✅ 测试通过")
    
    return all_passed


def test_format_reward_compatibility():
    """测试格式奖励管理器兼容性"""
    print("\n" + "="*80)
    print("测试2: 格式奖励管理器兼容性")
    print("="*80)
    
    # 创建配置
    config = RewardConfig()
    config.rule_re_edit_max_length_ideal = 50
    config.rule_re_edit_max_length_acceptable = 100
    config.rule_re_edit_max_length_tolerable = 150
    
    # 创建奖励管理器
    manager = RuleBasedFormatRewardManager(config)
    
    # 测试用例
    test_cases = [
        {
            "name": "单个 CoT + 单个 Re_edit",
            "output": """<CoT>The lighting is inconsistent.</CoT>
<Re_edit>Adjust the lighting.</Re_edit>""",
            "expected_pass": True
        },
        {
            "name": "多个 CoT (3个) + 单个 Re_edit",
            "output": """<CoT>The lighting is inconsistent.</CoT>
<CoT>The shadow direction is wrong.</CoT>
<CoT>The color temperature is off.</CoT>
<Re_edit>Adjust the lighting and shadows.</Re_edit>""",
            "expected_pass": True
        },
        {
            "name": "单个 CoT + 多个 Re_edit (3个)",
            "output": """<CoT>Multiple issues detected in the reflection.</CoT>
<Re_edit>Realign the reflection vertically.</Re_edit>
<Re_edit>Remove the gap between soles and reflection.</Re_edit>
<Re_edit>Rescale the reflection to match the person.</Re_edit>""",
            "expected_pass": True
        },
        {
            "name": "多个 CoT (3个) + 多个 Re_edit (3个)",
            "output": """<CoT>The reflected figure is not vertically aligned.</CoT>
<CoT>The reflection starts away from the soles.</CoT>
<CoT>The reflection appears shorter than the person.</CoT>
<Re_edit>Realign the reflection vertically.</Re_edit>
<Re_edit>Remove the gap at the contact point.</Re_edit>
<Re_edit>Rescale the reflection height.</Re_edit>""",
            "expected_pass": True
        }
    ]
    
    all_passed = True
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─'*80}")
        print(f"案例 {i}: {test['name']}")
        print(f"{'─'*80}")
        
        # 评估
        result = manager.image_edit_format_reward(test['output'])
        
        # 统计标签数量
        cot_count = test['output'].count('<CoT>')
        re_edit_count = test['output'].count('<Re_edit>')
        
        print(f"CoT 数量: {cot_count}")
        print(f"Re_edit 数量: {re_edit_count}")
        print(f"硬性要求通过: {result['hard_requirements_passed']}")
        print(f"最终得分: {result['overall_score']:.2f}")
        print(f"详细信息:")
        print(f"  - Re_edit 平均长度: {result['re_edit_content_length']}")
        print(f"  - 长度惩罚: {result['re_edit_length_penalty']:.2f}")
        print(f"  - 独立成行惩罚: {result['line_separation_penalty']:.2f}")
        
        # 验证
        if result['hard_requirements_passed'] != test['expected_pass']:
            print(f"\n❌ 测试失败: 硬性要求检查结果不符合预期")
            all_passed = False
        elif result['cot_count'] != cot_count:
            print(f"\n❌ 测试失败: CoT计数不正确 (期望{cot_count}, 得到{result['cot_count']})")
            all_passed = False
        elif result['re_edit_count'] != re_edit_count:
            print(f"\n❌ 测试失败: Re_edit计数不正确 (期望{re_edit_count}, 得到{result['re_edit_count']})")
            all_passed = False
        else:
            print(f"\n✅ 测试通过")
    
    return all_passed


def test_end_to_end_flow():
    """测试端到端流程"""
    print("\n" + "="*80)
    print("测试3: 端到端流程（提取 + 评分）")
    print("="*80)
    
    # 创建组件
    extractor = InstructionExtractorEnhanced(
        enable_multi_tag=True,
        concatenation_separator="; "
    )
    
    config = RewardConfig()
    config.rule_re_edit_max_length_ideal = 50
    config.rule_re_edit_max_length_acceptable = 100
    config.rule_re_edit_max_length_tolerable = 150
    manager = RuleBasedFormatRewardManager(config)
    
    # 复杂案例
    model_output = """<CoT>The reflected figure of the person on the shiny floor is not vertically aligned with the real figure — it leans slightly to the right instead of mirroring directly below the feet.</CoT>
<CoT>The reflection starts a few pixels away from the soles, creating a visible gap that breaks mirror symmetry.</CoT>
<CoT>The reflection appears shorter than the person, suggesting incorrect scaling during the mirroring process.</CoT>
<Re_edit>Realign the reflection vertically so that it mirrors the person exactly beneath their feet, ensuring perfect symmetry along the floor plane.</Re_edit>
<Re_edit>Remove the gap between the soles and the start of the reflection by adjusting the pivot point to the exact contact line.</Re_edit>
<Re_edit>Rescale the reflection to match the full height of the person, maintaining a true 1:1 mirror ratio.</Re_edit>"""
    
    print(f"\n模型输出:")
    print(model_output)
    
    # 提取指令
    extracted = extractor.extract_re_edit_instruction(model_output)
    print(f"\n提取的编辑指令:")
    print(f">>> {extracted}")
    
    # 评分
    score_result = manager.image_edit_format_reward(model_output)
    print(f"\n格式奖励评分:")
    print(f"  - 硬性要求通过: {score_result['hard_requirements_passed']}")
    print(f"  - CoT 数量: {score_result['cot_count']}")
    print(f"  - Re_edit 数量: {score_result['re_edit_count']}")
    print(f"  - Re_edit 平均长度: {score_result['re_edit_content_length']}")
    print(f"  - 最终得分: {score_result['overall_score']:.2f} / 5.0")
    
    # 验证
    print(f"\n验证结果:")
    checks = [
        ("提取了3个Re_edit指令", extracted.count(';') == 2),
        ("格式奖励通过硬性要求", score_result['hard_requirements_passed']),
        ("识别到3个CoT", score_result['cot_count'] == 3),
        ("识别到3个Re_edit", score_result['re_edit_count'] == 3),
        ("得分 > 0", score_result['overall_score'] > 0)
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    return all_passed


def compare_with_old_version():
    """对比新旧版本的信息增益"""
    print("\n" + "="*80)
    print("对比: 新旧版本信息增益")
    print("="*80)
    
    extractor = InstructionExtractorEnhanced(
        enable_multi_tag=True,
        concatenation_separator="; "
    )
    
    # 旧版本输出（受限）
    old_output = """<CoT>The reflection is misaligned.</CoT>
<Re_edit>Fix the reflection.</Re_edit>"""
    
    # 新版本输出（详细）
    new_output = """<CoT>The reflected figure of the person on the shiny floor is not vertically aligned with the real figure — it leans slightly to the right instead of mirroring directly below the feet.</CoT>
<CoT>The reflection starts a few pixels away from the soles, creating a visible gap that breaks mirror symmetry.</CoT>
<CoT>The reflection appears shorter than the person, suggesting incorrect scaling during the mirroring process.</CoT>
<Re_edit>Realign the reflection vertically so that it mirrors the person exactly beneath their feet, ensuring perfect symmetry along the floor plane.</Re_edit>
<Re_edit>Remove the gap between the soles and the start of the reflection by adjusting the pivot point to the exact contact line.</Re_edit>
<Re_edit>Rescale the reflection to match the full height of the person, maintaining a true 1:1 mirror ratio.</Re_edit>"""
    
    old_instruction = extractor.extract_re_edit_instruction(old_output)
    new_instruction = extractor.extract_re_edit_instruction(new_output)
    
    print(f"\n旧版本输出 (受限):")
    print(f"  CoT: 1个")
    print(f"  Re_edit: 1个")
    print(f"  提取指令: \"{old_instruction}\"")
    print(f"  指令字数: {len(old_instruction.split())} 词")
    print(f"  总字数: {len(old_output.split())} 词")
    
    print(f"\n新版本输出 (详细):")
    print(f"  CoT: 3个")
    print(f"  Re_edit: 3个")
    print(f"  提取指令: \"{new_instruction}\"")
    print(f"  指令字数: {len(new_instruction.split())} 词")
    print(f"  总字数: {len(new_output.split())} 词")
    
    instruction_gain = (len(new_instruction.split()) / len(old_instruction.split()) - 1) * 100
    total_gain = (len(new_output.split()) / len(old_output.split()) - 1) * 100
    
    print(f"\n信息增益:")
    print(f"  指令信息增益: +{instruction_gain:.0f}%")
    print(f"  总体信息增益: +{total_gain:.0f}%")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("System Prompt 兼容性测试")
    print("="*80)
    
    # 运行所有测试
    test1 = test_multi_tag_extraction()
    test2 = test_format_reward_compatibility()
    test3 = test_end_to_end_flow()
    compare_with_old_version()
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    results = [
        ("多标签提取和拼接", test1),
        ("格式奖励管理器兼容性", test2),
        ("端到端流程", test3)
    ]
    
    all_passed = all(result for _, result in results)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {status} - {name}")
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ 所有测试通过！系统已准备就绪，可以开始训练。")
    else:
        print("❌ 部分测试失败，请检查组件配置。")
    print("="*80)
    
    sys.exit(0 if all_passed else 1)

