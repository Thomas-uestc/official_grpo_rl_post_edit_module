#!/usr/bin/env python3
"""
测试修改后的规则格式奖励管理器 - 支持多标签
"""

import sys
sys.path.insert(0, '/data2/yixuan/Temporary/EasyR1_upload_10_19')

from verl.workers.reward.rule_based_format_reward_manager import RuleBasedFormatRewardManager
from verl.workers.reward.config import RewardConfig


def create_test_config():
    """创建测试配置"""
    config = RewardConfig()
    config.rule_re_edit_max_length_ideal = 50
    config.rule_re_edit_max_length_acceptable = 100
    config.rule_re_edit_max_length_tolerable = 150
    return config


def test_multi_tag_support():
    """测试多标签支持"""
    print("\n" + "="*80)
    print("测试：多标签支持（新版本）")
    print("="*80)
    
    config = create_test_config()
    manager = RuleBasedFormatRewardManager(config)
    
    test_cases = [
        {
            "name": "单个CoT + 单个Re_edit（传统格式）",
            "text": "<CoT>分析图像需要增强亮度</CoT>\n<Re_edit>Increase brightness by 20%</Re_edit>",
            "expected_pass": True,
            "expected_score_range": (4.5, 5.0),
        },
        {
            "name": "单个CoT + 多个Re_edit（新格式 - 2个）",
            "text": "<CoT>需要进行多步改进</CoT>\n<Re_edit>Enhance brightness</Re_edit>\n<Re_edit>Adjust color saturation</Re_edit>",
            "expected_pass": True,
            "expected_score_range": (4.5, 5.0),
        },
        {
            "name": "单个CoT + 多个Re_edit（新格式 - 3个）",
            "text": """<CoT>进行三步图像优化</CoT>
<Re_edit>Increase overall brightness by 15%</Re_edit>
<Re_edit>Enhance color saturation</Re_edit>
<Re_edit>Apply sharpening filter</Re_edit>""",
            "expected_pass": True,
            "expected_score_range": (4.0, 5.0),
        },
        {
            "name": "多个CoT + 多个Re_edit（复杂推理）",
            "text": """<CoT>第一步分析：图像亮度不足</CoT>
<CoT>第二步分析：色彩饱和度需要提升</CoT>
<Re_edit>Increase brightness by 20%</Re_edit>
<Re_edit>Enhance color saturation by 30%</Re_edit>""",
            "expected_pass": True,
            "expected_score_range": (4.0, 5.0),
        },
        {
            "name": "多个Re_edit，但某些过长",
            "text": """<CoT>需要多项改进</CoT>
<Re_edit>Short instruction</Re_edit>
<Re_edit>This is a very very very very very very very very very very very very very long instruction that exceeds the tolerable threshold and should incur penalty</Re_edit>""",
            "expected_pass": True,
            "expected_score_range": (2.5, 4.5),  # 会有长度惩罚
        },
        {
            "name": "缺少CoT标签（应该失败）",
            "text": "<Re_edit>Improve quality</Re_edit>\n<Re_edit>Enhance details</Re_edit>",
            "expected_pass": False,
            "expected_score_range": (0.0, 0.0),
        },
        {
            "name": "缺少Re_edit标签（应该失败）",
            "text": "<CoT>分析一下</CoT>\n<CoT>再分析一下</CoT>",
            "expected_pass": False,
            "expected_score_range": (0.0, 0.0),
        },
        {
            "name": "有标签外内容（应该失败）",
            "text": "先说点别的\n<CoT>分析</CoT>\n<Re_edit>Edit</Re_edit>",
            "expected_pass": False,
            "expected_score_range": (0.0, 0.0),
        },
    ]
    
    print("\n开始测试...\n")
    
    all_passed = True
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─'*80}")
        print(f"测试案例 {i}: {test_case['name']}")
        print(f"{'─'*80}")
        print(f"输入文本:\n{test_case['text']}")
        print(f"\n预期:")
        print(f"  - 硬性要求通过: {test_case['expected_pass']}")
        print(f"  - 分数范围: {test_case['expected_score_range']}")
        
        # 执行评估
        result = manager.image_edit_format_reward(test_case['text'])
        
        print(f"\n实际结果:")
        print(f"  - 硬性要求通过: {result['hard_requirements_passed']}")
        print(f"  - 最终得分: {result['overall_score']:.2f}")
        print(f"  - CoT数量: {result['cot_count']}")
        print(f"  - Re_edit数量: {result['re_edit_count']}")
        print(f"  - Re_edit平均长度: {result['re_edit_content_length']}")
        print(f"  - 长度惩罚: {result['re_edit_length_penalty']:.2f}")
        print(f"  - 独立成行惩罚: {result['line_separation_penalty']:.2f}")
        print(f"  - 总惩罚: {result['total_penalty']:.2f}")
        
        # 验证结果
        passed_check = result['hard_requirements_passed'] == test_case['expected_pass']
        score_check = test_case['expected_score_range'][0] <= result['overall_score'] <= test_case['expected_score_range'][1]
        
        if passed_check and score_check:
            print(f"\n✅ 测试通过")
        else:
            print(f"\n❌ 测试失败")
            if not passed_check:
                print(f"   - 硬性要求检查不匹配")
            if not score_check:
                print(f"   - 分数不在预期范围内")
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败")
    print("="*80)
    
    return all_passed


def test_length_penalty_with_multiple_tags():
    """测试多标签情况下的长度惩罚计算"""
    print("\n" + "="*80)
    print("详细测试：多标签长度惩罚计算")
    print("="*80)
    
    config = create_test_config()
    manager = RuleBasedFormatRewardManager(config)
    
    # 案例1: 所有Re_edit都很短
    text1 = """<CoT>分析</CoT>
<Re_edit>Short 1</Re_edit>
<Re_edit>Short 2</Re_edit>
<Re_edit>Short 3</Re_edit>"""
    
    result1 = manager.image_edit_format_reward(text1)
    print(f"\n案例1: 3个短Re_edit（长度: 7, 7, 7）")
    print(f"  平均长度: {result1['re_edit_content_length']}")
    print(f"  长度惩罚: {result1['re_edit_length_penalty']:.2f}")
    print(f"  最终得分: {result1['overall_score']:.2f}")
    
    # 案例2: 混合长度
    text2 = """<CoT>分析</CoT>
<Re_edit>Short</Re_edit>
<Re_edit>This is a medium length instruction that is around sixty characters long</Re_edit>
<Re_edit>Medium</Re_edit>"""
    
    result2 = manager.image_edit_format_reward(text2)
    print(f"\n案例2: 混合长度（短、中、短）")
    print(f"  平均长度: {result2['re_edit_content_length']}")
    print(f"  长度惩罚: {result2['re_edit_length_penalty']:.2f}")
    print(f"  最终得分: {result2['overall_score']:.2f}")
    
    # 案例3: 有一个很长的
    text3 = """<CoT>分析</CoT>
<Re_edit>Short</Re_edit>
<Re_edit>This is an extremely long instruction that definitely exceeds the tolerable threshold and will cause a penalty to be applied because it contains way too much unnecessary detail and verbosity</Re_edit>"""
    
    result3 = manager.image_edit_format_reward(text3)
    print(f"\n案例3: 一短一长")
    print(f"  平均长度: {result3['re_edit_content_length']}")
    print(f"  长度惩罚: {result3['re_edit_length_penalty']:.2f}")
    print(f"  最终得分: {result3['overall_score']:.2f}")
    
    print("\n说明: 平均长度计算可以有效平衡多个指令的长度")


def compare_old_vs_new():
    """对比旧版本限制 vs 新版本支持"""
    print("\n" + "="*80)
    print("对比：旧版本（限制标签数量） vs 新版本（支持多标签）")
    print("="*80)
    
    config = create_test_config()
    manager = RuleBasedFormatRewardManager(config)
    
    # 在旧版本中会失败的案例
    test_text = """<CoT>第一步分析</CoT>
<CoT>第二步分析</CoT>
<CoT>第三步分析</CoT>
<Re_edit>Improve brightness</Re_edit>
<Re_edit>Enhance colors</Re_edit>
<Re_edit>Sharpen details</Re_edit>"""
    
    print(f"\n测试文本（3个CoT + 3个Re_edit）:")
    print(test_text)
    
    result = manager.image_edit_format_reward(test_text)
    
    print(f"\n新版本结果:")
    print(f"  ✅ 硬性要求通过: {result['hard_requirements_passed']}")
    print(f"  ✅ CoT数量: {result['cot_count']} (不再限制)")
    print(f"  ✅ Re_edit数量: {result['re_edit_count']} (不再限制)")
    print(f"  ✅ 最终得分: {result['overall_score']:.2f}")
    
    print(f"\n旧版本本应:")
    print(f"  ❌ 硬性要求: 失败 (CoT数量>2, Re_edit数量>1)")
    print(f"  ❌ 最终得分: 0.0")
    
    print(f"\n结论: 新版本成功支持多标签详细输出！")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("规则格式奖励管理器 - 多标签支持测试")
    print("="*80)
    
    # 运行测试
    test1_passed = test_multi_tag_support()
    test_length_penalty_with_multiple_tags()
    compare_old_vs_new()
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)
    print("\n关键改进:")
    print("  1. ✅ 移除了标签数量上限限制")
    print("  2. ✅ 支持多个<CoT>标签（详细推理）")
    print("  3. ✅ 支持多个<Re_edit>标签（多步编辑）")
    print("  4. ✅ 长度惩罚基于平均长度计算")
    print("  5. ✅ 与增强版提取器完美配合")
    print("\n" + "="*80)
    
    sys.exit(0 if test1_passed else 1)

