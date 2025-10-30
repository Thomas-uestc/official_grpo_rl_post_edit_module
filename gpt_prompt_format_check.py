#!/usr/bin/env python3
"""
检查 GPT-4o API 五个维度的 prompt 语法和格式
"""

import re

def check_prompts():
    """检查所有维度的prompt"""
    
    # 从文件中读取
    with open('/data2/yixuan/Temporary/EasyR1_upload_10_19/verl/workers/reward/gpt41_reward_manager.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取所有维度的prompt
    dimensions = {
        "physical_geometric": (202, 244),
        "environment_context": (245, 287),
        "cultural_social": (288, 330),
        "logical_causal": (331, 373),
        "target_attribution": (374, 416)
    }
    
    lines = content.split('\n')
    
    print("="*80)
    print("GPT-4o API Prompt 格式检查报告")
    print("="*80)
    
    all_issues = []
    
    for dim_name, (start, end) in dimensions.items():
        print(f"\n{'─'*80}")
        print(f"维度: {dim_name}")
        print(f"{'─'*80}")
        
        dim_lines = lines[start-1:end]
        dim_text = '\n'.join(dim_lines)
        
        issues = []
        
        # 检查1: 减号符号一致性
        minus_chars = set()
        for i, line in enumerate(dim_lines, start=start):
            # 检查各种减号字符
            if '–' in line:  # en dash U+2013
                minus_chars.add(('–', 'en dash (U+2013)', i, line.strip()))
            if '−' in line:  # minus sign U+2212
                minus_chars.add(('−', 'minus sign (U+2212)', i, line.strip()))
            if re.search(r'\s-\s', line):  # hyphen-minus U+002D (with spaces)
                minus_chars.add(('-', 'hyphen-minus (U+002D)', i, line.strip()))
        
        if len(set(char for char, _, _, _ in minus_chars)) > 1:
            issues.append({
                "type": "⚠️  减号符号不一致",
                "detail": f"使用了 {len(set(char for char, _, _, _ in minus_chars))} 种不同的减号字符",
                "chars": list(set((char, name) for char, name, _, _ in minus_chars))
            })
        
        # 检查2: 分数范围声明
        score_range_pattern = r'0\.000\s+(?:and|to)\s+10\.000'
        if not re.search(score_range_pattern, dim_text):
            issues.append({
                "type": "❌ 分数范围缺失",
                "detail": "未找到 '0.000 and/to 10.000' 声明"
            })
        
        # 检查3: 基线分数声明
        baseline_pattern = r'baseline\s+5\.000|5\.000\s+\(neutral\)|starts?\s+at\s+5\.000'
        if not re.search(baseline_pattern, dim_text, re.IGNORECASE):
            issues.append({
                "type": "❌ 基线分数缺失",
                "detail": "未找到基线 5.000 声明"
            })
        
        # 检查4: 标准数量
        criteria_count = dim_text.count('. ') - dim_text.count('e.g.')
        # 简化计数：查找 "1." "2." 等模式
        numbered_items = len(re.findall(r'\n\s*\d+\.', dim_text))
        
        # 检查5: 输出格式说明
        if 'Output' not in dim_text and 'output' not in dim_text.lower():
            issues.append({
                "type": "⚠️  输出格式说明",
                "detail": "未明确找到 'Output' 关键词"
            })
        
        # 检查6: "no text" 或 "no extra text"
        if 'no text' not in dim_text.lower() and 'no extra text' not in dim_text.lower():
            issues.append({
                "type": "⚠️  输出限制说明",
                "detail": "未找到 'no text' 或 'no extra text'"
            })
        
        # 检查7: 三位小数说明
        if 'three decimal' not in dim_text.lower() and '3 decimal' not in dim_text.lower():
            issues.append({
                "type": "⚠️  小数位数说明",
                "detail": "未找到三位小数的说明"
            })
        
        # 检查8: 分隔符 ⸻
        separator_count = dim_text.count('⸻')
        if separator_count == 0:
            issues.append({
                "type": "⚠️  分隔符缺失",
                "detail": "未使用分隔符 ⸻"
            })
        
        # 输出结果
        if issues:
            print(f"\n发现 {len(issues)} 个问题:")
            for issue in issues:
                print(f"\n  {issue['type']}")
                print(f"    {issue['detail']}")
                if 'chars' in issue:
                    for char, name in issue['chars']:
                        print(f"      • '{char}' - {name}")
            all_issues.extend(issues)
        else:
            print(f"\n✅ 未发现格式问题")
        
        print(f"\n统计信息:")
        print(f"  • 编号项目数: {numbered_items}")
        print(f"  • 分隔符数量: {separator_count}")
        print(f"  • 总行数: {len(dim_lines)}")
    
    # 全局检查
    print(f"\n{'='*80}")
    print(f"全局检查")
    print(f"{'='*80}")
    
    # 检查所有维度是否使用相同的减号
    all_minus_types = set()
    for dim_name, (start, end) in dimensions.items():
        dim_lines = lines[start-1:end]
        dim_text = '\n'.join(dim_lines)
        if '–' in dim_text:
            all_minus_types.add('en dash (–)')
        if '−' in dim_text:
            all_minus_types.add('minus sign (−)')
        if re.search(r'\s-\s', dim_text):
            all_minus_types.add('hyphen-minus (-)')
    
    print(f"\n跨维度减号使用情况:")
    if len(all_minus_types) > 1:
        print(f"  ⚠️  不同维度使用了不同的减号符号:")
        for minus_type in all_minus_types:
            print(f"    • {minus_type}")
        print(f"\n  建议: 统一使用标准减号 '-' (hyphen-minus U+002D)")
    else:
        print(f"  ✅ 所有维度使用相同的减号符号")
    
    # 总结
    print(f"\n{'='*80}")
    print(f"检查总结")
    print(f"{'='*80}")
    
    if all_issues:
        print(f"\n⚠️  发现 {len(all_issues)} 个格式问题")
        print(f"\n主要问题:")
        print(f"  1. 减号符号不一致（使用了多种Unicode字符）")
        print(f"  2. 建议统一使用标准的 '-' 或保持当前符号一致")
        print(f"\n这些问题可能影响:")
        print(f"  • GPT-4o 的解析（虽然通常能容错）")
        print(f"  • 代码的可维护性")
        print(f"  • 跨平台显示的一致性")
    else:
        print(f"\n✅ 所有维度的prompt格式良好")
    
    return all_issues


def detailed_minus_check():
    """详细检查减号字符使用"""
    print(f"\n{'='*80}")
    print(f"详细减号字符分析")
    print(f"{'='*80}")
    
    with open('/data2/yixuan/Temporary/EasyR1_upload_10_19/verl/workers/reward/gpt41_reward_manager.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    dimensions = {
        "physical_geometric": (202, 244),
        "environment_context": (245, 287),
        "cultural_social": (288, 330),
        "logical_causal": (331, 373),
        "target_attribution": (374, 416)
    }
    
    for dim_name, (start, end) in dimensions.items():
        print(f"\n{dim_name}:")
        
        minus_usage = {
            'en_dash': [],
            'minus_sign': [],
            'hyphen': []
        }
        
        for i in range(start-1, end):
            line = lines[i]
            line_num = i + 1
            
            if '–' in line:
                minus_usage['en_dash'].append((line_num, line.strip()[:80]))
            if '−' in line:
                minus_usage['minus_sign'].append((line_num, line.strip()[:80]))
            if re.search(r'\s-\d', line):  # 查找 " -3" 这样的模式
                minus_usage['hyphen'].append((line_num, line.strip()[:80]))
        
        if minus_usage['en_dash']:
            print(f"  En dash (–): {len(minus_usage['en_dash'])} 处")
            for line_num, text in minus_usage['en_dash'][:2]:
                print(f"    行 {line_num}: {text}")
        
        if minus_usage['minus_sign']:
            print(f"  Minus sign (−): {len(minus_usage['minus_sign'])} 处")
            for line_num, text in minus_usage['minus_sign'][:2]:
                print(f"    行 {line_num}: {text}")
        
        if minus_usage['hyphen']:
            print(f"  Hyphen-minus (-): {len(minus_usage['hyphen'])} 处")
            for line_num, text in minus_usage['hyphen'][:2]:
                print(f"    行 {line_num}: {text}")


if __name__ == "__main__":
    issues = check_prompts()
    detailed_minus_check()
    
    print(f"\n{'='*80}")
    print(f"建议的修复方案")
    print(f"{'='*80}")
    print(f"""
1. 统一减号符号:
   当前: 混合使用 '–' (en dash) 和 '−' (minus sign)
   建议: 统一使用 '-' (标准减号)
   
   查找替换:
   '–' → '-'  (en dash 转为 hyphen-minus)
   '−' → '-'  (minus sign 转为 hyphen-minus)

2. 保持格式一致性:
   • 所有维度使用相同的分隔符 ⸻
   • 所有维度使用相同的评分范围格式
   • 所有维度使用相同的输出说明

3. 当前prompt虽有小问题，但不影响GPT-4o理解:
   • GPT-4o 能够处理各种Unicode字符
   • 主要问题是代码可维护性和一致性
   
4. 是否需要修复:
   • 如果prompt已经测试过且工作正常 → 可以不改
   • 如果追求代码规范和一致性 → 建议统一符号
""")

