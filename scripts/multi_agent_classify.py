#!/usr/bin/env python3
"""Multi-Agent Classification for Drug Safety Literature.

This module implements a multi-agent debate approach for classifying
medical literature for drug safety signals. Four specialized agents
collaborate to make classification decisions:

1. Pharmacologist Agent: Judges has_drug and has_ae
2. Clinician Agent: Judges has_causality and patient_mode
3. Analyst Agent: Judges article_type and has_special_situation
4. Arbitrator Agent: Synthesizes all judgments into final classification

Usage:
    CLASSIFY_MODE=multi_agent python scripts/wanfang_classify.py ...
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from openai import OpenAI


@dataclass
class AgentResult:
    """Result from a single agent."""
    agent_name: str
    judgments: dict[str, Any]
    reasoning: str
    confidence: float = 0.8


@dataclass
class MultiAgentResult:
    """Combined result from all agents."""
    pharmacologist: AgentResult
    clinician: AgentResult
    analyst: AgentResult
    arbitrator: AgentResult
    final_label: str
    final_label_cn: str
    has_drug: bool
    has_ae: bool
    has_causality: bool
    has_special_situation: bool
    patient_mode: str
    patient_max_n: int | None
    confidence: float
    needs_review: bool
    reasoning: str


# Label mappings
SAFETY_LABELS = {
    "Rejection": "拒绝 (缺少药物或AE)",
    "ICSR": "个例安全报告 (单患者)",
    "Multiple_Patients": "多患者报告 (>1例)",
    "ICSR+Multiple_Patients": "混合报告 (单+多患者)",
    "Other_Safety_Signal": "其他安全信号 (初筛)",
}


def _call_llm(prompt: str, system_prompt: str = "") -> dict:
    """Call LLM and parse JSON response."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    model = os.getenv("CLASSIFY_MODEL_NAME", "gpt-4o")
    is_reasoning_model = model.startswith("o1") or model.startswith("o3")

    messages = []
    if system_prompt and not is_reasoning_model:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    create_kwargs = {
        "model": model,
        "messages": messages,
    }

    if not is_reasoning_model:
        create_kwargs["temperature"] = 0
        create_kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**create_kwargs)
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def pharmacologist_agent(text: str, target_drug: str, drug_search_result: dict | None = None) -> AgentResult:
    """
    药物学专家 Agent：判断 has_drug 和 has_ae

    专长：
    - 识别目标药物在文中的提及和使用情况
    - 区分药物不良反应 vs 疾病本身症状
    - 区分人体AE vs 动物实验毒性
    """
    # 构建药物搜索信息
    drug_search_info = ""
    if drug_search_result and drug_search_result.get('found'):
        drug_search_info = f"""
## 药物搜索结果（预处理）
- 状态: ✅ 找到
- 出现次数: {drug_search_result.get('count', 0)}
- 匹配词: {', '.join(drug_search_result.get('matched_terms', [])[:5])}
- 上下文: {'; '.join(drug_search_result.get('contexts', [])[:2])}
"""
    elif drug_search_result:
        drug_search_info = f"""
## 药物搜索结果（预处理）
- 状态: ❌ 未找到
- 搜索词: {', '.join(drug_search_result.get('search_terms', [])[:5])}
"""

    prompt = f"""你是一位资深药物学专家。请分析这篇医学文献，判断目标药物和不良事件。

## 目标药物
{target_drug}
{drug_search_info}

## 你的任务

### 1. 判断 has_drug（目标药物是否在文中被提及）
- 搜索目标药物的中英文名、商品名、通用名
- 即使只是简单提及或背景介绍，也算 has_drug=True
- 注意OCR问题：中文可能有空格（如"卡 马 西 平"）

### 2. 判断 has_ae（是否存在与药物相关的不良事件）

⚠️ 关键区分：
| 算AE | 不算AE |
|------|--------|
| 病例报告中用药后出现的症状 | 疾病本身的症状（如肿瘤的腹泻） |
| 临床研究中记录的不良反应 | 动物实验中的毒性反应 |
| 具体患者的AE描述 | 综述中泛泛讨论的可能风险 |

## 文章内容
{text[:10000]}

## 请返回JSON
{{
    "has_drug": boolean,
    "has_drug_evidence": ["原文证据1", "原文证据2"],
    "has_drug_reasoning": "判断理由",
    "has_ae": boolean,
    "has_ae_evidence": ["原文证据1", "原文证据2"],
    "has_ae_reasoning": "判断理由，特别说明是否区分了疾病症状vs药物AE",
    "confidence": 0.0-1.0
}}"""

    result = _call_llm(prompt)

    return AgentResult(
        agent_name="pharmacologist",
        judgments={
            "has_drug": result.get("has_drug", False),
            "has_ae": result.get("has_ae", False),
            "has_drug_evidence": result.get("has_drug_evidence", []),
            "has_ae_evidence": result.get("has_ae_evidence", []),
        },
        reasoning=f"has_drug: {result.get('has_drug_reasoning', '')}\nhas_ae: {result.get('has_ae_reasoning', '')}",
        confidence=result.get("confidence", 0.7),
    )


def clinician_agent(text: str, has_ae: bool) -> AgentResult:
    """
    临床医生 Agent：判断 has_causality 和 patient_mode

    专长：
    - 评估药物与不良事件之间的因果关系
    - 识别患者数量和类型
    - 理解临床研究设计
    """
    prompt = f"""你是一位资深临床医生。请分析这篇医学文献的因果关系和患者情况。

## 前置判断
- has_ae（药物学专家已判断）: {has_ae}

## 你的任务

### 1. 判断 has_causality（药物与AE之间是否存在因果关系）

⚠️ 关键判断标准：
| 算因果关系 | 不算因果关系 |
|-----------|-------------|
| 病例报告中"用药后出现XX" | 仅列举可能的副作用 |
| 临床研究中的AE发生率统计 | 明确否定因果关系 |
| 停药后缓解、再用药复发 | 仅描述疾病自然病程 |
| 时间关联表述（治疗期间发生） | 综述讨论理论风险 |

### 2. 判断 patient_mode（患者模式）
- "single": 单个可识别患者（病例报告、个案）
- "multiple": 多个患者（队列研究、临床试验、回顾性分析）
- "mixed": 文章中既有单患者病例，又有多患者统计
- "unknown": 综述/指南，无明确患者信息

💡 提示：
- 标题含"1例"、"个案"、"病例报告" → 倾向 single
- "案例分享"类文章，每个病例是独立报告 → 倾向 single
- 有明确样本量（n=XX）→ 倾向 multiple

## 文章内容
{text[:10000]}

## 请返回JSON
{{
    "has_causality": boolean,
    "causality_evidence": ["原文证据1", "原文证据2"],
    "causality_reasoning": "判断理由",
    "patient_mode": "single|multiple|mixed|unknown",
    "patient_max_n": integer or null,
    "patient_evidence": ["原文证据1", "原文证据2"],
    "patient_reasoning": "判断理由",
    "confidence": 0.0-1.0
}}"""

    result = _call_llm(prompt)

    patient_max_n = result.get("patient_max_n")
    if patient_max_n is not None:
        try:
            patient_max_n = int(patient_max_n)
        except (ValueError, TypeError):
            patient_max_n = None

    return AgentResult(
        agent_name="clinician",
        judgments={
            "has_causality": result.get("has_causality", False),
            "patient_mode": result.get("patient_mode", "unknown"),
            "patient_max_n": patient_max_n,
            "causality_evidence": result.get("causality_evidence", []),
            "patient_evidence": result.get("patient_evidence", []),
        },
        reasoning=f"因果关系: {result.get('causality_reasoning', '')}\n患者模式: {result.get('patient_reasoning', '')}",
        confidence=result.get("confidence", 0.7),
    )


def analyst_agent(text: str, filename: str) -> AgentResult:
    """
    文献分析专家 Agent：判断文章类型和特殊情况

    专长：
    - 识别文章类型（综述/病例/临床研究/动物实验）
    - 识别特殊情况（药物无效、儿童用药、妊娠暴露等）
    """
    prompt = f"""你是一位医学文献分析专家。请分析这篇文献的类型和特殊情况。

## 文件名
{filename}

## 你的任务

### 1. 判断 article_type（文章类型）
- "case_report": 病例报告、个案、案例分享
- "clinical_study": 临床研究、临床试验、队列研究、回顾性分析
- "review": 综述、指南、述评、专家共识
- "animal_study": 动物实验、体外实验、细胞实验
- "unknown": 无法确定

💡 提示：
- 文件名前缀是目标药物，后面是文章标题
- 注意中文文章类型关键词

### 2. 判断 has_special_situation（是否存在特殊情况）

⚠️ 以下任一情况存在即为 True：

| 特殊情况 | 关键词 |
|---------|--------|
| 药物无效/疗效不佳 | 无效、治疗失败、控制不佳、换药、效果欠佳 |
| 儿童用药 | 患儿、小儿、儿童、婴儿、幼儿、新生儿、青少年 |
| 妊娠/哺乳期暴露 | 妊娠、孕妇、怀孕、哺乳、母乳、产妇 |
| 过量/中毒 | 过量、中毒、超剂量 |
| 用药错误 | 用药错误、给药错误、剂量错误 |
| 药物相互作用 | 药物相互作用、联合用药导致 |
| 超说明书用药 | 超说明书、超适应症、off-label |

## 文章内容
{text[:10000]}

## 请返回JSON
{{
    "article_type": "case_report|clinical_study|review|animal_study|unknown",
    "article_type_evidence": ["判断依据"],
    "article_type_reasoning": "判断理由",
    "has_special_situation": boolean,
    "special_types": ["具体是哪种特殊情况"],
    "special_evidence": ["原文证据1", "原文证据2"],
    "special_reasoning": "判断理由",
    "confidence": 0.0-1.0
}}"""

    result = _call_llm(prompt)

    return AgentResult(
        agent_name="analyst",
        judgments={
            "article_type": result.get("article_type", "unknown"),
            "has_special_situation": result.get("has_special_situation", False),
            "special_types": result.get("special_types", []),
            "article_type_evidence": result.get("article_type_evidence", []),
            "special_evidence": result.get("special_evidence", []),
        },
        reasoning=f"文章类型: {result.get('article_type_reasoning', '')}\n特殊情况: {result.get('special_reasoning', '')}",
        confidence=result.get("confidence", 0.7),
    )


def arbitrator_agent(
    pharmacologist_result: AgentResult,
    clinician_result: AgentResult,
    analyst_result: AgentResult,
) -> AgentResult:
    """
    仲裁专家 Agent：综合三方判断，解决分歧，给出最终分类

    职责：
    - 综合三位专家的判断
    - 识别并解决分歧
    - 应用分类规则给出最终标签
    """
    prompt = f"""你是药物安全分类仲裁专家。请综合以下三位专家的判断，给出最终分类。

## 药物学专家判断
- has_drug: {pharmacologist_result.judgments.get('has_drug')}
- has_ae: {pharmacologist_result.judgments.get('has_ae')}
- 推理: {pharmacologist_result.reasoning}
- 置信度: {pharmacologist_result.confidence}

## 临床医生判断
- has_causality: {clinician_result.judgments.get('has_causality')}
- patient_mode: {clinician_result.judgments.get('patient_mode')}
- patient_max_n: {clinician_result.judgments.get('patient_max_n')}
- 推理: {clinician_result.reasoning}
- 置信度: {clinician_result.confidence}

## 文献分析专家判断
- article_type: {analyst_result.judgments.get('article_type')}
- has_special_situation: {analyst_result.judgments.get('has_special_situation')}
- special_types: {analyst_result.judgments.get('special_types')}
- 推理: {analyst_result.reasoning}
- 置信度: {analyst_result.confidence}

## 分类规则

1. **Rejection**: 缺少drug 或 (缺少AE 且 缺少特殊情况)
2. **ICSR**: drug + (AE+因果 或 特殊情况) + 单患者(single)
3. **Multiple_Patients**: drug + (AE+因果 或 特殊情况) + 多患者(multiple)
4. **ICSR+Multiple_Patients**: 文章同时包含单患者病例报告和多患者数据，即patient_mode="mixed"时
5. **Other_Safety_Signal**: 其他情况（有安全价值但不满足上述条件）

## patient_mode判断规则
- **single**: 文章只报告1例患者（通过性别/年龄可区分的个例）
- **multiple**: 文章报告多例患者（如"3例"、"10%发生率"、纳入N例等）
- **mixed**: 文章同时包含独立的单患者病例和多患者统计数据 → 分类为ICSR+Multiple_Patients

## 特殊考虑

1. 如果 article_type 是 "animal_study" 或 "review"：
   - has_ae 应该更严格判断（动物毒性、综述讨论不算AE）
   - 如有分歧，倾向于 Rejection

2. 如果 article_type 是 "case_report"：
   - patient_mode 应倾向于 single
   - 隐含因果关系应被认可

3. 如果存在 special_situation（如药物无效、儿童用药）：
   - 即使没有传统AE，也可以构成安全信号

## 请给出最终判断

返回JSON:
{{
    "has_drug": boolean,
    "has_ae": boolean,
    "has_causality": boolean,
    "has_special_situation": boolean,
    "patient_mode": "single|multiple|mixed|unknown",
    "patient_max_n": integer or null,
    "label": "Rejection|ICSR|Multiple_Patients|ICSR+Multiple_Patients|Other_Safety_Signal",
    "confidence": 0.0-1.0,
    "disagreements": ["如有分歧，列出分歧点"],
    "resolution": "如何解决分歧",
    "final_reasoning": "最终判断的完整推理过程"
}}"""

    result = _call_llm(prompt)

    label = result.get("label", "Other_Safety_Signal")
    if label not in SAFETY_LABELS:
        label = "Other_Safety_Signal"

    return AgentResult(
        agent_name="arbitrator",
        judgments={
            "has_drug": result.get("has_drug", False),
            "has_ae": result.get("has_ae", False),
            "has_causality": result.get("has_causality", False),
            "has_special_situation": result.get("has_special_situation", False),
            "patient_mode": result.get("patient_mode", "unknown"),
            "patient_max_n": result.get("patient_max_n"),
            "label": label,
            "disagreements": result.get("disagreements", []),
            "resolution": result.get("resolution", ""),
        },
        reasoning=result.get("final_reasoning", ""),
        confidence=result.get("confidence", 0.7),
    )


def classify_with_multi_agent(
    text: str,
    filename: str,
    drug_keywords: list[str] | None = None,
    drug_search_result: dict | None = None,
    target_drug: str | None = None,
) -> MultiAgentResult:
    """
    使用 Multi-Agent 辩论方式进行分类。

    Args:
        text: 文章全文
        filename: 文件名（格式：目标药物-文章标题.pdf）
        drug_keywords: 药物关键词列表
        drug_search_result: 预处理的药物搜索结果
        target_drug: 目标药物名称

    Returns:
        MultiAgentResult: 包含所有 Agent 结果的分类结果
    """
    # 从文件名提取目标药物
    if target_drug is None:
        if "-" in filename:
            target_drug = filename.split("-")[0]
        else:
            target_drug = "(未知)"

    print("      📋 [1/4] 药物学专家分析中...")
    pharmacologist_result = pharmacologist_agent(text, target_drug, drug_search_result)

    print("      👨‍⚕️ [2/4] 临床医生分析中...")
    clinician_result = clinician_agent(text, pharmacologist_result.judgments.get("has_ae", False))

    print("      📚 [3/4] 文献分析专家分析中...")
    analyst_result = analyst_agent(text, filename)

    print("      ⚖️ [4/4] 仲裁专家综合判断中...")
    arbitrator_result = arbitrator_agent(pharmacologist_result, clinician_result, analyst_result)

    # 提取最终结果
    final_judgments = arbitrator_result.judgments
    label = final_judgments.get("label", "Other_Safety_Signal")

    patient_max_n = final_judgments.get("patient_max_n")
    if patient_max_n is not None:
        try:
            patient_max_n = int(patient_max_n)
        except (ValueError, TypeError):
            patient_max_n = None

    confidence = arbitrator_result.confidence

    # 构建综合推理
    reasoning = f"""## Multi-Agent 辩论结果

### 药物学专家 (置信度: {pharmacologist_result.confidence:.2f})
{pharmacologist_result.reasoning}

### 临床医生 (置信度: {clinician_result.confidence:.2f})
{clinician_result.reasoning}

### 文献分析专家 (置信度: {analyst_result.confidence:.2f})
{analyst_result.reasoning}

### 仲裁结论 (置信度: {arbitrator_result.confidence:.2f})
{arbitrator_result.reasoning}

分歧: {final_judgments.get('disagreements', [])}
解决: {final_judgments.get('resolution', '')}
"""

    return MultiAgentResult(
        pharmacologist=pharmacologist_result,
        clinician=clinician_result,
        analyst=analyst_result,
        arbitrator=arbitrator_result,
        final_label=label,
        final_label_cn=SAFETY_LABELS.get(label, "未知"),
        has_drug=final_judgments.get("has_drug", False),
        has_ae=final_judgments.get("has_ae", False),
        has_causality=final_judgments.get("has_causality", False),
        has_special_situation=final_judgments.get("has_special_situation", False),
        patient_mode=final_judgments.get("patient_mode", "unknown"),
        patient_max_n=patient_max_n,
        confidence=confidence,
        needs_review=confidence < 0.65,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    # 简单测试
    print("Multi-Agent Classification Module")
    print("Use with: CLASSIFY_MODE=multi_agent python scripts/wanfang_classify.py ...")
