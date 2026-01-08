#!/usr/bin/env python3
"""Wanfang Medical Paper Safety Classification Script.

This script classifies downloaded papers from Wanfang Medical database
using LLM (OpenAI GPT) for pharmacovigilance/drug safety classification.

文献检索业务基础流程：
在全文范围内以中英文商品名&活性成分名作为关键词进行检索，检索出本周期内上抛到
CNKI & Wanfang数据库中的文献。针对所有检索出来的文献进行人工审阅，识别文章中
是否提及任何诺华药相关安全病例或潜在信号。

Classification categories (药物安全分类):
- Rejection: 文章中缺少drug(诺华药)或AE(不良事件)任意一个要素
- ICSR: (drug+AE+因果关系+单个患者) OR (drug+特殊情况+单个患者)
- Multiple_Patients: (drug+AE+因果关系+多个患者) OR (drug+特殊情况+多个患者)
- ICSR+Multiple_Patients: 一篇文章同时满足ICSR和Multiple_Patients的条件
- Other_Safety_Signal: 不符合上面类型的都初筛成signal

Usage:
    # Classify all papers in data/papers/
    python scripts/wanfang_classify.py --drugs "替格瑞洛,ticagrelor"

    # With drug keywords file
    python scripts/wanfang_classify.py --drugs-file data/drug_keywords.txt

    # Specify custom directory
    python scripts/wanfang_classify.py --input-dir data/papers --drugs "药物名"
"""

import argparse
import csv
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# Multi-Agent Classification (optional)
try:
    from multi_agent_classify import classify_with_multi_agent, MultiAgentResult
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    MULTI_AGENT_AVAILABLE = False

# Load environment variables
load_dotenv()

# Directories
DATA_DIR = Path(__file__).parent.parent / "data"
PAPERS_DIR = DATA_DIR / "papers"
DEFAULT_OUTPUT = DATA_DIR / "classification_results.csv"
DEFAULT_DRUGS_FILE = DATA_DIR / "novartis_drugs.txt"


# Classification labels
SAFETY_LABELS = {
    "Rejection": "拒绝 (缺少药物或AE)",
    "ICSR": "个例安全报告 (单患者)",
    "Multiple_Patients": "多患者报告 (>1例)",
    "ICSR+Multiple_Patients": "混合报告 (同时有单患者和多患者)",
    "Other_Safety_Signal": "其他安全信号 (初筛)",
}

PATIENT_MODES = {"single", "multiple", "mixed", "unknown"}


@dataclass
class PatientInfo:
    mode: str  # single / multiple / mixed / unknown
    max_n: int | None
    evidence: list[str]


@dataclass
class ClassificationResult:
    filename: str
    label: str
    label_cn: str
    has_drug: bool
    has_ae: bool
    has_causality: bool
    has_special_situation: bool
    patient_mode: str
    patient_max_n: int | None
    confidence: float
    drug_evidence: list[str]
    ae_evidence: list[str]
    causality_evidence: list[str]
    special_evidence: list[str]
    patient_evidence: list[str]
    # 各字段的独立 reasoning
    has_drug_reasoning: str
    has_ae_reasoning: str
    has_causality_reasoning: str
    has_special_reasoning: str
    patient_reasoning: str
    reasoning: str
    needs_review: bool
    extract_method: str
    text_length: int
    classify_time: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    error: str = ""


# ============================================================
# V2 结构化抽取架构 (v15) - 数据结构定义
# ============================================================

@dataclass
class DocStruct:
    """文档结构化解析结果"""
    body_text: str           # 正文部分
    ref_text: str            # 参考文献部分
    author_text: str         # 作者信息部分
    abbrev_map: dict         # 缩写映射 {"CSA": "环孢素", ...}
    article_type: str        # 综述/动物/病例/临床
    target_drug: str         # 目标药物(从文件名提取)
    raw_text: str            # 原始全文


@dataclass
class DrugMention:
    """药物提及结构"""
    where: str              # body / ref / author
    context_type: str       # study / intro / discussion / citation
    mention_polarity: str   # used / not_used / history / background / unknown
    surface_name: str       # 原文表述
    normalized: str         # 标准化名称
    novartis_judgement: str # yes / no / unknown
    novartis_rule_hit: str  # step1~step5 命中的规则（空串表示未命中）
    non_novartis_reason: str  # 如判断非诺华药，说明原因
    evidence_span: str      # 原文片段


@dataclass
class AEEvent:
    """不良事件结构"""
    event_type: str         # normal_ae / special_situation / other_ae
    special_subtype: str    # 特殊情况子类型: 孕期暴露/LOE/overdose/medication_error/off_label/儿童用药/药物相互作用/疾病进展/其他
    description: str        # 事件描述
    related_drug: str       # 关联的药物名称(必须填写)
    related_drug_evidence: str  # 关联药物的原文证据
    is_target_drug_related: bool  # 是否与目标药相关（LLM初判，规则层复核）
    context_type: str       # study / case / citation / background / unknown
    context_evidence: str   # 上下文判断证据
    source: str             # table / text / unknown
    raw_count_in_table: int | None  # 表格中事件计数（如1）
    human_flag: str         # human / nonhuman / unclear
    timing_relation: str    # after_drug / unclear / none
    causality: str          # explicit / implicit / none / not_attributable_combo
    causality_evidence: str # 因果关系的原文证据
    patient_count: str      # single / multiple / unknown
    patient_evidence: str   # 患者数证据
    off_label: bool         # 是否超说明书
    off_label_product: str  # 如off_label为true，填写产品名
    whitelist_ok: bool      # off-label时是否在白名单
    evidence_span: str      # 原文片段


@dataclass
class ExtractionResult:
    """LLM结构化抽取结果"""
    drug_mentions: list[DrugMention]
    ae_events: list[AEEvent]
    raw_response: dict = field(default_factory=dict)


@dataclass
class AuditResult:
    """审计层结果"""
    triggered_rules: list[str]   # 触发的规则ID列表
    corrections: dict            # 修正内容 {field: (old, new, reason)}
    retry_needed: bool           # 是否需要补问
    retry_questions: list[str]   # 补问问题列表


@dataclass
class ClassificationResultV2:
    """V2版本分类结果（结构化抽取）"""
    filename: str
    label: str
    label_cn: str

    # 核心判断
    novartis_exposure: bool      # 是否有诺华药暴露
    has_qualifying_event: bool   # 是否有合格事件

    # 结构化抽取结果
    extraction: ExtractionResult
    audit: AuditResult

    # 决策依据
    confidence: float
    reasoning: str
    decision_path: str           # 决策路径描述

    # 元信息
    needs_review: bool
    extract_method: str
    text_length: int
    classify_time: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    error: str = ""


# ============================================================
# V2 结构化抽取架构 - Phase 2: 输入处理模块
# ============================================================

def clean_text_v2(text: str) -> str:
    """清洗PDF提取的文本，修复常见问题。

    处理:
    1. 断行修复：中文字符间的换行符
    2. 断词修复：英文单词中间的换行符
    3. 多余空白符清理
    """
    if not text:
        return ""

    # 1. 移除PDF提取中的页眉页脚噪音（常见模式）
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # 跳过纯页码行
        if re.match(r'^[\d\s·\-—]+$', stripped):
            continue
        # 跳过常见页眉模式
        if re.match(r'^(第\s*\d+\s*页|Page\s*\d+|Vol\.\s*\d+)', stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # 2. 修复中文字符间的断行（保留段落间的换行）
    # 如果前一个字符和后一个字符都是中文，移除中间的换行
    text = re.sub(r'([\u4e00-\u9fff])\n([\u4e00-\u9fff])', r'\1\2', text)

    # 3. 修复英文单词内的断行（连字符断词）
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # 4. 清理多余空白
    text = re.sub(r'[ \t]+', ' ', text)  # 多个空格/制表符 → 单空格
    text = re.sub(r'\n{3,}', '\n\n', text)  # 多余空行 → 双换行

    return text.strip()


def extract_abbreviations(text: str) -> dict[str, str]:
    """从文本中提取缩写映射。

    匹配模式:
    - "环孢素(CSA)" → {"CSA": "环孢素"}
    - "cyclosporin A (CSA)" → {"CSA": "cyclosporin A"}
    - "CSA（环孢素）" → {"CSA": "环孢素"}

    Returns:
        dict: 缩写 → 全称的映射
    """
    abbrev_map = {}

    # 模式1: 中文名(英文缩写) 或 英文名(缩写)
    # 例如: 环孢素(CSA), cyclosporin A (CSA)
    pattern1 = r'([\u4e00-\u9fff\w\s\-]+?)\s*[（\(]([A-Z][A-Z0-9\-]{1,10})[）\)]'
    for match in re.finditer(pattern1, text):
        full_name = match.group(1).strip()
        abbrev = match.group(2).strip()
        if len(full_name) >= 2 and abbrev not in abbrev_map:
            abbrev_map[abbrev] = full_name

    # 模式2: 缩写（中文全称）
    # 例如: CSA（环孢素）
    pattern2 = r'([A-Z][A-Z0-9\-]{1,10})\s*[（\(]([\u4e00-\u9fff]+)[）\)]'
    for match in re.finditer(pattern2, text):
        abbrev = match.group(1).strip()
        full_name = match.group(2).strip()
        if len(full_name) >= 2 and abbrev not in abbrev_map:
            abbrev_map[abbrev] = full_name

    return abbrev_map


def segment_document(text: str) -> tuple[str, str, str]:
    """将文档分割为正文、参考文献和作者信息部分。

    Args:
        text: 完整文档文本

    Returns:
        tuple: (body_text, ref_text, author_text)
    """
    # 默认值
    body_text = text
    ref_text = ""
    author_text = ""

    # 参考文献标记
    ref_patterns = [
        r'\n\s*参\s*考\s*文\s*献\s*\n',
        r'\n\s*References?\s*\n',
        r'\n\s*REFERENCES?\s*\n',
        r'\n\s*参考文献[:：]\s*\n',
        r'\n\s*\[参考文献\]\s*\n',
    ]

    ref_start = -1
    for pattern in ref_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            ref_start = match.start()
            break

    if ref_start > 0:
        body_text = text[:ref_start]
        ref_text = text[ref_start:]

    # 作者信息标记（通常在文章开头）
    # 寻找作者单位、通讯作者等信息
    author_patterns = [
        r'(作者单位[:：].+?)(?=\n\s*\n)',
        r'(通讯作者[:：].+?)(?=\n\s*\n)',
        r'(Author.+?affiliation.+?)(?=\n\s*\n)',
        r'(\d{6}\s+[\u4e00-\u9fff]+[省市].+?(?:医院|大学|研究院|中心).+?)(?=\n\s*\n)',
    ]

    for pattern in author_patterns:
        match = re.search(pattern, body_text[:3000], re.IGNORECASE | re.DOTALL)
        if match:
            author_text = match.group(1).strip()
            break

    return body_text.strip(), ref_text.strip(), author_text.strip()


def parse_document_v2(pdf_path: Path, target_drug: str | None = None) -> DocStruct:
    """解析PDF文档，返回结构化的文档信息。

    Args:
        pdf_path: PDF文件路径
        target_drug: 目标药物名称（可选，如不提供则从文件名提取）

    Returns:
        DocStruct: 结构化文档信息
    """
    filename = pdf_path.name

    # 提取目标药物
    if target_drug is None:
        target_drug = extract_target_drug_from_filename(filename) or ""

    # 提取文本
    raw_text, method = extract_pdf_text(pdf_path)

    if not raw_text.strip():
        return DocStruct(
            body_text="",
            ref_text="",
            author_text="",
            abbrev_map={},
            article_type="unknown",
            target_drug=target_drug,
            raw_text=""
        )

    # 清洗文本
    cleaned_text = clean_text_v2(raw_text)

    # 分割文档
    body_text, ref_text, author_text = segment_document(cleaned_text)

    # 提取缩写映射
    abbrev_map = extract_abbreviations(cleaned_text)

    # 检测文章类型
    article_type_result = detect_article_type(cleaned_text, filename)
    article_type = article_type_result.get('type', 'unknown')

    return DocStruct(
        body_text=body_text,
        ref_text=ref_text,
        author_text=author_text,
        abbrev_map=abbrev_map,
        article_type=article_type,
        target_drug=target_drug,
        raw_text=cleaned_text
    )


# ============================================================
# V2 结构化抽取架构 - Phase 3: LLM结构化抽取模块
# ============================================================

def build_extraction_prompt_v2(doc: DocStruct, drug_keywords: list[str]) -> tuple[str, str]:
    """构建V2版本的结构化抽取prompt。

    Args:
        doc: 结构化文档信息
        drug_keywords: 药物关键词列表

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    drug_hint = ", ".join(drug_keywords[:50]) if drug_keywords else "(未提供)"
    abbrev_hint = ", ".join(f"{k}={v}" for k, v in list(doc.abbrev_map.items())[:10]) if doc.abbrev_map else "(无)"

    system_prompt = """你是一位资深的药物警戒文献信息抽取专家。

你的任务是从医学文献中**结构化抽取**两类关键信息（drug_mentions 与 ae_events），覆盖文中所有药物与AE。最终分类由规则层完成，你不需要输出分类标签。

================================================================================
## 抽取关键原则
================================================================================
1. 先抽“药物-AE配对”，再由规则层判断是否涉及目标药。
2. 目标药只是检索关键词，不等于AE致因药，禁止默认归因到目标药。
3. 每个AE事件必须给出related_drug（允许为unknown或X+Y）。
4. 文章最终标签仅由“可绑定到目标药”的合格事件贡献：请在每个AEEvent中填 is_target_drug_related 与 related_drug_evidence，方便规则层二次判断。

================================================================================
## 第一步：药物提及与诺华判断（仅用于drug_mentions）
================================================================================

### 位置判断（最优先）
- 产品**只出现在参考文献/作者信息**中，正文无提及 → 非诺华药暴露
- 产品在**正文任意位置**出现 → 继续判断是否为诺华产品

### 诺华产品判断规则（按顺序1→5，符合即停止）

**step1_商品名判断**（最优先）
- 提到诺华商品名 → 诺华药
- 诺华商品名包括：山地明、新山地明、善宁、善龙、格列卫、维全特、诺欣妥、可善挺、恩瑞格、派立噻等
- 提到非诺华商品名（如其他厂家品牌） → 非诺华药

**step2_厂家/批文判断**
- 文章注明厂家是诺华/北京诺华/Novartis → 诺华药
- 批准文号对应诺华产品 → 诺华药
- 明确注明其他厂家 → 非诺华药

**step3_国产进口状态判断**
- 文章提到"国产XX"，但诺华该产品只有进口 → 非诺华药
- 文章提到"进口XX"，诺华有进口产品 → 可能诺华药

**step4_剂型判断**
- 注意包含关系：文章"片剂"+诺华"缓释片" → 算诺华药
- 文章"缓释片"+诺华"片剂" → 不算诺华药
- 剂型与诺华产品完全不符 → 非诺华药

**step5_规格判断**
- 文章明确给出规格，且诺华不生产该规格 → 非诺华药
- 注意区分"剂量"和"规格"

**无辅助信息时**
- 仅使用通用名(如"环孢素")且无上述任何辅助判断信息 →
  - 若该名称出现在“药物关键词参考”中，可判定为诺华药(novartis_judgement=yes)
  - 否则标为unknown（不默认yes）

### 特殊注意事项
1. **缩写识别**：文章定义的缩写(如"环孢素(CsA)")，后文所有CsA都指环孢素
2. **换行连接**：PDF提取可能有断行，需自动连接（如"环孢\n素"→"环孢素"）
3. **多处提及分别判断**：作者试验用非诺华药，但引言/讨论引用他人数据用同成分药，需分别判断
4. **盐酸莫西沙星特殊规则**：如治疗非眼部感染 → 非诺华药；眼部感染或不确定 → 诺华药

================================================================================
## 第二步：不良事件(AE)判断
================================================================================

### 基本定义
任何用药后发生的所有异常，只需看**时间关系**（先用药后发生，看不出时间可保守判断为有）

### 非AE排除（非常重要）
以下内容**不属于AEEvent**，禁止抽成AE事件：
- 诊断检查/影像学/显像/扫描本身的“分辨率低、图像质量差、结果混淆、假阳性/假阴性、难以操作/难以解读、检查敏感性/特异性”等（属于检测性能或操作问题，不是患者不良事件）
- 仅讨论“检测结果是否准确/是否可重复/是否受干预影响”，但未描述患者出现症状/体征/实验室异常/疾病进展

### AE分类

**Normal AE（人体用药后异常）**
- 体征/症状/疾病：用药后高热/头晕/低血糖/皮疹等
- 检查值异常：文中注明异常，或超出正常范围
- 疾病复发、停药后疾病加重/进展
- **ADR统计/回顾性分析特例（非常重要）**：如果文章是“药品不良反应(ADR)回顾/统计”，且正文或表格/列表出现“药品名称 + ADR例数/占比(%)/排序”，则这属于药物相关AE信息：
  - 必须抽取为 `AEEvent`（`event_type="normal_ae"`，`causality="implicit"`）
  - `related_drug` 必须填写该行/该条对应的药品名称（不要写unknown）
  - `patient_count` 按例数：例数>=2 → multiple；例数==1 → single；未给例数 → unknown
  - `source` 填 `table`（若来自表格），并在 `raw_count_in_table` 填例数（如 5）
  - 例子（这些都属于“药品名称 + 例数/占比”）：`环孢素软胶囊 5 1.63`、`环孢素软胶囊 5（1.63%）`、`环孢素软胶囊 5(1.63)`
  - **当表格/列表包含目标药时，必须优先抽取目标药对应的那一条**

**特殊情况(Special Situation)**
1. **孕期/哺乳期暴露**：用药后妊娠/孕检阳性/生产/流产/早产/死胎等
2. **感染传播**：通过药物传播的感染病
3. **撤药/回弹反应**：停药后原始症状更严重
4. **LOE(疗效缺失)**：
	   - 用药后target适应症未改善/无效/效果不佳/患者耐药
	   - 适应症相关症状持续出现（如用诺欣妥治心衰，用药后仍胸闷）
	   - 免疫抑制剂(米芙/环孢素/巴利昔单抗)使用后出现排斥或GVHD
	   - 实验室检查值用药前后无变化
	   - **临床研究“疗效/总有效率”表格特例（非常重要）**：如果结果部分出现“显效/有效/无效/总有效率/总有效率=显效率+有效率”等，并且给出了**组别+例数/占比(%)**（例如：`表1 两组患者临床疗效比较[例(%)] ... 无效 4(8.00)`），则必须抽取 `LOE` 事件：
	     - `event_type="special_situation"`, `special_subtype="LOE"`，`description` 表达“部分患者治疗无效/疗效不佳/未改善”
	     - `patient_count` 按无效例数判断（>=2 multiple；==1 single），`raw_count_in_table` 填无效例数，`source="table"`, `context_type="study"`
	     - 如果治疗方案中明确包含目标药（尤其对照组/基础治疗方案里列出了目标药），则该 LOE 的 `related_drug` 填目标药名，并 `is_target_drug_related=true`（不要写unknown）
	   - 常见写法提示：疗效不佳/欠佳/未达标/控制不佳/未缓解/失败/总有效率/显效率/无效率/无效例数/有效例数
5. **Overdose**：超过说明书最大剂量
6. **有意错误用药/药物滥用/依赖/成瘾**
7. **Medication errors**：
   - 偶然暴露（误服）
   - 职业暴露（药液接触医护）
   - 处方/分发错误
   - 药物管理不善（漏服/给药途径错误/自行掰片等）
   - 注意：患者病好自行停药不算
8. **药物相互作用/药食相互作用**
9. **疾病进展/加重**：适应症进一步加重、肿瘤转移
10. **意外获益**：治疗A病时B病好转
11. **儿童用药**：患者年龄超出说明书允许范围（无儿童用药章节则<18岁）
12. **Off-label use**：未按说明书适应症/剂量用药

**其他AE(Other AE)**
- 手术成功率≠100%、不良反应发生率≠0等
- 体外实验/动物实验/菌种耐药测试中的安全问题
- 有潜在风险但证据不充分
- 诊断检查/影像学/显像/扫描的准确性问题（如假阳性/假阴性、分辨率、操作或解读困难）不算AE

================================================================================
## 事件级药物归因规则（必须遵守）
================================================================================
1. 每个AEEvent必须填写related_drug：
   - 文中明确写“X药导致/诱发/相关” → related_drug=X
   - 联合用药且无法明确归因 → causality="not_attributable_combo" 且 related_drug="X+Y"
   - 仅知道“用药后出现”但无法判断是哪一个药 → related_drug="unknown"
2. 禁止默认归因：目标药只是文献检索关键词，不等于AE的致因药。
3. 但在临床研究/队列中，如果“不良反应/AE表格或结果”明确是在目标药治疗期间统计，且没有其他可疑药物，则 related_drug 应填目标药名，并 is_target_drug_related=true（source=table 时尤其如此）。

================================================================================
## 第三步：因果关系判断
================================================================================

**有因果关系(explicit)**
- 文章明确提到：药物X"导致/引起/致使/诱发"AE
- 药物中毒/药敏/药疹/过敏反应
- "因XXX(不良事件)停用药物X"
- 明确的副作用side effect/不良反应ADR

**联合用药因果判断**
- 药物A + 药物B联用，发生AE → 若不能明确归因到单一药物，设 causality="not_attributable_combo" 且 related_drug="A+B"
- 研究药物为A+B，a组单用A，b组用A+B：
  - a组AE → 有因果关系（related_drug=A）
  - b组AE → 无法归因（not_attributable_combo，related_drug=A+B）

**隐含因果(implicit)**
- 临床研究中记录的治疗相关AE
- 病例报告中"用药后出现XX"（若未指明具体药物，related_drug=unknown）

================================================================================
## 第四步：患者数量判断
================================================================================

**单患者(single)**
- 通过性别/年龄可区分个例患者
- 群组试验表格/描述中出现"1例"

**多患者(multiple)**
- 正文/图表提到：3例、10%、多例等

================================================================================
## Off-label特殊产品规则
================================================================================

Off-label use需根据产品判断分类：
- **善宁、善龙、米芙、派立噻、恩瑞格、维莫思、泰立沙、派威妥** → 判ICSR/Multiple_Patients
- **其他产品** → 判Other_Safety_Signal

================================================================================
## 输出格式
================================================================================
返回JSON:
{
  "drug_mentions": [
    {
      "where": "body|ref|author",
      "context_type": "study|intro|discussion|citation",
      "mention_polarity": "used|not_used|history|background|unknown",
      "surface_name": "原文中的药物名称表述",
      "normalized": "标准化药物名称",
      "novartis_judgement": "yes|no|unknown",
      "novartis_rule_hit": "step1_商品名|step2_厂家|step3_国产进口|step4_剂型|step5_规格|无辅助信息",
      "non_novartis_reason": "如判断非诺华药，说明原因",
      "evidence_span": "包含该药物提及的原文片段(20-50字)"
    }
  ],
  "ae_events": [
    {
      "event_type": "normal_ae|special_situation|other_ae",
      "special_subtype": "孕期暴露|LOE|overdose|medication_error|off_label|儿童用药|药物相互作用|疾病进展|其他",
      "description": "事件简要描述",
      "related_drug": "关联的药物名称(必须填；不明确填unknown，联用不可归因填X+Y)",
      "related_drug_evidence": "原文中支撑related_drug的片段(10-40字)",
      "is_target_drug_related": true,
      "context_type": "study|case|citation|background|unknown",
      "context_evidence": "用于判断context_type的原文片段(10-60字)",
      "source": "table|text|unknown",
      "raw_count_in_table": 1,
      "human_flag": "human|nonhuman|unclear",
      "timing_relation": "after_drug|unclear|none",
      "causality": "explicit|implicit|none|not_attributable_combo",
      "causality_evidence": "因果关系的原文证据",
      "patient_count": "single|multiple|unknown",
      "patient_evidence": "患者数量的证据(如'1例'/'纳入60例'/'3例头疼')",
      "off_label": false,
      "off_label_product": "如off_label为true，填写产品名",
      "evidence_span": "包含该事件的原文片段(30-80字)"
    }
  ]
}"""

    user_prompt = f"""请从以下文献中进行结构化抽取。

## 检索关键词（来自文件名，仅用于索引，不用于AE归因）
{doc.target_drug or "(无法确定)"}

## 药物关键词参考
{drug_hint}

## 文章缩写映射
{abbrev_hint}

## 文章类型（规则检测）
{doc.article_type}

## 正文内容(BODY)
---
{truncate_text(doc.body_text, 35000)}
---

## 参考文献部分(REF) [前2000字]
---
{doc.ref_text[:2000] if doc.ref_text else "(无)"}
---

## 作者信息部分(AUTHOR)
---
{doc.author_text or "(无)"}
---

请返回结构化JSON，包含 drug_mentions[] 和 ae_events[] 两个数组。
每个药物提及和事件都必须带有 evidence_span（原文片段）。"""

    return system_prompt, user_prompt


def parse_extraction_response(response_json: dict) -> ExtractionResult:
    """解析LLM返回的JSON，转换为ExtractionResult对象。

    Args:
        response_json: LLM返回的JSON字典

    Returns:
        ExtractionResult: 结构化抽取结果
    """
    drug_mentions = []
    ae_events = []

    # 解析 drug_mentions
    raw_drugs = response_json.get("drug_mentions", []) or []
    for dm in raw_drugs:
        if not isinstance(dm, dict):
            continue
        drug_mentions.append(DrugMention(
            where=str(dm.get("where", "body")),
            context_type=str(dm.get("context_type", "study")),
            mention_polarity=str(dm.get("mention_polarity", "unknown")),
            surface_name=str(dm.get("surface_name", "")),
            normalized=str(dm.get("normalized", "")),
            novartis_judgement=str(dm.get("novartis_judgement", "unknown")),
            novartis_rule_hit=str(dm.get("novartis_rule_hit", "")),
            non_novartis_reason=str(dm.get("non_novartis_reason", "")),
            evidence_span=str(dm.get("evidence_span", ""))
        ))

    # 解析 ae_events
    raw_events = response_json.get("ae_events", []) or []
    for ae in raw_events:
        if not isinstance(ae, dict):
            continue
        related_drug = str(ae.get("related_drug", "")).strip()
        if not related_drug:
            related_drug = "unknown"
        raw_count = ae.get("raw_count_in_table", None)
        try:
            raw_count_in_table = int(raw_count) if raw_count is not None else None
        except Exception:
            raw_count_in_table = None
        ae_events.append(AEEvent(
            event_type=str(ae.get("event_type", "other_ae")),
            special_subtype=str(ae.get("special_subtype", "")),
            description=str(ae.get("description", "")),
            related_drug=related_drug,
            related_drug_evidence=str(ae.get("related_drug_evidence", "")),
            is_target_drug_related=bool(ae.get("is_target_drug_related", False)),
            context_type=str(ae.get("context_type", "unknown")),
            context_evidence=str(ae.get("context_evidence", "")),
            source=str(ae.get("source", "unknown")),
            raw_count_in_table=raw_count_in_table,
            human_flag=str(ae.get("human_flag", "unclear")),
            timing_relation=str(ae.get("timing_relation", "unclear")),
            causality=str(ae.get("causality", "none")),
            causality_evidence=str(ae.get("causality_evidence", "")),
            patient_count=str(ae.get("patient_count", "unknown")),
            patient_evidence=str(ae.get("patient_evidence", "")),
            off_label=bool(ae.get("off_label", False)),
            off_label_product=str(ae.get("off_label_product", "")),
            whitelist_ok=bool(ae.get("whitelist_ok", False)),
            evidence_span=str(ae.get("evidence_span", ""))
        ))

    return ExtractionResult(
        drug_mentions=drug_mentions,
        ae_events=ae_events,
        raw_response=response_json
    )


def extract_with_openai_v2(doc: DocStruct, drug_keywords: list[str]) -> ExtractionResult:
    """使用OpenAI进行V2版本结构化抽取。

    Args:
        doc: 结构化文档信息
        drug_keywords: 药物关键词列表

    Returns:
        ExtractionResult: 结构化抽取结果
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ExtractionResult(
            drug_mentions=[],
            ae_events=[],
            raw_response={"error": "OPENAI_API_KEY not set"}
        )

    # Guard against hanging network calls.
    # NOTE: In some environments, the default socket read may hang indefinitely; enforce explicit request timeouts.
    try:
        import httpx  # type: ignore
        timeout = httpx.Timeout(connect=15.0, read=90.0, write=30.0, pool=15.0)
    except Exception:
        timeout = 90.0

    client = OpenAI(api_key=api_key, timeout=timeout, max_retries=2)
    system_prompt, user_prompt = build_extraction_prompt_v2(doc, drug_keywords)

    try:
        model = os.getenv("CLASSIFY_MODEL_NAME", "gpt-4o")
        is_reasoning_model = model.startswith("o1") or model.startswith("o3")

        create_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        }

        if not is_reasoning_model:
            create_kwargs["temperature"] = 0
            create_kwargs["response_format"] = {"type": "json_object"}

        # Per-request timeout (overrides client default if supported by the SDK).
        create_kwargs.setdefault("timeout", timeout)
        response = client.chat.completions.create(**create_kwargs)
        content = response.choices[0].message.content or "{}"

        # 解析JSON
        response_json = json.loads(content)
        return parse_extraction_response(response_json)

    except json.JSONDecodeError as e:
        return ExtractionResult(
            drug_mentions=[],
            ae_events=[],
            raw_response={"error": f"JSON parse error: {e}"}
        )
    except Exception as e:
        return ExtractionResult(
            drug_mentions=[],
            ae_events=[],
            raw_response={"error": str(e)}
        )


# ============================================================
# V2 结构化抽取架构 - Phase 4: 审计层
# ============================================================

def normalize_drug_text(text: str) -> str:
    """Normalize drug names for keyword matching."""
    if not text:
        return ""
    return re.sub(r"[\s\-_\/()（）,，。·]+", "", text).lower()


def build_drug_keyword_set(drug_keywords: list[str]) -> set[str]:
    """Build a normalized keyword set for Novartis drug matching."""
    return {normalize_drug_text(k) for k in drug_keywords if k.strip()}


def is_novartis_keyword_match(name: str, keyword_set: set[str]) -> bool:
    """Check if a name matches any Novartis keyword."""
    name_norm = normalize_drug_text(name)
    if not name_norm:
        return False
    return any(kw and kw in name_norm for kw in keyword_set)


def is_novartis_mention(dm: DrugMention, keyword_set: set[str]) -> bool:
    """Determine if a drug mention indicates Novartis exposure."""
    if dm.novartis_judgement == "yes":
        return True
    if dm.novartis_judgement != "unknown":
        return False
    return (
        is_novartis_keyword_match(dm.surface_name, keyword_set) or
        is_novartis_keyword_match(dm.normalized, keyword_set)
    )


def build_target_alias_set(target_drug: str, drug_keywords: list[str]) -> set[str]:
    """Build a normalized alias set for matching target drug mentions."""
    aliases: set[str] = set()
    if target_drug:
        aliases.add(normalize_drug_text(target_drug))

    # Reuse the alias map used in V1 search for common drugs.
    drug_aliases = {
        "卡马西平": ["carbamazepine", "tegretol", "得理多"],
        "奥卡西平": ["oxcarbazepine", "trileptal", "曲莱"],
        "缬沙坦": ["valsartan", "代文"],
        "来曲唑": ["letrozole", "芙瑞"],
        "环孢素": ["cyclosporine", "ciclosporin", "新山地明", "sandimmun", "sandimmune"],
        "布林佐胺": ["brinzolamide", "派立明"],
        "司库奇尤单抗": ["secukinumab", "可善挺", "cosentyx"],
        "妥布霉素": ["tobramycin", "托百士"],
        "雷珠单抗": ["ranibizumab", "诺适得", "lucentis"],
        "沙库巴曲缬沙坦": ["sacubitril/valsartan", "诺欣妥", "entresto"],
        "甲磺酸伊马替尼": ["imatinib", "格列卫", "gleevec", "glivec"],
        "伊马替尼": ["imatinib", "格列卫", "gleevec", "glivec"],
        "奥曲肽": ["octreotide", "善宁", "善龙", "sandostatin"],
        "帕唑帕尼": ["pazopanib", "维全特", "votrient"],
    }

    target_norm = normalize_drug_text(target_drug)
    for main_name, alias_list in drug_aliases.items():
        main_norm = normalize_drug_text(main_name)
        alias_norms = {normalize_drug_text(a) for a in alias_list}
        if target_norm and (target_norm == main_norm or target_norm in alias_norms):
            aliases.add(main_norm)
            aliases.update(alias_norms)

    # Include keywords that match/contain the target (handles product/generic variants).
    if target_norm:
        for kw in drug_keywords:
            kw_norm = normalize_drug_text(kw)
            if not kw_norm:
                continue
            if target_norm in kw_norm or kw_norm in target_norm:
                aliases.add(kw_norm)

    aliases.discard("")
    return aliases


def is_target_match(name: str, target_alias_set: set[str]) -> bool:
    """Check if a name matches the target drug (aliases)."""
    if not name:
        return False
    name_norm = normalize_drug_text(name)
    if not name_norm:
        return False
    # Handle combo strings like "A+B"
    parts = re.split(r"[+＋/、,，;；]", name_norm)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if any(alias and (alias in part or part in alias) for alias in target_alias_set):
            return True
    return any(alias and (alias in name_norm or name_norm in alias) for alias in target_alias_set)


def is_negative_use_span(text: str) -> bool:
    """Heuristic for 'not used' medication mentions."""
    if not text:
        return False
    t = text.lower()
    cues = ["未服用", "未使用", "未应用", "未给予", "未用药", "未曾使用", "停用", "未接受", "排除", "无近期用药"]
    return any(cue.lower() in t for cue in cues)


def audit_extraction(
    extraction: ExtractionResult,
    doc: DocStruct,
    drug_keyword_set: set[str]
) -> AuditResult:
    """审计LLM抽取结果，检测常见错误并标记需要修正的内容。

    审计规则:
    1. reference_only_novartis: 诺华药只在 ref/author 出现 → 强制无暴露
    2. missing_novartis_path: body有药但无step1-5命中 → 标记需审核
    3. patient_count_no_evidence: patient_count无证据 → 标记需审核
    4. combo_attribution: 明确多药联用 → 可能需标记无法归因
    5. nonhuman_ae: AE的human_flag不是human → 可能不合格

    Args:
        extraction: LLM结构化抽取结果
        doc: 文档结构信息
        drug_keyword_set: 诺华药关键词集合

    Returns:
        AuditResult: 审计结果
    """
    triggered_rules = []
    corrections = {}
    retry_questions = []

    # 规则1: reference_only_novartis
    # 诺华药物仅出现在参考文献/作者信息中
    # Audit uses broad Novartis keyword matching (not target-scoped).
    body_novartis_mentions = [
        dm for dm in extraction.drug_mentions
        if dm.where == "body" and is_novartis_mention(dm, drug_keyword_set)
    ]
    ref_author_novartis_mentions = [
        dm for dm in extraction.drug_mentions
        if dm.where in ("ref", "author") and is_novartis_mention(dm, drug_keyword_set)
    ]

    if not body_novartis_mentions and ref_author_novartis_mentions:
        triggered_rules.append("reference_only_novartis")
        corrections["novartis_exposure"] = (
            True, False,
            "诺华药仅在参考文献/作者信息中出现，不在正文中"
        )

    # 规则2: missing_novartis_path
    # 正文中有药物提及，但没有step1-5命中
    body_mentions = [dm for dm in extraction.drug_mentions if dm.where == "body"]
    for dm in body_mentions:
        if dm.novartis_judgement == "unknown" and not dm.novartis_rule_hit:
            if (
                is_novartis_keyword_match(dm.surface_name, drug_keyword_set) or
                is_novartis_keyword_match(dm.normalized, drug_keyword_set)
            ):
                continue
            triggered_rules.append("missing_novartis_path")
            retry_questions.append(
                f"药物'{dm.surface_name}'在正文中出现，但未命中诺华药判断规则(step1-5)，请确认是否为诺华产品"
            )
            break  # 只记录一次

    # 规则3: patient_count_no_evidence
    # AE事件有patient_count但无evidence
    for ae in extraction.ae_events:
        if ae.patient_count in ("single", "multiple") and not ae.patient_evidence:
            triggered_rules.append("patient_count_no_evidence")
            retry_questions.append(
                f"事件'{ae.description[:30]}'的患者数({ae.patient_count})缺少证据"
            )
            break

    # 规则4: nonhuman_ae
    # 检查是否有非人体AE被标记为安全事件
    human_ae_count = sum(1 for ae in extraction.ae_events if ae.human_flag == "human")
    nonhuman_ae_count = sum(1 for ae in extraction.ae_events if ae.human_flag == "nonhuman")
    if nonhuman_ae_count > 0 and human_ae_count == 0:
        triggered_rules.append("nonhuman_only")
        # 这不是错误，但需要在分类时考虑

    # 规则5: combo_attribution
    # 检查是否存在明确的联合用药且标记了因果关系
    combo_keywords = ["联合", "合用", "联用", "合并用药", "combination"]
    for ae in extraction.ae_events:
        if ae.causality in ("explicit", "implicit"):
            evidence_lower = ae.evidence_span.lower()
            if any(kw in evidence_lower for kw in combo_keywords):
                # 检查是否有多个药物
                drug_count = sum(
                    1 for dm in extraction.drug_mentions
                    if dm.where == "body" and dm.surface_name.lower() in evidence_lower
                )
                if drug_count > 1:
                    triggered_rules.append("combo_attribution")
                    # 不修正causality，只标记
                    break

    return AuditResult(
        triggered_rules=triggered_rules,
        corrections=corrections,
        retry_needed=len(retry_questions) > 0,
        retry_questions=retry_questions
    )


def apply_audit_corrections(
    extraction: ExtractionResult,
    audit: AuditResult
) -> ExtractionResult:
    """应用审计层的修正到抽取结果。

    注意：当前版本的审计修正主要影响后续的分类决策，
    而不是直接修改ExtractionResult。

    Args:
        extraction: 原始抽取结果
        audit: 审计结果

    Returns:
        ExtractionResult: 可能修正后的抽取结果
    """
    # 当前版本不直接修改extraction，而是让classify_by_rules_v2根据audit结果做决策
    return extraction


# ============================================================
# V2 结构化抽取架构 - Phase 5: 规则分类升级
# ============================================================

# Off-label特殊产品白名单：这些产品的off-label use可以判ICSR/Multiple_Patients
OFF_LABEL_WHITELIST_PRODUCTS = {
    "善宁", "善龙", "米芙", "派立噻", "恩瑞格", "维莫思", "泰立沙", "派威妥",
    # 英文名/通用名
    "sandostatin", "somatuline", "myfortic", "exjade", "votrient", "tasigna", "promacta",
}


def is_off_label_whitelist_product(product_name: str) -> bool:
    """判断产品是否在off-label白名单中。

    白名单产品的off-label use可以进入ICSR/Multiple_Patients分类。
    非白名单产品的off-label use只能进入Other_Safety_Signal。

    Args:
        product_name: 产品名称

    Returns:
        bool: 是否在白名单中
    """
    if not product_name:
        return False
    product_lower = product_name.lower()
    for whitelist_name in OFF_LABEL_WHITELIST_PRODUCTS:
        if whitelist_name.lower() in product_lower or product_lower in whitelist_name.lower():
            return True
    return False


def _infer_context_type_fallback(ae: AEEvent) -> str:
    """Infer context type if LLM didn't provide it reliably."""
    ctx = (ae.context_type or "").strip().lower()
    text = f"{ae.context_evidence}\n{ae.evidence_span}\n{ae.causality_evidence}".lower()
    citation_cues = [
        "参考文献", "文献报道", "据报道", "既往研究", "研究表明", "已有报道", "见文献", "报道如下", "meta",
    ]
    if any(cue in text for cue in citation_cues) or re.search(r"\[\s*\d+\s*\]", text):
        return "citation"
    background_cues = ["可能", "常见", "风险", "提示", "应注意", "发生率", "%", "约为", "potential", "may"]
    # Treat generic risk statements as background unless there's explicit temporal/case framing.
    strong_time_cues = ["用药后", "治疗期间", "服用后", "停药", "再激发", "challenge", "dechallenge", "rechallenge"]
    case_cues = ["病例", "例", "患者，", "患者,", "n=", "入组", "纳入"]
    if any(cue in text for cue in background_cues) and not any(cue in text for cue in strong_time_cues) and not any(cue in text for cue in case_cues):
        return "background"
    # Override: statistics-style sentences are usually background/citation even if they mention '患者'
    if ("发生率" in text or "%" in text or "约为" in text) and not any(cue in text for cue in ["本研究", "试验组", "对照组", "纳入"]):
        return "background"
    if ctx in ("study", "case", "citation", "background"):
        return ctx
    return "unknown"


def _is_countable_context(ae: AEEvent) -> bool:
    ctx = _infer_context_type_fallback(ae)
    return ctx not in ("citation", "background")


def _extract_table_counts(text: str) -> set[int]:
    """Extract counts from table-like snippets such as '1（2.86）'."""
    if not text:
        return set()
    counts = set()
    for m in re.finditer(r"(\d{1,3})\s*[（(]\s*\d+(?:\.\d+)?\s*[)）]", text):
        try:
            counts.add(int(m.group(1)))
        except Exception:
            continue
    return counts


def is_target_related_event(ae: AEEvent, target_alias_set: set[str]) -> bool:
    """Determine whether an event can be bound to the target drug."""
    if not target_alias_set:
        return False
    if ae.related_drug and ae.related_drug != "unknown" and is_target_match(ae.related_drug, target_alias_set):
        return True

    # Evidence-based fallback (no free default):
    # - If the event explicitly cites the target drug in any evidence, accept.
    combined = f"{ae.related_drug_evidence}\n{ae.context_evidence}\n{ae.evidence_span}".strip()
    combined_norm = normalize_drug_text(combined)
    if any(alias and alias in combined_norm for alias in target_alias_set):
        return True

    # Trial AE tables often omit drug name in the AE row; rely on LLM + table source.
    if ae.is_target_drug_related and (ae.source or "").lower() == "table":
        if "不良反应" in (ae.evidence_span or "") or "adverse" in (ae.evidence_span or "").lower():
            return True

    # If LLM explicitly marked target-related and context is countable, accept as bound.
    if ae.is_target_drug_related and _is_countable_context(ae):
        return True

    return False


def is_qualifying_event(
    ae: AEEvent,
    audit: AuditResult,
    *,
    target_alias_set: set[str],
    require_target: bool = True,
) -> bool:
    """判断AE事件是否为"合格事件"（可纳入安全报告的事件）。

    合格事件定义:
    1. normal_ae: 必须是human + (explicit或implicit因果) + 非not_attributable_combo
    2. special_situation:
       - 如果是off_label，需要产品在白名单(善宁/善龙/米芙/派立噻/恩瑞格/维莫思/泰立沙/派威妥)
       - 否则直接合格（但审计层可能降级）

    Args:
        ae: AE事件
        audit: 审计结果（用于检查是否被降级）

    Returns:
        bool: 是否为合格事件
    """
    # 检查审计层是否对此事件有降级修正
    ae_key = f"ae_{ae.description[:20]}_qualification"
    if ae_key in audit.corrections:
        old_val, new_val, reason = audit.corrections[ae_key]
        if new_val is False:
            return False

    if not _is_countable_context(ae):
        return False

    if ae.causality == "not_attributable_combo":
        return False

    # Hard gate: only target-related events can contribute to ICSR/Multiple.
    # We verify using alias matching; LLM flag is advisory.
    if require_target:
        if not is_target_related_event(ae, target_alias_set):
            return False

    if ae.event_type == "normal_ae":
        # 常规AE：必须是人体+有因果+非联用不可归因
        if ae.human_flag != "human":
            return False
        if ae.causality not in ("explicit", "implicit"):
            return False
        return True

    elif ae.event_type == "special_situation":
        # 特殊情况
        if ae.special_subtype == "LOE":
            # LOE/无效必须明确绑定目标药
            if not is_target_related_event(ae, target_alias_set):
                return False
            # If evidence doesn't explicitly mention the drug, allow study/table LOE where target is the regimen.
            span_norm = normalize_drug_text(ae.evidence_span)
            if target_alias_set and not any(alias and alias in span_norm for alias in target_alias_set):
                ctx = _infer_context_type_fallback(ae)
                if not (ctx == "study" or (ae.source or "").lower() == "table"):
                    return False
        if ae.off_label:
            # 超说明书用药需要在白名单
            # 动态判断：如果LLM返回了off_label_product，用它判断
            # 否则使用LLM返回的whitelist_ok
            if ae.off_label_product:
                return is_off_label_whitelist_product(ae.off_label_product)
            return ae.whitelist_ok
        # 其他特殊情况（儿童用药、药物无效等）直接合格
        # 但需要是人体相关
        if ae.human_flag == "nonhuman":
            return False
        return True

    # other_ae 类型不计入合格事件，但可能作为 Other_Safety_Signal
    return False


def determine_patient_mode_v2(ae_events: list[AEEvent]) -> str:
    """根据AE事件列表确定患者模式。

    Args:
        ae_events: 合格的AE事件列表

    Returns:
        str: 患者模式 (single/multiple/mixed/unknown)
    """
    has_single = False
    has_multiple = False

    for ae in ae_events:
        counts = _extract_table_counts(ae.patient_evidence or "") | _extract_table_counts(ae.evidence_span or "")
        has_table_single = (ae.raw_count_in_table == 1) or (1 in counts)
        if ae.patient_count == "single" or has_table_single:
            has_single = True
        elif ae.patient_count == "multiple":
            has_multiple = True

    if has_single and has_multiple:
        return "mixed"
    elif has_single:
        return "single"
    elif has_multiple:
        return "multiple"
    else:
        return "unknown"


def classify_by_rules_v2(
    extraction: ExtractionResult,
    audit: AuditResult,
    *,
    drug_keyword_set: set[str],
    target_drug: str,
    target_alias_set: set[str],
) -> tuple[str, str, bool, bool]:
    """V2版本的规则分类决策树。

    决策树:
    1. novartis_exposure = False → Rejection
    2. novartis_exposure + 无ae_events → Rejection
    3. 有qualifying_event(single) + qualifying_event(multiple) → ICSR+Multiple_Patients
    4. 仅有qualifying_event(single) → ICSR
    5. 仅有qualifying_event(multiple) → Multiple_Patients
    6. 有诺华暴露+有事件但不合格 → Other_Safety_Signal

    Args:
        extraction: 结构化抽取结果
        audit: 审计结果
        drug_keyword_set: 诺华药关键词集合
        target_drug: 目标药物名称（文件名前缀）
        target_alias_set: 目标药别名集合（归一化）

    Returns:
        tuple: (label, decision_path, novartis_exposure, has_qualifying_event)
    """
    # Step 1: 判断诺华暴露
    # 检查审计层是否强制修正了暴露状态
    if "novartis_exposure" in audit.corrections:
        _, novartis_exposure, _ = audit.corrections["novartis_exposure"]
    else:
        # 暴露：正文中出现目标药物（used/history），并满足诺华药关键词或LLM判断
        def is_positive_polarity(dm: DrugMention) -> bool:
            pol = (dm.mention_polarity or "unknown").lower()
            if pol in ("used", "history"):
                return True
            if pol == "not_used":
                return False
            # Fallback: detect negation in evidence span
            return not is_negative_use_span(dm.evidence_span)

        novartis_exposure = any(
            dm.where == "body"
            and is_target_match(dm.surface_name or dm.normalized, target_alias_set)
            and is_positive_polarity(dm)
            and is_novartis_mention(dm, drug_keyword_set)
            for dm in extraction.drug_mentions
        )

    # 如果无暴露 → Rejection
    if not novartis_exposure:
        return "Rejection", "无诺华药暴露", False, False

    # Step 2: 检查是否有可计入的AE事件（排除citation/background）
    countable_events = [
        ae for ae in extraction.ae_events
        if _is_countable_context(ae) and ae.human_flag != "nonhuman"
    ]
    if not countable_events:
        return "Rejection", "有暴露但无AE事件", True, False

    # Step 3: 筛选合格事件
    qualifying_events = [
        ae for ae in countable_events
        if is_qualifying_event(ae, audit, target_alias_set=target_alias_set, require_target=True)
    ]

    if not qualifying_events:
        # 有暴露、有事件但无合格事件 → Other_Safety_Signal
        return "Other_Safety_Signal", "有暴露+有事件但无合格事件", True, False

    # Step 4: 确定患者模式
    patient_mode = determine_patient_mode_v2(qualifying_events)

    # Step 5: 根据患者模式决定标签
    if patient_mode == "mixed":
        return "ICSR+Multiple_Patients", "有暴露+合格事件(单+多)", True, True
    elif patient_mode == "single":
        return "ICSR", "有暴露+合格事件(单患者)", True, True
    elif patient_mode == "multiple":
        return "Multiple_Patients", "有暴露+合格事件(多患者)", True, True
    else:
        # unknown patient_mode → Other_Safety_Signal
        return "Other_Safety_Signal", "有暴露+合格事件但患者数未知", True, True


def classify_paper_v2(
    pdf_path: Path,
    drug_keywords: list[str],
) -> ClassificationResultV2:
    """V2版本的完整分类流程。

    流程:
    1. 解析文档 (parse_document_v2)
    2. 结构化抽取 (extract_with_openai_v2)
    3. 审计层 (audit_extraction)
    4. 规则分类 (classify_by_rules_v2)

    Args:
        pdf_path: PDF文件路径
        drug_keywords: 药物关键词列表

    Returns:
        ClassificationResultV2: V2版本分类结果
    """
    filename = pdf_path.name

    # Step 1: 解析文档
    doc = parse_document_v2(pdf_path)

    if not doc.raw_text.strip():
        # 文本提取失败
        return ClassificationResultV2(
            filename=filename,
            label="Error",
            label_cn="错误",
            novartis_exposure=False,
            has_qualifying_event=False,
            extraction=ExtractionResult([], [], {}),
            audit=AuditResult([], {}, False, []),
            confidence=0.0,
            reasoning="文本提取失败",
            decision_path="",
            needs_review=True,
            extract_method="none",
            text_length=0,
            error="Text extraction failed"
        )

    # Step 2: 结构化抽取
    extraction = extract_with_openai_v2(doc, drug_keywords)
    drug_keyword_set = build_drug_keyword_set(drug_keywords)
    target_alias_set = build_target_alias_set(doc.target_drug, drug_keywords)

    # 检查抽取是否有错误
    if "error" in extraction.raw_response:
        return ClassificationResultV2(
            filename=filename,
            label="Error",
            label_cn="错误",
            novartis_exposure=False,
            has_qualifying_event=False,
            extraction=extraction,
            audit=AuditResult([], {}, False, []),
            confidence=0.0,
            reasoning=f"抽取错误: {extraction.raw_response.get('error')}",
            decision_path="",
            needs_review=True,
            extract_method="pdftotext",
            text_length=len(doc.raw_text),
            error=str(extraction.raw_response.get("error", ""))
        )

    # Step 3: 审计层
    audit = audit_extraction(extraction, doc, drug_keyword_set)

    # Step 4: 规则分类
    label, decision_path, novartis_exposure, has_qualifying = classify_by_rules_v2(
        extraction,
        audit,
        drug_keyword_set=drug_keyword_set,
        target_drug=doc.target_drug,
        target_alias_set=target_alias_set,
    )

    # 计算置信度
    confidence = calculate_confidence_v2(extraction, audit)

    # 生成reasoning
    reasoning = generate_reasoning_v2(extraction, audit, label, decision_path)

    return ClassificationResultV2(
        filename=filename,
        label=label,
        label_cn=SAFETY_LABELS.get(label, "未知"),
        novartis_exposure=novartis_exposure,
        has_qualifying_event=has_qualifying,
        extraction=extraction,
        audit=audit,
        confidence=confidence,
        reasoning=reasoning,
        decision_path=decision_path,
        needs_review=confidence < 0.7 or audit.retry_needed,
        extract_method="pdftotext",
        text_length=len(doc.raw_text),
    )


def calculate_confidence_v2(extraction: ExtractionResult, audit: AuditResult) -> float:
    """计算V2版本分类结果的置信度。

    Args:
        extraction: 结构化抽取结果
        audit: 审计结果

    Returns:
        float: 置信度 (0.0-1.0)
    """
    confidence = 0.85  # 基础置信度

    # 审计层触发规则会降低置信度
    if audit.triggered_rules:
        confidence -= 0.05 * len(audit.triggered_rules)

    # 需要补问会大幅降低置信度
    if audit.retry_needed:
        confidence -= 0.15

    # 药物提及有诺华规则命中会提高置信度
    has_rule_hit = any(
        dm.novartis_rule_hit
        for dm in extraction.drug_mentions
        if dm.where == "body"
    )
    if has_rule_hit:
        confidence += 0.05

    # AE事件有明确因果会提高置信度
    has_explicit_causality = any(
        ae.causality == "explicit"
        for ae in extraction.ae_events
    )
    if has_explicit_causality:
        confidence += 0.05

    return max(0.3, min(0.95, confidence))


def generate_reasoning_v2(
    extraction: ExtractionResult,
    audit: AuditResult,
    label: str,
    decision_path: str
) -> str:
    """生成V2版本分类结果的推理说明。

    Args:
        extraction: 结构化抽取结果
        audit: 审计结果
        label: 分类标签
        decision_path: 决策路径

    Returns:
        str: 推理说明
    """
    parts = []

    # 决策路径
    parts.append(f"决策: {label} ({decision_path})")

    # 药物提及统计
    body_drugs = [dm for dm in extraction.drug_mentions if dm.where == "body"]
    ref_drugs = [dm for dm in extraction.drug_mentions if dm.where in ("ref", "author")]
    parts.append(f"药物提及: 正文{len(body_drugs)}个, 参考文献{len(ref_drugs)}个")

    # AE事件统计
    if extraction.ae_events:
        human_ae = [ae for ae in extraction.ae_events if ae.human_flag == "human"]
        nonhuman_ae = [ae for ae in extraction.ae_events if ae.human_flag == "nonhuman"]
        parts.append(f"AE事件: 人体{len(human_ae)}个, 非人体{len(nonhuman_ae)}个")

    # 审计层触发
    if audit.triggered_rules:
        parts.append(f"审计规则触发: {', '.join(audit.triggered_rules)}")

    # 审计修正
    if audit.corrections:
        corrections_str = "; ".join(
            f"{k}: {reason}"
            for k, (old, new, reason) in audit.corrections.items()
        )
        parts.append(f"修正: {corrections_str}")

    return " | ".join(parts)


def which(cmd: str) -> str | None:
    """Find executable in PATH."""
    import shutil
    return shutil.which(cmd)


def extract_pdf_text(pdf_path: Path, max_pages: int = 30) -> tuple[str, str]:
    """Extract text from PDF using pdftotext or pymupdf."""
    # Try pdftotext first
    pdftotext = which("pdftotext")
    if pdftotext:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = Path(tmpdir) / "out.txt"
                proc = subprocess.run(
                    [pdftotext, "-layout", "-enc", "UTF-8", "-l", str(max_pages), str(pdf_path), str(out_path)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if proc.returncode == 0 and out_path.exists():
                    text = out_path.read_text(encoding="utf-8", errors="ignore")
                    if len(text.strip()) >= 50:
                        return text, "pdftotext"
        except Exception:
            pass

    # Fallback to pymupdf
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        texts = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            texts.append(page.get_text())
        doc.close()
        text = "\n\n".join(texts)
        if text.strip():
            return text, "pymupdf"
    except ImportError:
        pass
    except Exception:
        pass

    return "", "none"


def truncate_text(text: str, max_chars: int = 45000) -> str:
    """Truncate text to max characters, keeping head and tail."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return text[:head] + "\n\n[...truncated...]\n\n" + text[-tail:]


def extract_target_drug_from_filename(filename: str) -> str | None:
    """从文件名前缀提取目标药物名称。

    根据专家反馈，PDF文件名格式通常为: "药物名-文章标题.pdf"
    文件名前缀（第一个"-"之前的部分）即为该文献的目标监测药物。

    Args:
        filename: PDF文件名

    Returns:
        目标药物名称，如果无法提取则返回None
    """
    if not filename:
        return None

    # 去除扩展名
    name = filename.rsplit('.', 1)[0] if '.' in filename else filename

    # 按第一个 "-" 分割，取前缀作为目标药物
    if '-' in name:
        target_drug = name.split('-', 1)[0].strip()
        if target_drug:
            return target_drug

    return None


import re


def search_drug_in_text(text: str, target_drug: str, drug_keywords: list[str]) -> dict:
    """在全文中搜索目标药物，返回搜索结果和上下文。

    Args:
        text: 文章全文
        target_drug: 目标药物名称（从文件名提取）
        drug_keywords: 药物关键词列表（包含别名）

    Returns:
        dict: {
            'found': bool,  # 是否找到
            'count': int,   # 出现次数
            'matched_terms': list[str],  # 匹配到的具体词
            'contexts': list[str],  # 上下文片段（最多5个）
            'search_terms': list[str],  # 搜索的关键词
        }
    """
    if not text or not target_drug:
        return {
            'found': False, 'count': 0, 'matched_terms': [],
            'contexts': [], 'search_terms': []
        }

    # 构建搜索词列表：目标药物 + 相关别名
    search_terms = [target_drug.lower()]

    # 从药物关键词列表中找相关别名
    target_lower = target_drug.lower()
    for kw in drug_keywords:
        kw_lower = kw.lower()
        # 如果关键词包含目标药物或目标药物包含关键词
        if target_lower in kw_lower or kw_lower in target_lower:
            if kw_lower not in search_terms:
                search_terms.append(kw_lower)
        # 常见药物别名映射
        drug_aliases = {
            '卡马西平': ['carbamazepine', 'tegretol', '得理多'],
            '奥卡西平': ['oxcarbazepine', 'trileptal', '曲莱'],
            '缬沙坦': ['valsartan', '代文'],
            '来曲唑': ['letrozole', '芙瑞'],
            '环孢素': ['cyclosporine', 'ciclosporin', '新山地明', 'sandimmun'],
            '布林佐胺': ['brinzolamide', '派立明'],
            '司库奇尤单抗': ['secukinumab', '可善挺', 'cosentyx'],
            '妥布霉素': ['tobramycin', '托百士'],
            '雷珠单抗': ['ranibizumab', '诺适得', 'lucentis'],
            '沙库巴曲缬沙坦': ['sacubitril/valsartan', '诺欣妥', 'entresto'],
            '甲磺酸伊马替尼': ['imatinib', '格列卫', 'gleevec', 'glivec'],
            '伊马替尼': ['imatinib', '格列卫', 'gleevec', 'glivec'],
            'octreotide': ['奥曲肽', '善龙', 'sandostatin'],
            'pazopanib': ['帕唑帕尼', '维全特', 'votrient'],
        }
        for main_name, aliases in drug_aliases.items():
            if target_lower == main_name.lower() or target_lower in [a.lower() for a in aliases]:
                for alias in aliases:
                    if alias.lower() not in search_terms:
                        search_terms.append(alias.lower())
                if main_name.lower() not in search_terms:
                    search_terms.append(main_name.lower())

    # 在文本中搜索
    text_lower = text.lower()
    # 创建去除空格的版本（处理OCR空格问题，如"卡 马 西 平"）
    text_no_space = re.sub(r'\s+', '', text_lower)

    matched_terms = []
    all_positions = []

    for term in search_terms:
        term_lower = term.lower()
        term_no_space = re.sub(r'\s+', '', term_lower)

        # 方法1: 直接匹配（原文本）
        pattern = re.escape(term_lower)
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            matched_terms.append(term)
            for m in matches:
                all_positions.append((m.start(), m.end(), term))

        # 方法2: 去空格后匹配（处理OCR问题）
        if not matches and len(term_no_space) >= 2:
            # 在去空格的文本中搜索
            pattern_no_space = re.escape(term_no_space)
            matches_no_space = list(re.finditer(pattern_no_space, text_no_space))
            if matches_no_space:
                matched_terms.append(f"{term}(OCR修正)")
                # 估算原文位置（不精确但足够）
                for m in matches_no_space:
                    # 使用去空格位置的1.5倍作为估算
                    est_pos = int(m.start() * 1.5)
                    all_positions.append((est_pos, est_pos + len(term), term))

        # 方法3: 允许字符间有空格的模式（如"卡 马 西 平"）
        if not matches and len(term_lower) >= 2:
            # 构建允许空格的正则：卡\s*马\s*西\s*平
            spaced_pattern = r'\s*'.join(re.escape(c) for c in term_lower)
            matches_spaced = list(re.finditer(spaced_pattern, text_lower))
            if matches_spaced:
                if term not in matched_terms and f"{term}(OCR修正)" not in matched_terms:
                    matched_terms.append(f"{term}(空格)")
                for m in matches_spaced:
                    all_positions.append((m.start(), m.end(), term))

    # 去重并排序位置
    all_positions = sorted(set(all_positions), key=lambda x: x[0])

    # 提取上下文（前后各50个字符）
    contexts = []
    used_ranges = []
    for start, end, term in all_positions[:10]:  # 最多处理10个匹配
        # 避免重叠的上下文
        overlap = False
        for used_start, used_end in used_ranges:
            if not (end + 50 < used_start or start - 50 > used_end):
                overlap = True
                break
        if overlap:
            continue

        ctx_start = max(0, start - 50)
        ctx_end = min(len(text), end + 50)
        context = text[ctx_start:ctx_end].replace('\n', ' ').strip()
        # 标记匹配词
        context = f"...{context}..."
        contexts.append(context)
        used_ranges.append((ctx_start, ctx_end))

        if len(contexts) >= 5:
            break

    return {
        'found': len(matched_terms) > 0,
        'count': len(all_positions),
        'matched_terms': list(set(matched_terms)),
        'contexts': contexts,
        'search_terms': search_terms[:10],  # 只返回前10个搜索词
    }


# 文章类型常量
ARTICLE_TYPES = {
    'animal_study': '动物实验',
    'case_report': '病例报告',
    'review': '综述/指南',
    'clinical_study': '临床研究',
    'unknown': '未知类型',
}


def detect_article_type(text: str, filename: str) -> dict:
    """基于关键词检测文章类型。

    Args:
        text: 文章全文
        filename: 文件名（用于提取标题）

    Returns:
        dict: {
            'type': str,  # 文章类型代码
            'type_cn': str,  # 文章类型中文
            'confidence': float,  # 置信度
            'evidence': list[str],  # 匹配到的关键词证据
        }
    """
    if not text:
        return {'type': 'unknown', 'type_cn': '未知类型', 'confidence': 0.0, 'evidence': []}

    text_lower = text.lower()
    # 提取标题（文件名中"-"后面的部分，或前2000字符）
    title = filename.split('-', 1)[1] if '-' in filename else filename
    title = title.rsplit('.', 1)[0] if '.' in title else title
    title_lower = title.lower()

    # 文章开头部分（更重要）
    text_head = text_lower[:3000]

    evidence = []
    scores = {
        'animal_study': 0,
        'case_report': 0,
        'review': 0,
        'clinical_study': 0,
    }

    # ========== 动物实验检测 ==========
    animal_keywords = {
        '小鼠': 3, '大鼠': 3, 'mice': 3, 'mouse': 3, 'rat': 3, 'rats': 3,
        '实验动物': 3, '动物实验': 3, '动物模型': 3, 'animal model': 3,
        '造模': 2, '模型组': 2, '实验组大鼠': 3, '实验组小鼠': 3,
        '灌胃': 2, '腹腔注射': 2, '尾静脉': 2,
        '兔': 1, '豚鼠': 2, '犬': 1,
    }
    for kw, score in animal_keywords.items():
        # 检查去空格版本（处理OCR问题）
        kw_no_space = kw.replace(' ', '')
        text_no_space = re.sub(r'\s+', '', text_lower)
        if kw in text_lower or kw_no_space in text_no_space:
            scores['animal_study'] += score
            evidence.append(f"动物实验:{kw}")

    # 如果有"患者"出现在前2000字符，降低动物实验得分
    if '患者' in text_head or 'patient' in text_head:
        scores['animal_study'] = max(0, scores['animal_study'] - 3)

    # ========== 病例报告检测 ==========
    case_keywords = {
        '1例': 4, '一例': 4, '1 例': 4,
        '个案': 3, '病例报告': 4, 'case report': 4,
        '案例分享': 4, '病案分享': 4, '病案': 2,
        '个例': 3, '单例': 3,
    }
    for kw, score in case_keywords.items():
        if kw in title_lower:
            scores['case_report'] += score + 2  # 标题中出现权重更高
            evidence.append(f"病例报告(标题):{kw}")
        elif kw in text_head:
            scores['case_report'] += score
            evidence.append(f"病例报告:{kw}")

    # ========== 综述/指南检测 ==========
    review_keywords = {
        '综述': 4, '进展': 3, '研究进展': 4,
        '指南': 4, 'guideline': 4, 'review': 3,
        '专家共识': 4, '诊疗规范': 3, '诊治进展': 3,
        '文献复习': 3, '系统评价': 3, 'meta分析': 3, 'meta-analysis': 3,
    }
    for kw, score in review_keywords.items():
        if kw in title_lower:
            scores['review'] += score + 2
            evidence.append(f"综述(标题):{kw}")
        elif kw in text_head:
            scores['review'] += score
            evidence.append(f"综述:{kw}")

    # ========== 临床研究检测 ==========
    clinical_keywords = {
        '临床研究': 4, '临床试验': 4, 'clinical trial': 4, 'clinical study': 4,
        '随机': 3, '对照组': 3, '观察组': 3, '治疗组': 3,
        '纳入标准': 3, '排除标准': 3, '入组': 2,
        '例患者': 3, '名患者': 3,
        '回顾性分析': 3, '前瞻性': 3,
        'n=': 2, 'p<': 2, 'p=': 2, 'p值': 2,
    }
    for kw, score in clinical_keywords.items():
        if kw in text_lower:
            scores['clinical_study'] += score
            evidence.append(f"临床研究:{kw}")

    # ========== 确定最终类型 ==========
    max_score = max(scores.values())
    if max_score < 3:
        return {
            'type': 'unknown',
            'type_cn': '未知类型',
            'confidence': 0.5,
            'evidence': evidence[:5],
        }

    # 找出得分最高的类型
    best_type = max(scores, key=scores.get)

    # 计算置信度
    total_score = sum(scores.values()) or 1
    confidence = min(0.95, 0.5 + (scores[best_type] / total_score) * 0.5)

    return {
        'type': best_type,
        'type_cn': ARTICLE_TYPES[best_type],
        'confidence': round(confidence, 2),
        'evidence': evidence[:5],
    }


def classify_by_rules(
    has_drug: bool,
    has_ae: bool,
    has_causality: bool,
    has_special_situation: bool,
    patient_mode: str,
) -> str:
    """Rule-based classification logic.

    分类判断逻辑（根据专家反馈修订 v2）：
    1. Rejection：缺少drug，或者既无AE也无特殊情况（完全无安全监测价值）
    2. ICSR：drug + (AE+因果关系 OR 特殊情况) + 单个患者
    3. Multiple_Patients：drug + (AE+因果关系 OR 特殊情况) + 多个患者
    4. ICSR+Multiple_Patients：一篇文章同时满足ICSR和Multiple_Patients的条件
    5. Other_Safety_Signal：有drug且有AE/特殊情况，但缺少因果关系或患者信息（有风险，需关注）

    关键修订：
    - 特殊情况（儿童用药、药物无效等）可以独立构成安全信号，不需要AE
    - 只有完全无安全价值才Rejection，有drug+AE/特殊情况至少是Signal
    """
    # Rejection: 缺少药物
    if not has_drug:
        return "Rejection"

    # 判断是否有安全信号价值：有AE或有特殊情况
    has_safety_signal = has_ae or has_special_situation

    # Rejection: 既无AE也无特殊情况（完全无安全监测价值）
    if not has_safety_signal:
        return "Rejection"

    # 满足ICSR/Multiple_Patients的条件：
    # - (AE + 因果关系) OR 特殊情况
    # 特殊情况（儿童用药、药物无效、妊娠暴露等）可以独立构成安全信号
    meets_criteria = (has_ae and has_causality) or has_special_situation

    if patient_mode == "single":
        # 单个患者：满足条件则ICSR，否则Other_Safety_Signal
        return "ICSR" if meets_criteria else "Other_Safety_Signal"

    if patient_mode == "multiple":
        # 多个患者(>1例)：满足条件则Multiple_Patients，否则Other_Safety_Signal
        return "Multiple_Patients" if meets_criteria else "Other_Safety_Signal"

    if patient_mode == "mixed":
        # 混合(同时有单患者和多患者描述)：满足条件则ICSR+Multiple_Patients
        return "ICSR+Multiple_Patients" if meets_criteria else "Other_Safety_Signal"

    # 其他情况（unknown等）：有drug+AE/特殊情况但缺少患者信息，仍有风险价值
    return "Other_Safety_Signal"


def classify_with_openai(text: str, filename: str, drug_keywords: list[str]) -> ClassificationResult:
    """Classify paper using OpenAI GPT for drug safety."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ClassificationResult(
            filename=filename, label="Error", label_cn="错误",
            has_drug=False, has_ae=False, has_causality=False, has_special_situation=False,
            patient_mode="unknown", patient_max_n=None, confidence=0.0,
            drug_evidence=[], ae_evidence=[], causality_evidence=[],
            special_evidence=[], patient_evidence=[],
            has_drug_reasoning="", has_ae_reasoning="", has_causality_reasoning="",
            has_special_reasoning="", patient_reasoning="", reasoning="",
            needs_review=True, extract_method="", text_length=0,
            error="OPENAI_API_KEY not set"
        )

    client = OpenAI(api_key=api_key)
    drug_hint = ", ".join(drug_keywords[:100]) if drug_keywords else "(未提供药物关键词)"

    # 从文件名提取目标药物
    target_drug = extract_target_drug_from_filename(filename)
    target_drug_hint = f"【{target_drug}】" if target_drug else "(无法从文件名提取)"

    # 文章类型检测
    article_type_result = detect_article_type(text, filename)

    # 全文搜索目标药物
    drug_search_result = search_drug_in_text(text, target_drug, drug_keywords) if target_drug else None

    system_prompt = """你是一位资深的药物警戒信息提取专家。
你的任务是从医学/科学文献中提取关键安全信息，用于诺华药物安全监测。

文献检索业务背景：
在全文范围内以中英文商品名&活性成分名作为关键词进行检索，检索出上抛到CNKI & Wanfang数据库中的文献。
针对所有检索出来的文献进行审阅，识别文章中是否提及任何诺华药相关安全病例或潜在信号。

⚠️ 重要：目标药物判断规则（根据专家反馈修订）
- PDF文件名格式为: "目标药物名-文章标题.pdf"
- **文件名前缀（第一个"-"之前的部分）即为该文献的目标监测药物**
- 即使文章内容主要讨论的是其他药物，只要文中提及了文件名前缀所示的目标药物，就应该判定has_drug=True
- 例如: "卡马西平-左乙拉西坦致剥脱性皮炎.pdf" → 目标药物是"卡马西平"，不是"左乙拉西坦"

分类判断逻辑（根据专家反馈修订 v2）：
1. Rejection：缺少drug，或者既无AE也无特殊情况（完全无安全监测价值）
2. ICSR：drug + (AE+因果关系 OR 特殊情况) + 单个患者
3. Multiple_Patients：drug + (AE+因果关系 OR 特殊情况) + 多个患者
4. ICSR+Multiple_Patients：一篇文章同时满足ICSR和Multiple_Patients的条件
5. Other_Safety_Signal：有drug且有AE/特殊情况，但缺少因果关系或患者信息（有风险，需关注）

关键修订：
- 特殊情况（儿童用药、药物无效等）可以独立构成安全信号，不需要AE
- 只有完全无安全价值才Rejection，有drug+AE/特殊情况至少是Signal

需要提取的字段：

1. **has_drug** (boolean): 文章是否提及目标药物？
   - ⚠️ 目标药物 = 文件名前缀（第一个"-"之前的部分）
   - 在文章中搜索该目标药物的任何提及（中英文名、商品名、通用名均可）
   - 即使只是简单提及或作为背景信息，也算has_drug=True

2. **has_ae** (boolean): 是否描述了与药物使用相关的不良事件(AE)？
   - ✅ YES的情况（必须是人体临床中实际发生的）：
     - 病例报告中具体患者用药后出现的不良反应
     - 临床研究中明确记录的不良反应数据和发生率
     - 有具体患者、具体症状、具体时间的AE描述
   - ❌ NO的情况（必须严格排除）：
     - 综述/指南中假设性讨论（如"该药可能导致XX"、"常见副作用包括"）
     - 仅列举药物名称和疾病名称，但无具体病例证据
     - 动物实验中的毒性反应（不算人体AE）
     - 疾病本身症状（如肿瘤患者的腹泻是疾病症状，非药物AE）
     - 文献背景介绍中提及的一般性风险讨论
   - ⚠️ 关键判断：必须是"人体临床中实际发生的、有具体证据的药物相关AE"

3. **has_causality** (boolean): 是否有因果关系表述将药物与不良事件联系起来？
   - ✅ YES的情况：
     - 明确归因："与...相关"、"由...引起"、"归因于"、"药物诱发"、"导致"
     - 时间关联+明确因果："用药后出现XX症状"、"治疗期间发生"、"停药后缓解"
         - 去激发/再激发阳性
     - 病例报告中明确描述药物引起的症状
     - ⚠️ 临床研究隐含因果（新增）：
       * "治疗期间记录不良反应"、"观察指标包括不良反应"
       * "两组不良反应比较"、"试验组vs对照组AE发生率"
       * 对照研究设计本身隐含了对治疗相关AE的因果判断
   - ❌ NO的情况：
     - 综述/指南仅泛泛讨论药物可能的副作用（无具体病例）
     - 明确否定因果关系
     - 仅描述疾病自然病程
   - ⚠️ 临床研究中如果将"不良反应"作为观察指标，即视为存在隐含因果

4. **has_special_situation** (boolean): 是否存在以下特殊情况？⚠️ 特殊情况可独立构成安全信号
   - 妊娠/哺乳期暴露 (Pregnancy/lactation exposure)
   - 儿童用药 (Pediatric use - 患者为儿童/婴幼儿)
   - 药物无效/疗效不佳（需明确表述）："无效"、"治疗失败"、"未能控制"
   - 过量 (Overdose)
   - 用药错误 (Medication error)
   - 药物相互作用 (Drug-drug interaction)
   - 超说明书用药 (Off-label use)
   - ❌ 注意：常规临床研究中的"联合用药"、"加量"不算特殊情况

5. **patient_mode** (string): 患者识别
   - "single": 单个可识别患者
     * 标题含"1例"、"个案"、"病例报告"
     * ⚠️ "案例分享"/"病例分享"类文献：即使包含多个病例（病案1、病案2），每个病例仍是独立的单患者报告，应判为single
   - "multiple": 多个患者（队列研究、临床试验、回顾性分析）
     * 必须有明确样本量（如"纳入100例"、"n=50"）
     * 必须是多例患者的合并研究/统计分析
   - "mixed": 文章中既有单患者病例，又有多患者统计数据
   - "unknown": 综述/指南，无明确患者信息
   - ⚠️ 优先级规则：先识别文献类型，再判断患者模式。"案例分享"优先判为single

仅返回包含这些字段和证据数组的JSON对象。"""

    # 构建药物搜索结果提示
    if drug_search_result and drug_search_result['found']:
        drug_search_info = f"""
📍 【全文检索结果】目标药物在文中出现情况：
   - 检索状态: ✅ 找到
   - 出现次数: {drug_search_result['count']}次
   - 匹配词: {', '.join(drug_search_result['matched_terms'])}
   - 上下文片段:
"""
        for i, ctx in enumerate(drug_search_result['contexts'][:3], 1):
            drug_search_info += f"     [{i}] {ctx}\n"
    elif drug_search_result:
        drug_search_info = f"""
📍 【全文检索结果】目标药物在文中出现情况：
   - 检索状态: ❌ 未找到
   - 搜索词: {', '.join(drug_search_result['search_terms'][:5])}
   - ⚠️ 注意：全文检索未找到目标药物，请仔细核实文章内容
"""
    else:
        drug_search_info = ""

    # 构建文章类型提示
    article_type_info = f"""
📋 【文章类型检测】
   - 检测结果: {article_type_result['type_cn']}
   - 置信度: {article_type_result['confidence']}
   - 证据: {', '.join(article_type_result['evidence'][:3]) if article_type_result['evidence'] else '无'}
"""

    # 根据文章类型生成特定的判断指导（柔和建议，不强制）
    if article_type_result['type'] == 'animal_study':
        type_specific_guidance = """
💡 【仅供参考】规则检测提示本文可能是"动物实验"类型，但请以实际内容为准：
   - 纯动物实验（药物仅用于造模）通常不含人体安全信息
   - 但如果文章同时讨论了人体安全性数据、已知AE等，仍可能有价值
   - 请根据文章实际内容自主判断"""
    elif article_type_result['type'] == 'review':
        type_specific_guidance = """
💡 【仅供参考】规则检测提示本文可能是"综述/指南"类型，但请以实际内容为准：
   - 综述中如果仅泛泛讨论可能的副作用，一般不算具体AE
   - 但如果综述中引用了具体病例或AE数据统计，可按实际情况判断
   - 请根据文章实际内容自主判断"""
    elif article_type_result['type'] == 'case_report':
        type_specific_guidance = """
💡 【仅供参考】规则检测提示本文可能是"病例报告/案例分享"类型，但请以实际内容为准：
   - 病例报告中"用药后出现XX"一般可视为存在因果关系
   - patient_mode: 案例分享类文章可考虑判定为 single
   - 请根据文章实际内容自主判断"""
    elif article_type_result['type'] == 'clinical_study':
        type_specific_guidance = """
💡 【仅供参考】规则检测提示本文可能是"临床研究"类型，但请以实际内容为准：
   - 临床研究中记录的不良反应一般可视为存在因果关系
   - 请根据文章实际内容自主判断"""
    else:
        type_specific_guidance = ""

    user_prompt = f"""⚠️ 本文献的目标监测药物（从文件名前缀提取）: {target_drug_hint}
文件名: {filename}
{article_type_info}{type_specific_guidance}
{drug_search_info}
其他药物关键词参考: {drug_hint}

提取步骤:
1. 阅读文章，理解实际内容（文章类型检测仅供参考，以实际内容为准）
2. 根据【全文检索结果】和文章内容，判断是否提到目标药物
   - 如果检索找到且有明确上下文，通常 has_drug=True
   - 如果药物仅作为工具/背景提及，无安全监测价值，可考虑 has_drug=False
3. 判断 has_ae：
   - 关键问题：文章中是否描述了与药物相关的具体不良事件？
   - 具体的患者AE描述、AE发生率统计 → has_ae=True
   - 仅讨论疾病本身症状、理论风险 → has_ae=False
4. 判断 has_causality：
   - 明确因果表述（"导致"、"引起"、"相关"）→ has_causality=True
   - 临床研究/病例报告中的AE一般可视为存在隐含因果关系
5. 检查特殊情况（儿童用药、药物无效/疗效不佳、怀孕暴露等）
6. 判断患者数量（根据文章实际内容）:
   - single: 单个可识别患者的病例报告
   - multiple: 多患者研究、队列研究
   - mixed: 同时有单患者病例和多患者数据
   - unknown: 无明确患者信息

分类逻辑说明:
- Rejection: 缺少drug或AE任意一个要素
- ICSR: (drug+AE+因果关系+单患者) OR (drug+特殊情况+单患者)
- Multiple_Patients: (drug+AE+因果关系+多患者) OR (drug+特殊情况+多患者)
- ICSR+Multiple_Patients: 同时满足ICSR和Multiple_Patients
- Other_Safety_Signal: 其他情况初筛为signal

置信度评分:
0.90-1.0: 所有字段都有明确证据
0.75-0.89: 主要字段清晰
0.60-0.74: 部分字段模糊
<0.60: 证据不足

文章内容:
---
{truncate_text(text)}
---

返回JSON格式:
{{
  "has_drug": boolean,
  "has_drug_reasoning": "判断理由：为何认为有/无目标诺华药物",
  "has_ae": boolean,
  "has_ae_reasoning": "判断理由：为何认为有/无不良事件描述",
  "has_causality": boolean,
  "has_causality_reasoning": "判断理由：为何认为有/无因果关系表述",
  "has_special_situation": boolean,
  "has_special_reasoning": "判断理由：为何认为有/无特殊情况",
  "patient_mode": "single|multiple|mixed|unknown",
  "patient_reasoning": "判断理由：为何判定为该患者模式",
  "patient_max_n": integer or null,
  "confidence": 0.0-1.0,
  "reasoning": "整体分析总结",
  "evidence": {{
    "drug": ["原文中提及药物的证据"],
    "ae": ["原文中不良事件的描述"],
    "causality": ["原文中因果关系的表述"],
    "special_situation": ["原文中特殊情况的描述"],
    "patient": ["原文中患者信息的描述，包括数量判断依据"]
  }}
}}"""

    try:
        # 使用专门的分类模型配置，默认 gpt-4o
        model = os.getenv("CLASSIFY_MODEL_NAME", "gpt-4o")
        # o1/o3 models don't support temperature parameter
        is_reasoning_model = model.startswith("o1") or model.startswith("o3")

        create_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        }

        # Only set temperature for non-reasoning models
        if not is_reasoning_model:
            create_kwargs["temperature"] = 0
            create_kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**create_kwargs)

        content = response.choices[0].message.content or "{}"
        obj = json.loads(content)

        # Extract fields
        has_drug = bool(obj.get("has_drug", False))
        has_ae = bool(obj.get("has_ae", False))
        has_causality = bool(obj.get("has_causality", False))
        has_special = bool(obj.get("has_special_situation", False))

        patient_mode = str(obj.get("patient_mode", "unknown")).lower()
        if patient_mode not in PATIENT_MODES:
            patient_mode = "unknown"

        patient_max_n = obj.get("patient_max_n")
        if patient_max_n is not None:
            try:
                patient_max_n = int(patient_max_n)
            except (ValueError, TypeError):
                patient_max_n = None

        confidence = float(obj.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        # Apply rule-based classification
        label = classify_by_rules(has_drug, has_ae, has_causality, has_special, patient_mode)

        # Extract evidence
        evidence = obj.get("evidence", {}) or {}
        drug_evidence = evidence.get("drug", []) or []
        ae_evidence = evidence.get("ae", []) or []
        causality_evidence = evidence.get("causality", []) or []
        special_evidence = evidence.get("special_situation", []) or []
        patient_evidence = evidence.get("patient", []) or []

        # Extract per-field reasoning
        has_drug_reasoning = obj.get("has_drug_reasoning", "")
        has_ae_reasoning = obj.get("has_ae_reasoning", "")
        has_causality_reasoning = obj.get("has_causality_reasoning", "")
        has_special_reasoning = obj.get("has_special_reasoning", "")
        patient_reasoning = obj.get("patient_reasoning", "")

        return ClassificationResult(
            filename=filename,
            label=label,
            label_cn=SAFETY_LABELS.get(label, "未知"),
            has_drug=has_drug,
            has_ae=has_ae,
            has_causality=has_causality,
            has_special_situation=has_special,
            patient_mode=patient_mode,
            patient_max_n=patient_max_n,
            confidence=confidence,
            drug_evidence=drug_evidence[:5],
            ae_evidence=ae_evidence[:5],
            causality_evidence=causality_evidence[:5],
            special_evidence=special_evidence[:5],
            patient_evidence=patient_evidence[:5],
            has_drug_reasoning=has_drug_reasoning,
            has_ae_reasoning=has_ae_reasoning,
            has_causality_reasoning=has_causality_reasoning,
            has_special_reasoning=has_special_reasoning,
            patient_reasoning=patient_reasoning,
            reasoning=obj.get("reasoning", ""),
            needs_review=confidence < 0.65,
            extract_method="",
            text_length=len(text),
        )

    except json.JSONDecodeError as e:
        return ClassificationResult(
            filename=filename, label="Error", label_cn="错误",
            has_drug=False, has_ae=False, has_causality=False, has_special_situation=False,
            patient_mode="unknown", patient_max_n=None, confidence=0.0,
            drug_evidence=[], ae_evidence=[], causality_evidence=[],
            special_evidence=[], patient_evidence=[],
            has_drug_reasoning="", has_ae_reasoning="", has_causality_reasoning="",
            has_special_reasoning="", patient_reasoning="", reasoning="",
            needs_review=True, extract_method="", text_length=len(text),
            error=f"JSON parse error: {e}"
        )
    except Exception as e:
        return ClassificationResult(
            filename=filename, label="Error", label_cn="错误",
            has_drug=False, has_ae=False, has_causality=False, has_special_situation=False,
            patient_mode="unknown", patient_max_n=None, confidence=0.0,
            drug_evidence=[], ae_evidence=[], causality_evidence=[],
            special_evidence=[], patient_evidence=[],
            has_drug_reasoning="", has_ae_reasoning="", has_causality_reasoning="",
            has_special_reasoning="", patient_reasoning="", reasoning="",
            needs_review=True, extract_method="", text_length=len(text),
            error=str(e)
        )


def critique_classification(
    initial_result: ClassificationResult,
    text: str,
    article_type: str,
    filename: str = ""
) -> ClassificationResult:
    """
    Self-Critique 层：审视初步判断，发现并修正常见错误。

    支持五种审核模式：
    1. has_ae 过于宽松：综述/动物实验中的AE误判
    2. has_ae 过于严格：病例/临床研究中遗漏隐含AE
    3. has_causality 过于严格：病例报告/临床研究中的隐含因果被遗漏
    4. has_special_situation 过于严格：遗漏药物无效、儿童用药、妊娠暴露等特殊情况
    5. patient_mode 案例分享误判：将"案例分享"类文献误判为multiple
    """
    # 文章类型中文映射
    type_cn_map = {
        'review': '综述/指南',
        'animal_study': '动物实验',
        'case_report': '病例报告',
        'clinical_study': '临床研究',
        'unknown': '未知'
    }
    article_type_cn = type_cn_map.get(article_type, article_type)

    # 确定审核模式
    critique_modes = []

    # 模式1: has_ae 可能过于宽松（综述/动物实验中误判AE）
    if initial_result.has_ae and article_type in ['review', 'animal_study']:
        critique_modes.append('ae_too_loose')

    # 模式2: has_causality 可能过于严格（病例/临床研究中遗漏隐含因果）
    # 修复：移除 has_ae 前置条件，因为 ae_too_strict 可能会修正 has_ae
    # 让因果审核独立于 AE 判断
    if (initial_result.has_drug and
        not initial_result.has_causality and
        article_type in ['case_report', 'clinical_study']):
        critique_modes.append('causality_too_strict')

    # 模式3: has_special_situation 可能过于严格（遗漏特殊情况）
    # 触发条件：有药物但无特殊情况，且文本中可能包含特殊情况关键词
    if (initial_result.has_drug and
        not initial_result.has_special_situation and
        article_type in ['case_report', 'clinical_study']):
        # 检查是否可能存在特殊情况关键词
        special_keywords = [
            '无效', '疗效不佳', '治疗失败', '未能控制', '控制不佳', '病情未改善',
            '换药', '停药', '更换', '调整方案',
            '儿童', '小儿', '患儿', '婴儿', '幼儿', '新生儿', '青少年',
            '妊娠', '孕妇', '怀孕', '哺乳', '母乳', '产妇',
            '过量', '中毒', '超剂量',
            '用药错误', '给药错误', '剂量错误',
            '联合用药', '药物相互作用', '合用', '配伍',
            '超说明书', '超适应症', 'off-label',
        ]
        text_lower = text.lower()
        if any(kw in text_lower for kw in special_keywords):
            critique_modes.append('special_too_strict')

    # 模式4: has_ae 可能过于严格（遗漏临床研究中的隐含AE）
    # 触发条件：has_ae=False + 病例/临床研究 + 有药物 + 文中含AE相关关键词
    if (initial_result.has_drug and
        not initial_result.has_ae and
        article_type in ['case_report', 'clinical_study']):
        ae_hint_keywords = [
            '不良反应', '记录', '观察', '监测', '安全性',
            '服用', '口服', '用药', '治疗期间'
        ]
        text_lower = text.lower()
        if any(kw in text_lower for kw in ae_hint_keywords):
            critique_modes.append('ae_too_strict')

    # 模式5: patient_mode "案例分享"误判（将案例分享误判为multiple）
    # 触发条件：patient_mode=multiple + 文件名或正文含"案例分享"
    if initial_result.patient_mode == 'multiple':
        case_sharing_keywords = ['案例分享', '病例分享', '病案分享', '案例举隅', '病案举隅']
        text_lower = text.lower()
        filename_lower = filename.lower()
        if (any(kw in text_lower for kw in case_sharing_keywords) or
            any(kw in filename_lower for kw in case_sharing_keywords)):
            critique_modes.append('patient_mode_case_sharing')

    if not critique_modes:
        return initial_result

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return initial_result

    client = OpenAI(api_key=api_key)
    result = initial_result

    # 依次执行每种审核模式
    for mode in critique_modes:
        if mode == 'ae_too_loose':
            result = _critique_ae_too_loose(client, result, text, article_type_cn)
        elif mode == 'causality_too_strict':
            result = _critique_causality_too_strict(client, result, text, article_type_cn)
        elif mode == 'special_too_strict':
            result = _critique_special_too_strict(client, result, text, article_type_cn)
        elif mode == 'ae_too_strict':
            result = _critique_ae_too_strict(client, result, text, article_type_cn)
        elif mode == 'patient_mode_case_sharing':
            result = _critique_patient_mode_case_sharing(client, result, text, filename)

    return result


def _critique_ae_too_loose(
    client,
    initial_result: ClassificationResult,
    text: str,
    article_type_cn: str
) -> ClassificationResult:
    """审核 has_ae 是否过于宽松（综述/动物实验误判）"""

    critique_prompt = f"""你是药物安全分类审核专家。请审视以下分类判断是否存在常见错误。

## 初步判断
- has_ae: {initial_result.has_ae}
- has_ae_reasoning: {initial_result.has_ae_reasoning}
- 文章类型: {article_type_cn}

## 需检查的常见错误
1. 【综述误判】综述/指南中仅泛泛讨论药物可能的副作用（如"该药物可能导致XX"），无具体病例报告，不应判定 has_ae=True
2. 【动物实验】纯动物实验中的毒性反应（如大鼠肝损伤）不算人体AE，不应判定 has_ae=True
3. 【疾病症状】疾病本身的症状（如神经内分泌肿瘤的腹泻、潮红）不是药物AE

## 相关原文片段
{text[:4000]}

## 请判断
1. 初步判断是否存在上述错误？
2. 如存在错误，has_ae 应该修正为什么？
3. 给出修正理由。

返回JSON:
{{
    "has_error": boolean,
    "corrected_has_ae": boolean,
    "correction_reasoning": "修正理由"
}}"""

    try:
        model = os.getenv("CLASSIFY_MODEL_NAME", "gpt-4o")
        is_reasoning_model = model.startswith("o1") or model.startswith("o3")

        create_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": critique_prompt}],
        }

        if not is_reasoning_model:
            create_kwargs["temperature"] = 0
            create_kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**create_kwargs)
        content = response.choices[0].message.content or "{}"
        critique_result = json.loads(content)

        if critique_result.get("has_error"):
            corrected_has_ae = critique_result.get("corrected_has_ae", False)
            correction_reasoning = critique_result.get("correction_reasoning", "")

            # 重新应用规则判断
            new_label = classify_by_rules(
                initial_result.has_drug,
                corrected_has_ae,
                initial_result.has_causality,
                initial_result.has_special_situation,
                initial_result.patient_mode
            )

            return replace(
                initial_result,
                has_ae=corrected_has_ae,
                has_ae_reasoning=f"{initial_result.has_ae_reasoning}\n[Self-Critique:AE修正]: {correction_reasoning}",
                label=new_label,
                label_cn=SAFETY_LABELS.get(new_label, "未知")
            )

    except Exception as e:
        print(f"      ⚠️ Self-Critique (AE) error: {e}")

    return initial_result


def _critique_causality_too_strict(
    client,
    initial_result: ClassificationResult,
    text: str,
    article_type_cn: str
) -> ClassificationResult:
    """审核 has_causality 是否过于严格（病例/临床研究中遗漏隐含因果）"""

    critique_prompt = f"""你是药物安全分类审核专家。请审视以下分类判断是否遗漏了文章中的因果关系证据。

## 初步判断
- has_ae: {initial_result.has_ae}
- has_causality: {initial_result.has_causality}
- has_causality_reasoning: {initial_result.has_causality_reasoning}
- 文章类型: {article_type_cn}

## 重要原则
has_causality 的判断目的是确定文章是否包含药物-AE因果分析信息，用于药物警戒文献筛选：
- ⚠️ 即使AE是由文中其他药物（非目标监测药物）引起的，只要文章包含明确的药物-AE因果关系表述，has_causality仍应为True
- 这样做是为了确保包含安全性信息的文献能被正确标记，供人工审核

## 需检查的遗漏情况
在病例报告或临床研究中，以下情况应视为存在因果关系（has_causality=True）：

1. 【明确归因】文章中任何药物被明确归因为AE的原因（如"X药导致Y症状"、"Y由X引起"）
2. 【病例报告因果】病例报告中描述"用药后出现XX症状"，即使未明确说"导致"，也应视为存在因果
3. 【临床研究AE】临床研究/试验中记录的不良反应发生率（如"治疗组不良反应发生率15%"），应视为存在隐含因果
4. 【时间关联】明确的时间关联表述（如"服药3天后出现"、"治疗期间发生"）应视为因果证据
5. 【去激发/再激发】停药后症状缓解、再用药后复发，是强因果证据

## 相关原文片段
{text[:4000]}

## 请判断
1. 文章中是否存在任何药物-AE因果关系的表述（不限于目标药物）？
2. 如果初步判断遗漏了因果关系，has_causality 应该修正为True
3. 给出修正理由和具体证据。

返回JSON:
{{
    "has_error": boolean,
    "corrected_has_causality": boolean,
    "correction_reasoning": "修正理由"
}}"""

    try:
        model = os.getenv("CLASSIFY_MODEL_NAME", "gpt-4o")
        is_reasoning_model = model.startswith("o1") or model.startswith("o3")

        create_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": critique_prompt}],
        }

        if not is_reasoning_model:
            create_kwargs["temperature"] = 0
            create_kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**create_kwargs)
        content = response.choices[0].message.content or "{}"
        critique_result = json.loads(content)

        if critique_result.get("has_error"):
            corrected_has_causality = critique_result.get("corrected_has_causality", True)
            correction_reasoning = critique_result.get("correction_reasoning", "")

            # 重新应用规则判断
            new_label = classify_by_rules(
                initial_result.has_drug,
                initial_result.has_ae,
                corrected_has_causality,
                initial_result.has_special_situation,
                initial_result.patient_mode
            )

            return replace(
                initial_result,
                has_causality=corrected_has_causality,
                has_causality_reasoning=f"{initial_result.has_causality_reasoning}\n[Self-Critique:因果修正]: {correction_reasoning}",
                label=new_label,
                label_cn=SAFETY_LABELS.get(new_label, "未知")
            )

    except Exception as e:
        print(f"      ⚠️ Self-Critique (Causality) error: {e}")

    return initial_result


def _critique_special_too_strict(
    client,
    initial_result: ClassificationResult,
    text: str,
    article_type_cn: str
) -> ClassificationResult:
    """审核 has_special_situation 是否过于严格（遗漏特殊情况）"""

    critique_prompt = f"""你是药物安全分类审核专家。请审视以下分类判断是否遗漏了特殊情况。

## 初步判断
- has_drug: {initial_result.has_drug}
- has_special_situation: {initial_result.has_special_situation}
- has_special_reasoning: {initial_result.has_special_reasoning}
- 文章类型: {article_type_cn}

## 需检查的特殊情况（任一存在即应判定 has_special_situation=True）

1. 【药物无效/疗效不佳】⚠️ 这是最常遗漏的特殊情况
   - 关键词："无效"、"疗效不佳"、"治疗失败"、"未能控制"、"控制不佳"、"病情未改善"
   - 关键词："换药"、"更换治疗方案"、"调整用药"、"效果欠佳"
   - 注意：即使文章主题不是讨论药物无效，只要提到目标药物"无效/失败"就算

2. 【儿童用药】
   - 患者为儿童、婴幼儿、青少年（<18岁）
   - 关键词："患儿"、"小儿"、"儿童"、"婴儿"、"幼儿"、"新生儿"

3. 【妊娠/哺乳期暴露】
   - 患者为孕妇或哺乳期妇女
   - 关键词："妊娠"、"孕妇"、"怀孕"、"哺乳"、"母乳"、"产妇"

4. 【过量/中毒】
   - 药物过量使用或中毒
   - 关键词："过量"、"中毒"、"超剂量"

5. 【用药错误】
   - 给药错误、剂量错误、用法错误
   - 关键词："用药错误"、"给药错误"、"剂量错误"

6. 【药物相互作用】
   - 与其他药物的相互作用导致问题
   - 关键词："药物相互作用"、"联合用药不良反应"

7. 【超说明书用药】
   - 超适应症、超剂量、超人群使用
   - 关键词："超说明书"、"超适应症"、"off-label"

## 相关原文片段
{text[:4000]}

## 请判断
1. 初步判断是否遗漏了上述任一特殊情况？
2. 如有遗漏，has_special_situation 应该修正为什么？
3. 给出修正理由和具体证据。

返回JSON:
{{
    "has_error": boolean,
    "corrected_has_special": boolean,
    "correction_reasoning": "修正理由，包括具体是哪种特殊情况"
}}"""

    try:
        model = os.getenv("CLASSIFY_MODEL_NAME", "gpt-4o")
        is_reasoning_model = model.startswith("o1") or model.startswith("o3")

        create_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": critique_prompt}],
        }

        if not is_reasoning_model:
            create_kwargs["temperature"] = 0
            create_kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**create_kwargs)
        content = response.choices[0].message.content or "{}"
        critique_result = json.loads(content)

        if critique_result.get("has_error"):
            corrected_has_special = critique_result.get("corrected_has_special", True)
            correction_reasoning = critique_result.get("correction_reasoning", "")

            # 重新应用规则判断
            new_label = classify_by_rules(
                initial_result.has_drug,
                initial_result.has_ae,
                initial_result.has_causality,
                corrected_has_special,
                initial_result.patient_mode
            )

            return replace(
                initial_result,
                has_special_situation=corrected_has_special,
                has_special_reasoning=f"{initial_result.has_special_reasoning}\n[Self-Critique:特殊情况修正]: {correction_reasoning}",
                label=new_label,
                label_cn=SAFETY_LABELS.get(new_label, "未知")
            )

    except Exception as e:
        print(f"      ⚠️ Self-Critique (Special) error: {e}")

    return initial_result


def _critique_ae_too_strict(
    client,
    initial_result: ClassificationResult,
    text: str,
    article_type_cn: str
) -> ClassificationResult:
    """审核 has_ae 是否过于严格（遗漏了临床研究中的隐含AE）"""

    critique_prompt = f"""你是药物安全分类审核专家。请审视以下分类判断是否遗漏了隐含的不良事件信息。

## 初步判断
- has_drug: {initial_result.has_drug}
- has_ae: {initial_result.has_ae} (当前判断为False)
- has_ae_reasoning: {initial_result.has_ae_reasoning}
- 文章类型: {article_type_cn}

## 需检查的遗漏情况

1. 【临床研究隐含AE】
   - 如果是临床研究/对照研究，且"不良反应"作为观察指标
   - 关键词："记录不良反应"、"观察不良反应"、"不良反应发生率"
   - 关键词："两组不良反应比较"、"治疗组vs对照组"
   - 即使全文未详细列出AE，研究设计本身隐含了AE监测

2. 【病例报告背景用药】
   - 病例报告中患者有明确的目标药物用药记录
   - 即使主要AE不是目标药物引起，背景用药构成安全监测场景
   - 关键词："服用/口服[目标药物]"、"既往用药"

3. 【治疗期间观察】
   - 临床研究中"治疗期间密切观察/监测"
   - 隐含了对潜在AE的关注

## 相关原文片段
{text[:4000]}

## 请判断
1. 初步判断是否遗漏了上述任一隐含AE情况？
2. 如有遗漏，has_ae 应该修正为什么？
3. 给出修正理由和具体证据。

返回JSON:
{{
    "has_error": boolean,
    "corrected_has_ae": boolean,
    "correction_reasoning": "修正理由"
}}"""

    try:
        model = os.getenv("CLASSIFY_MODEL_NAME", "gpt-4o")
        is_reasoning_model = model.startswith("o1") or model.startswith("o3")

        create_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": critique_prompt}],
        }

        if not is_reasoning_model:
            create_kwargs["temperature"] = 0
            create_kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**create_kwargs)
        content = response.choices[0].message.content or "{}"
        critique_result = json.loads(content)

        if critique_result.get("has_error"):
            corrected_has_ae = critique_result.get("corrected_has_ae", True)
            correction_reasoning = critique_result.get("correction_reasoning", "")

            # 重新应用规则判断
            new_label = classify_by_rules(
                initial_result.has_drug,
                corrected_has_ae,
                initial_result.has_causality,
                initial_result.has_special_situation,
                initial_result.patient_mode
            )

            return replace(
                initial_result,
                has_ae=corrected_has_ae,
                has_ae_reasoning=f"{initial_result.has_ae_reasoning}\n[Self-Critique:AE过严修正]: {correction_reasoning}",
                label=new_label,
                label_cn=SAFETY_LABELS.get(new_label, "未知")
            )

    except Exception as e:
        print(f"      ⚠️ Self-Critique (AE过严) error: {e}")

    return initial_result


def _critique_patient_mode_case_sharing(
    client,
    initial_result: ClassificationResult,
    text: str,
    filename: str
) -> ClassificationResult:
    """审核 patient_mode 是否将"案例分享"误判为 multiple"""

    critique_prompt = f"""你是药物安全分类审核专家。请审视患者模式判断是否正确。

## 初步判断
- patient_mode: {initial_result.patient_mode} (当前判断为multiple)
- patient_reasoning: {initial_result.patient_reasoning}
- 文件名: {filename}

## 需检查的误判情况

**"案例分享"类文献特殊规则**：
- 如果文章类型是"案例分享"/"病例分享"/"病案分享"
- 即使包含多个病例（如"病案1"、"病案2"、"案例一"、"案例二"）
- 每个病例都是**独立的单患者报告(ICSR)**
- 应该判断为 patient_mode="single"，而非 "multiple"

## 判断依据
- 标题或正文含"案例分享"/"病例分享"/"病案分享" → single
- 正文结构为"病案1...病案2..." → 多个独立单例，算single
- 明确样本量"纳入XX例"并做统计分析 → 才是真正的 multiple

## 相关原文片段
{text[:3000]}

## 请判断
1. 该文献是否为"案例分享"类型？
2. 如果是，patient_mode 应该修正为 "single" 吗？
3. 给出修正理由。

返回JSON:
{{
    "has_error": boolean,
    "corrected_patient_mode": "single" or "multiple",
    "correction_reasoning": "修正理由"
}}"""

    try:
        model = os.getenv("CLASSIFY_MODEL_NAME", "gpt-4o")
        is_reasoning_model = model.startswith("o1") or model.startswith("o3")

        create_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": critique_prompt}],
        }

        if not is_reasoning_model:
            create_kwargs["temperature"] = 0
            create_kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**create_kwargs)
        content = response.choices[0].message.content or "{}"
        critique_result = json.loads(content)

        if critique_result.get("has_error"):
            corrected_patient_mode = critique_result.get("corrected_patient_mode", "single")
            correction_reasoning = critique_result.get("correction_reasoning", "")

            # 重新应用规则判断
            new_label = classify_by_rules(
                initial_result.has_drug,
                initial_result.has_ae,
                initial_result.has_causality,
                initial_result.has_special_situation,
                corrected_patient_mode
            )

            return replace(
                initial_result,
                patient_mode=corrected_patient_mode,
                patient_reasoning=f"{initial_result.patient_reasoning}\n[Self-Critique:案例分享修正]: {correction_reasoning}",
                label=new_label,
                label_cn=SAFETY_LABELS.get(new_label, "未知")
            )

    except Exception as e:
        print(f"      ⚠️ Self-Critique (案例分享) error: {e}")

    return initial_result


def load_drug_keywords(path: Path) -> list[str]:
    """Load drug keywords from file."""
    if not path.exists():
        return []
    keywords = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keywords.append(line)
    return keywords


def classify_papers(
    input_dir: Path,
    output_path: Path,
    drug_keywords: list[str],
    max_papers: int = 0,
) -> list[ClassificationResult]:
    """Classify all papers in input directory."""
    pdf_files = sorted(input_dir.glob("*.pdf"))
    total = len(pdf_files)

    if max_papers > 0:
        pdf_files = pdf_files[:max_papers]

    print(f"\n📚 Classifying {len(pdf_files)} papers (from {total} total)")
    print(f"   Drug keywords: {len(drug_keywords)}")
    print("=" * 60)

    results: list[ClassificationResult] = []

    for idx, pdf_path in enumerate(pdf_files, 1):
        filename = pdf_path.name
        print(f"\n[{idx}/{len(pdf_files)}] 📄 {filename[:50]}...")

        # Extract text
        print("      Extracting text...")
        text, method = extract_pdf_text(pdf_path)

        if not text.strip():
            print("      ❌ Could not extract text")
            results.append(ClassificationResult(
                filename=filename, label="Error", label_cn="错误",
                has_drug=False, has_ae=False, has_causality=False, has_special_situation=False,
                patient_mode="unknown", patient_max_n=None, confidence=0.0,
                drug_evidence=[], ae_evidence=[], causality_evidence=[],
                special_evidence=[], patient_evidence=[],
                has_drug_reasoning="", has_ae_reasoning="", has_causality_reasoning="",
                has_special_reasoning="", patient_reasoning="", reasoning="",
                needs_review=True, extract_method=method, text_length=0,
                error="Text extraction failed"
            ))
            continue

        print(f"      Extracted {len(text)} chars via {method}")

        # 选择分类模式
        classify_mode = os.getenv("CLASSIFY_MODE", "default").lower()

        if classify_mode == "multi_agent" and MULTI_AGENT_AVAILABLE:
            # Multi-Agent 辩论模式
            print("      🤖 Classifying with Multi-Agent debate...")
            ma_result = classify_with_multi_agent(text, filename, drug_keywords)

            # 转换为 ClassificationResult
            result = ClassificationResult(
                filename=filename,
                label=ma_result.final_label,
                label_cn=ma_result.final_label_cn,
                has_drug=ma_result.has_drug,
                has_ae=ma_result.has_ae,
                has_causality=ma_result.has_causality,
                has_special_situation=ma_result.has_special_situation,
                patient_mode=ma_result.patient_mode,
                patient_max_n=ma_result.patient_max_n,
                confidence=ma_result.confidence,
                drug_evidence=ma_result.pharmacologist.judgments.get("has_drug_evidence", []),
                ae_evidence=ma_result.pharmacologist.judgments.get("has_ae_evidence", []),
                causality_evidence=ma_result.clinician.judgments.get("causality_evidence", []),
                special_evidence=ma_result.analyst.judgments.get("special_evidence", []),
                patient_evidence=ma_result.clinician.judgments.get("patient_evidence", []),
                has_drug_reasoning=f"[药物学专家] {ma_result.pharmacologist.reasoning}",
                has_ae_reasoning=f"[药物学专家] {ma_result.pharmacologist.reasoning}",
                has_causality_reasoning=f"[临床医生] {ma_result.clinician.reasoning}",
                has_special_reasoning=f"[文献分析] {ma_result.analyst.reasoning}",
                patient_reasoning=f"[临床医生] {ma_result.clinician.reasoning}",
                reasoning=ma_result.reasoning,
                needs_review=ma_result.needs_review,
                extract_method=method,
                text_length=len(text),
            )
        else:
            # 原有分类模式
            print("      Classifying with LLM...")
            result = classify_with_openai(text, filename, drug_keywords)
            result.extract_method = method

            # Self-Critique 层（可选，通过环境变量控制）
            if os.getenv("ENABLE_SELF_CRITIQUE", "false").lower() == "true":
                article_type_result = detect_article_type(text, filename)
                original_label = result.label
                original_has_ae = result.has_ae
                original_has_causality = result.has_causality
                original_has_special = result.has_special_situation
                original_patient_mode = result.patient_mode
                result = critique_classification(result, text, article_type_result['type'], filename)

                # 输出修正信息
                corrections = []
                if result.has_ae != original_has_ae:
                    corrections.append(f"AE:{original_has_ae}→{result.has_ae}")
                if result.has_causality != original_has_causality:
                    corrections.append(f"因果:{original_has_causality}→{result.has_causality}")
                if result.has_special_situation != original_has_special:
                    corrections.append(f"特殊:{original_has_special}→{result.has_special_situation}")
                if result.patient_mode != original_patient_mode:
                    corrections.append(f"患者:{original_patient_mode}→{result.patient_mode}")
                if corrections:
                    print(f"      🔄 Self-Critique [{', '.join(corrections)}]: {original_label} → {result.label}")

        results.append(result)

        if result.error:
            print(f"      ❌ Error: {result.error}")
        else:
            print(f"      ✅ {result.label} ({result.label_cn})")
            print(f"         Confidence: {result.confidence:.2f}")
            flags = []
            if result.has_drug:
                flags.append("Drug✓")
            if result.has_ae:
                flags.append("AE✓")
            if result.has_causality:
                flags.append("Causality✓")
            if result.has_special_situation:
                flags.append("Special✓")
            print(f"         Flags: {' '.join(flags) or 'None'}")
            if result.needs_review:
                print("         ⚠️ Needs human review")

    # Write results to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename", "label", "label_cn", "confidence", "needs_review",
        "has_drug", "has_ae", "has_causality", "has_special_situation",
        "patient_mode", "patient_max_n",
        "drug_evidence", "ae_evidence", "causality_evidence", "special_evidence", "patient_evidence",
        "has_drug_reasoning", "has_ae_reasoning", "has_causality_reasoning",
        "has_special_reasoning", "patient_reasoning",
        "reasoning", "extract_method", "text_length", "classify_time", "error"
    ]

    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            # Convert lists to strings
            for key in ["drug_evidence", "ae_evidence", "causality_evidence", "special_evidence", "patient_evidence"]:
                row[key] = "; ".join(row[key]) if row[key] else ""
            writer.writerow(row)

    # Summary
    print("\n" + "=" * 60)
    print("📊 Classification Summary:")

    label_counts: dict[str, int] = {}
    error_count = 0
    review_count = 0
    for r in results:
        if r.error:
            error_count += 1
        else:
            label_counts[r.label] = label_counts.get(r.label, 0) + 1
            if r.needs_review:
                review_count += 1

    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"   {label}: {count}")

    if error_count:
        print(f"   Errors: {error_count}")
    if review_count:
        print(f"   ⚠️ Needs review: {review_count}")

    print(f"\n📁 Results saved to: {output_path}")
    print("=" * 60)

    return results


# ============================================================
# V2 结构化抽取架构 - Phase 6: 集成入口
# ============================================================

def classify_papers_v2(
    input_dir: Path,
    output_path: Path,
    drug_keywords: list[str],
    max_papers: int = 0,
) -> list[ClassificationResultV2]:
    """V2版本的批量分类入口。

    Args:
        input_dir: 输入目录（包含PDF文件）
        output_path: 输出CSV文件路径
        drug_keywords: 药物关键词列表
        max_papers: 最大分类数量（0表示无限制）

    Returns:
        list[ClassificationResultV2]: 分类结果列表
    """
    pdf_files = sorted(input_dir.glob("*.pdf"))
    total = len(pdf_files)

    if max_papers > 0:
        pdf_files = pdf_files[:max_papers]

    print(f"\n📚 [V2] Classifying {len(pdf_files)} papers (from {total} total)")
    print(f"   Drug keywords: {len(drug_keywords)}")
    print(f"   Mode: Structured Extraction (v15)")
    print("=" * 60)

    results: list[ClassificationResultV2] = []

    # 写入CSV（V2格式）- 增量写入，避免中途中断丢失进度
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames_v2 = [
        "filename", "label", "label_cn", "confidence", "needs_review",
        "novartis_exposure", "has_qualifying_event", "decision_path",
        "body_drug_count", "ae_event_count", "human_ae_count",
        "audit_rules", "audit_corrections",
        "reasoning", "extract_method", "text_length", "classify_time", "error"
    ]

    def to_csv_row(result: ClassificationResultV2) -> dict:
        body_drugs = [dm for dm in result.extraction.drug_mentions if dm.where == "body"]
        human_ae = [ae for ae in result.extraction.ae_events if ae.human_flag == "human"]
        return {
            "filename": result.filename,
            "label": result.label,
            "label_cn": result.label_cn,
            "confidence": result.confidence,
            "needs_review": result.needs_review,
            "novartis_exposure": result.novartis_exposure,
            "has_qualifying_event": result.has_qualifying_event,
            "decision_path": result.decision_path,
            "body_drug_count": len(body_drugs),
            "ae_event_count": len(result.extraction.ae_events),
            "human_ae_count": len(human_ae),
            "audit_rules": "; ".join(result.audit.triggered_rules),
            "audit_corrections": "; ".join(
                f"{k}:{reason}" for k, (old, new, reason) in result.audit.corrections.items()
            ),
            "reasoning": result.reasoning,
            "extract_method": result.extract_method,
            "text_length": result.text_length,
            "classify_time": result.classify_time,
            "error": result.error,
        }

    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_v2, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for idx, pdf_path in enumerate(pdf_files, 1):
            filename = pdf_path.name
            print(f"\n[{idx}/{len(pdf_files)}] 📄 {filename[:50]}...")

            # V2分类流程
            print("      🔄 Parsing document...")
            result = classify_paper_v2(pdf_path, drug_keywords)
            results.append(result)

            # 增量写入并flush
            writer.writerow(to_csv_row(result))
            f.flush()

            if result.error:
                print(f"      ❌ Error: {result.error}")
            else:
                print(f"      ✅ {result.label} ({result.label_cn})")
                print(f"         Confidence: {result.confidence:.2f}")
                print(f"         Decision: {result.decision_path}")
                # 显示抽取统计
                body_drugs = len([dm for dm in result.extraction.drug_mentions if dm.where == "body"])
                ae_count = len(result.extraction.ae_events)
                print(f"         Extracted: {body_drugs} body drugs, {ae_count} AE events")
                # 显示审计信息
                if result.audit.triggered_rules:
                    print(f"         Audit: {', '.join(result.audit.triggered_rules)}")
                if result.needs_review:
                    print("         ⚠️ Needs human review")

    # Summary
    print("\n" + "=" * 60)
    print("📊 [V2] Classification Summary:")

    label_counts: dict[str, int] = {}
    error_count = 0
    review_count = 0
    for r in results:
        if r.error:
            error_count += 1
        else:
            label_counts[r.label] = label_counts.get(r.label, 0) + 1
            if r.needs_review:
                review_count += 1

    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"   {label}: {count}")

    if error_count:
        print(f"   Errors: {error_count}")
    if review_count:
        print(f"   ⚠️ Needs review: {review_count}")

    print(f"\n📁 Results saved to: {output_path}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Wanfang Paper Safety Classification Script")
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        default=PAPERS_DIR,
        help=f"Directory containing PDF files (default: {PAPERS_DIR})",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--drugs", "-d",
        type=str,
        default="",
        help="Comma-separated drug keywords (e.g., '替格瑞洛,ticagrelor')",
    )
    parser.add_argument(
        "--drugs-file", "-f",
        type=Path,
        default=None,
        help="Path to drug keywords file (one per line)",
    )
    parser.add_argument(
        "--max-papers", "-m",
        type=int,
        default=0,
        help="Maximum papers to classify (0 = unlimited)",
    )
    parser.add_argument(
        "--use-v2",
        action="store_true",
        help="Use V2 structured extraction architecture (v15)",
    )

    args = parser.parse_args()

    # Load drug keywords
    drug_keywords = []
    if args.drugs:
        drug_keywords.extend([k.strip() for k in args.drugs.split(",") if k.strip()])
    if args.drugs_file:
        drug_keywords.extend(load_drug_keywords(args.drugs_file))

    # 如果没有提供药物关键词，尝试加载默认清单
    if not drug_keywords and DEFAULT_DRUGS_FILE.exists():
        drug_keywords.extend(load_drug_keywords(DEFAULT_DRUGS_FILE))
        print(f"Loaded default drug keywords from: {DEFAULT_DRUGS_FILE}")

    if not drug_keywords:
        print("Warning: No drug keywords provided. Use --drugs or --drugs-file")

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1

    pdf_count = len(list(args.input_dir.glob("*.pdf")))
    if pdf_count == 0:
        print(f"Error: No PDF files found in {args.input_dir}")
        return 1

    print("=" * 60)
    print("📚 Wanfang Paper Safety Classification")
    print("=" * 60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output}")
    print(f"Drug keywords: {len(drug_keywords)}")
    if drug_keywords:
        print(f"   Examples: {', '.join(drug_keywords[:5])}")
    print(f"Max papers: {args.max_papers if args.max_papers > 0 else 'unlimited'}")
    print(f"Found {pdf_count} PDF files")
    print(f"Mode: {'V2 Structured Extraction' if args.use_v2 else 'V1 (5-Boolean)'}")
    print("=" * 60)

    if args.use_v2:
        # 使用 V2 结构化抽取架构
        results = classify_papers_v2(
            input_dir=args.input_dir,
            output_path=args.output,
            drug_keywords=drug_keywords,
            max_papers=args.max_papers,
        )
    else:
        # 使用原有 V1 架构
        results = classify_papers(
            input_dir=args.input_dir,
            output_path=args.output,
            drug_keywords=drug_keywords,
            max_papers=args.max_papers,
        )

    error_count = sum(1 for r in results if r.error)
    return 1 if error_count == len(results) else 0


if __name__ == "__main__":
    exit(main())
