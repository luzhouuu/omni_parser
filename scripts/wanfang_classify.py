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
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Directories
DATA_DIR = Path(__file__).parent.parent / "data"
PAPERS_DIR = DATA_DIR / "papers"
DEFAULT_OUTPUT = DATA_DIR / "classification_results.csv"


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
    reasoning: str
    needs_review: bool
    extract_method: str
    text_length: int
    classify_time: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    error: str = ""


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


def classify_by_rules(
    has_drug: bool,
    has_ae: bool,
    has_causality: bool,
    has_special_situation: bool,
    patient_mode: str,
) -> str:
    """Rule-based classification logic.

    分类判断逻辑：
    1. Rejection：文章中缺少drug(诺华药)或AE(不良事件)任意一个要素
    2. ICSR：(drug+AE+因果关系+单个患者) OR (drug+特殊情况+单个患者)
    3. Multiple_Patients：(drug+AE+因果关系+多个患者) OR (drug+特殊情况+多个患者)
    4. ICSR+Multiple_Patients：一篇文章同时满足ICSR和Multiple_Patients的条件
    5. Other_Safety_Signal：不符合上面类型的都初筛成signal
    """
    # Rejection: 缺少 drug 或 AE 任意一个要素
    if not has_drug or not has_ae:
        return "Rejection"

    # 满足ICSR/Multiple_Patients的条件：
    # - (AE + 因果关系) OR 特殊情况
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

    # 其他情况（unknown等）都初筛成signal
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
            special_evidence=[], patient_evidence=[], reasoning="",
            needs_review=True, extract_method="", text_length=0,
            error="OPENAI_API_KEY not set"
        )

    client = OpenAI(api_key=api_key)
    drug_hint = ", ".join(drug_keywords[:100]) if drug_keywords else "(未提供药物关键词)"

    system_prompt = """你是一位资深的药物警戒信息提取专家。
你的任务是从医学/科学文献中提取关键安全信息，用于诺华药物安全监测。

文献检索业务背景：
在全文范围内以中英文商品名&活性成分名作为关键词进行检索，检索出上抛到CNKI & Wanfang数据库中的文献。
针对所有检索出来的文献进行审阅，识别文章中是否提及任何诺华药相关安全病例或潜在信号。

分类判断逻辑：
1. Rejection：文章中缺少drug(诺华药)或AE(不良事件)任意一个要素
2. ICSR：(drug+AE+因果关系+单个患者) OR (drug+特殊情况+单个患者)
3. Multiple_Patients：(drug+AE+因果关系+多个患者) OR (drug+特殊情况+多个患者)
4. ICSR+Multiple_Patients：一篇文章同时满足ICSR和Multiple_Patients的条件
5. Other_Safety_Signal：不符合上面类型的都初筛成signal

需要提取的字段：

1. **has_drug** (boolean): 文章是否提及目标诺华药物？
   - 使用提供的药物关键词列表（中英文商品名、活性成分名）作为参考
   - 注意：PDF文件名前缀通常包含对应的诺华产品名

2. **has_ae** (boolean): 是否描述了任何不良事件(AE)？
   - 副作用、毒性反应、不良反应、安全事件
   - 任何可能与药物使用相关的负面健康结果

3. **has_causality** (boolean): 是否有明确的因果关系表述将药物与事件联系起来？
   - YES: "与...相关"、"由...引起"、"归因于"、"药物诱发"、"治疗相关"
   - YES: "怀疑与...相关"、阳性再激发/去激发试验
   - NO: 仅有时间关联而无归因
   - NO: 否定陈述（"与...无关"、"不相关"）
   - NO: 仅有人群统计数据而无个体归因

4. **has_special_situation** (boolean): 是否存在以下特殊情况？
   - 妊娠/哺乳期暴露 (Pregnancy/lactation exposure)
   - 儿童用药 (Pediatric use - children, infants)
   - 药物无效/疗效不佳 (Lack of efficacy/therapeutic failure)
   - 过量 (Overdose)
   - 用药错误 (Medication error)
   - 药物相互作用 (Drug-drug interaction)
   - 超说明书用药 (Off-label use)

5. **patient_mode** (string): 患者识别
   - "single": 单个可识别患者 (n=1, 病例报告, 有年龄/性别可判断单一患者存在, 或文章提到"1例")
   - "multiple": 多个患者 (n>1, 作为队列描述)
   - "mixed": 文章中同时存在单患者部分和多患者部分
   - "unknown": 无明确患者信息或仅有汇总统计数据

仅返回包含这些字段和证据数组的JSON对象。"""

    user_prompt = f"""目标诺华药物关键词（中英文商品名 & 活性成分名）:
{drug_hint}

提取步骤:
1. 仔细阅读文章全文
2. 识别是否提到目标诺华药物（注意：PDF前缀通常包含对应产品名）
3. 查找是否描述了不良事件(AE)
4. 查找是否有明确的因果关系表述（"与药物相关"、"药物引起"等）
5. 检查是否存在特殊情况（儿童用药、药物无效、怀孕暴露等）
6. 判断患者数量:
   - single: 单个患者（年龄性别可判断单一患者存在，或文章提到"1例"）
   - multiple: 多个患者（>1例）
   - mixed: 同时有单患者和多患者描述
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
  "has_ae": boolean,
  "has_causality": boolean,
  "has_special_situation": boolean,
  "patient_mode": "single|multiple|mixed|unknown",
  "patient_max_n": integer or null,
  "confidence": 0.0-1.0,
  "reasoning": "简要说明分析过程，包括为何判定为某个分类",
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
            special_evidence=[], patient_evidence=[], reasoning="",
            needs_review=True, extract_method="", text_length=len(text),
            error=f"JSON parse error: {e}"
        )
    except Exception as e:
        return ClassificationResult(
            filename=filename, label="Error", label_cn="错误",
            has_drug=False, has_ae=False, has_causality=False, has_special_situation=False,
            patient_mode="unknown", patient_max_n=None, confidence=0.0,
            drug_evidence=[], ae_evidence=[], causality_evidence=[],
            special_evidence=[], patient_evidence=[], reasoning="",
            needs_review=True, extract_method="", text_length=len(text),
            error=str(e)
        )


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
                special_evidence=[], patient_evidence=[], reasoning="",
                needs_review=True, extract_method=method, text_length=0,
                error="Text extraction failed"
            ))
            continue

        print(f"      Extracted {len(text)} chars via {method}")

        # Classify
        print("      Classifying with LLM...")
        result = classify_with_openai(text, filename, drug_keywords)
        result.extract_method = method
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

    args = parser.parse_args()

    # Load drug keywords
    drug_keywords = []
    if args.drugs:
        drug_keywords.extend([k.strip() for k in args.drugs.split(",") if k.strip()])
    if args.drugs_file:
        drug_keywords.extend(load_drug_keywords(args.drugs_file))

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
    print("=" * 60)

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
