/**
 * PDF Classification Page
 */

// DOM Elements
const fileInput = document.getElementById('file');
const fileMeta = document.getElementById('fileMeta');
const maxPagesInput = document.getElementById('maxPages');
const drugKeywordsInput = document.getElementById('drug-keywords');
const btn = document.getElementById('btn');
const status = document.getElementById('status');
const resultsDiv = document.getElementById('results');
const resultsSummary = document.getElementById('results-summary');

// Label display mapping
const LABEL_MAP = {
    "Rejection": { text: "Rejection", desc: "拒绝", class: "bad" },
    "ICSR": { text: "ICSR", desc: "个例安全报告", class: "ok" },
    "Multiple_Patients": { text: "Multiple_Patients", desc: "多患者报告", class: "info" },
    "ICSR+Multiple_Patients": { text: "ICSR+Multiple_Patients", desc: "混合报告", class: "warn" },
    "Other_Safety_Signal": { text: "Other_Safety_Signal", desc: "其他安全信号", class: "warn" },
    "LLM_ERROR": { text: "LLM_ERROR", desc: "处理错误", class: "bad" }
};

// Patient mode mapping
const PATIENT_MODE_MAP = {
    "single": "单个患者",
    "multiple": "多个患者",
    "mixed": "混合",
    "unknown": "未知"
};

// Extract method mapping
const METHOD_MAP = {
    "pdftotext": "PDF文本",
    "pymupdf": "PyMuPDF",
    "pdf_ocr": "PDF OCR",
    "tesseract": "图片OCR",
    "txt": "文本文件",
    "none": "无法提取"
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    fileInput.addEventListener('change', handleFileChange);
    btn.addEventListener('click', onSubmit);
    loadHealth();
});

// Load health info
async function loadHealth() {
    try {
        const res = await fetch("/api/health");
        const data = await res.json();
        const env = data.env || {};
        const toolPill = (ok) => `<span class="pill ${ok ? "ok" : "bad"}">${ok ? "正常" : "缺失"}</span>`;

        document.getElementById("health").innerHTML =
            `OpenAI ${toolPill(!!env.openai_api_key)} | ` +
            `分类模型: <code>${env.classify_model || "gpt-4o"}</code> | ` +
            `pdftotext ${toolPill(!!data.tools?.pdftotext)}`;
    } catch (e) {
        document.getElementById("health").textContent = "健康检查失败";
    }
}

// Handle file selection change
function handleFileChange() {
    const files = fileInput.files;
    if (!files || files.length === 0) {
        fileMeta.textContent = "";
        return;
    }

    if (files.length === 1) {
        fileMeta.textContent = `${files[0].name} (${humanBytes(files[0].size)})`;
    } else {
        let totalSize = 0;
        for (const f of files) totalSize += f.size;
        fileMeta.textContent = `${files.length} 个文件 (总计 ${humanBytes(totalSize)})`;
    }
}

// Human readable bytes
function humanBytes(bytes) {
    const units = ["B", "KB", "MB", "GB"];
    let n = bytes;
    let i = 0;
    while (n >= 1024 && i < units.length - 1) {
        n /= 1024;
        i += 1;
    }
    return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

// Submit for classification
async function onSubmit() {
    const files = fileInput.files;

    status.className = "status";
    status.textContent = "";

    if (!files || files.length === 0) {
        status.className = "status error";
        status.textContent = "请选择文件。";
        return;
    }

    const maxPages = maxPagesInput.value || "50";
    const drugKeywords = drugKeywordsInput.value.trim();

    const fd = new FormData();
    for (const file of files) {
        fd.append("files", file, file.name);
    }

    btn.disabled = true;
    status.className = "status";
    status.textContent = "处理中...";

    try {
        let url = `/api/classify?max_pages=${encodeURIComponent(maxPages)}`;
        if (drugKeywords) {
            url += `&drug_keywords=${encodeURIComponent(drugKeywords)}`;
        }

        const res = await fetch(url, {
            method: "POST",
            body: fd,
        });

        const data = await res.json();

        if (!res.ok) {
            throw new Error(data && data.error ? data.error : `HTTP ${res.status}`);
        }

        renderResults(data.results || []);
        status.className = "status ok";
        status.textContent = `分类完成，共 ${(data.results || []).length} 个文件`;

    } catch (e) {
        status.className = "status error";
        status.textContent = `处理失败：${e.message || e}`;
    } finally {
        btn.disabled = false;
    }
}

// Render results
function renderResults(results) {
    if (!results || results.length === 0) {
        resultsDiv.innerHTML = `<div class="meta">暂无结果</div>`;
        resultsSummary.style.display = 'none';
        return;
    }

    // Summary
    const summary = { total: results.length, needsReview: 0 };
    results.forEach(r => {
        const label = r.label || 'Unknown';
        summary[label] = (summary[label] || 0) + 1;
        if (r.needs_review === 'True' || r.needs_review === true) {
            summary.needsReview++;
        }
    });

    resultsSummary.style.display = 'flex';
    resultsSummary.innerHTML = `
        <span>总计: <strong>${summary.total}</strong></span>
        <span class="pill ok">ICSR: ${summary.ICSR || 0}</span>
        <span class="pill info">Multiple: ${summary.Multiple_Patients || 0}</span>
        <span class="pill bad">Rejection: ${summary.Rejection || 0}</span>
        <span class="pill warn">Signal: ${summary.Other_Safety_Signal || 0}</span>
        ${summary.needsReview > 0 ? `<span class="pill bad">需复核: ${summary.needsReview}</span>` : ''}
    `;

    // Table
    const rows = results.map(r => {
        const label = r.label || "LLM_ERROR";
        const labelInfo = LABEL_MAP[label] || { text: label, desc: "", class: "bad" };

        // Confidence
        const confidence = parseFloat(r.confidence) || 0;
        const confidenceStr = `${(confidence * 100).toFixed(0)}%`;
        let confidenceClass = "";
        let confidenceLevel = "";
        if (confidence >= 0.90) {
            confidenceClass = "ok";
            confidenceLevel = "高";
        } else if (confidence >= 0.75) {
            confidenceClass = "info";
            confidenceLevel = "中高";
        } else if (confidence >= 0.60) {
            confidenceClass = "warn";
            confidenceLevel = "中";
        } else {
            confidenceClass = "bad";
            confidenceLevel = "低";
        }

        // Signal flags
        const signalItems = [
            { key: "drug", label: "药物", value: r.has_drug === 'True' || r.has_drug === true },
            { key: "ae", label: "不良事件", value: r.has_ae === 'True' || r.has_ae === true },
            { key: "causality", label: "因果关系", value: r.has_causality === 'True' || r.has_causality === true },
            { key: "special", label: "特殊情况", value: r.has_special_situation === 'True' || r.has_special_situation === true },
        ];

        const signalFlags = `
            <div class="signal-flags">
                ${signalItems.map(s =>
                    `<span class="signal-flag ${s.value ? 'yes' : 'no'}">
                        <span class="icon">${s.value ? '✓' : '✗'}</span>${s.label}
                    </span>`
                ).join('')}
            </div>`;

        // Patient info
        const patientMode = PATIENT_MODE_MAP[r.patient_mode] || r.patient_mode || "-";
        const patientCount = r.patient_max_n ? `(${r.patient_max_n}例)` : "";
        const patientDisplay = patientMode !== "-" ? `${patientMode}${patientCount}` : "-";

        // Extract method
        const extractMethod = METHOD_MAP[r.extract_method] || r.extract_method || "-";

        // Needs review
        const needsReview = r.needs_review === 'True' || r.needs_review === true;

        // Evidence parsing
        const parseEvidence = (str) => {
            if (!str) return [];
            if (typeof str === 'string') return str.split(';').map(s => s.trim()).filter(s => s);
            if (Array.isArray(str)) return str;
            return [];
        };

        const drugEvidence = parseEvidence(r.drug_evidence);
        const aeEvidence = parseEvidence(r.ae_evidence);
        const causalityEvidence = parseEvidence(r.causality_evidence);
        const specialEvidence = parseEvidence(r.special_evidence);
        const patientEvidence = parseEvidence(r.patient_evidence);

        // Build extraction fields
        const extractionFields = [
            { label: "目标药物", value: r.has_drug === 'True' || r.has_drug === true, evidence: drugEvidence },
            { label: "不良事件", value: r.has_ae === 'True' || r.has_ae === true, evidence: aeEvidence },
            { label: "因果关系", value: r.has_causality === 'True' || r.has_causality === true, evidence: causalityEvidence },
            { label: "特殊情况", value: r.has_special_situation === 'True' || r.has_special_situation === true, evidence: specialEvidence },
            { label: "患者模式", value: patientDisplay, evidence: patientEvidence, isText: true },
        ];

        // Rule logic
        const ruleLogic = label === 'Rejection'
            ? `<code>无drug 或 (无AE 且 无特殊情况)</code> → <span class="pill bad">Rejection</span>`
            : label === 'ICSR'
            ? `<code>drug + (特殊情况 或 (AE+因果关系)) + 单患者</code> → <span class="pill ok">ICSR</span>`
            : label === 'Multiple_Patients'
            ? `<code>drug + (特殊情况 或 (AE+因果关系)) + 多患者</code> → <span class="pill info">Multiple</span>`
            : label === 'ICSR+Multiple_Patients'
            ? `<code>drug + (特殊情况 或 (AE+因果关系)) + 混合患者</code> → <span class="pill warn">Mixed</span>`
            : `<code>drug + (AE或特殊情况) + 缺少因果/患者信息</code> → <span class="pill warn">Signal</span>`;

        return `
            <tr>
                <td class="mono" title="${escapeHtml(r.filename || r.file_name || '-')}">${escapeHtml(truncate(r.filename || r.file_name || '-', 35))}</td>
                <td>
                    <span class="pill ${labelInfo.class}">${labelInfo.text}</span>
                    <div style="font-size:11px;color:#666;margin-top:2px;">${labelInfo.desc}</div>
                    ${needsReview ? '<div style="margin-top:4px;"><span class="pill bad">需人工复核</span></div>' : ''}
                </td>
                <td>
                    <span class="pill ${confidenceClass}">${confidenceStr}</span>
                    <div style="font-size:11px;color:#666;margin-top:2px;">${confidenceLevel}置信度</div>
                </td>
                <td>${signalFlags}</td>
                <td class="mono">${patientDisplay}</td>
                <td class="mono">${extractMethod}</td>
                <td>
                    <details>
                        <summary>查看详情</summary>
                        <div class="details-content">
                            <div class="detail-section">
                                <div class="detail-title">🔍 LLM 抽取结果</div>
                                <div class="extraction-grid">
                                    ${extractionFields.map(f => `
                                        <div class="extraction-field">
                                            <div class="extraction-header">
                                                <span class="extraction-label">${f.label}</span>
                                                ${f.isText
                                                    ? `<span class="extraction-value text">${f.value}</span>`
                                                    : `<span class="extraction-value ${f.value ? 'yes' : 'no'}">${f.value ? '是 ✓' : '否 ✗'}</span>`
                                                }
                                            </div>
                                            ${f.evidence && f.evidence.length > 0
                                                ? `<div class="extraction-evidence">${f.evidence.slice(0, 2).map(e => `<span class="evidence-snippet">"${escapeHtml(truncate(e, 80))}"</span>`).join('')}</div>`
                                                : ''
                                            }
                                        </div>
                                    `).join('')}
                                </div>
                            </div>

                            <div class="detail-section">
                                <div class="detail-title">📐 规则分类逻辑</div>
                                <div class="rule-logic">${ruleLogic}</div>
                            </div>

                            ${r.reasoning ? `
                            <div class="detail-section">
                                <div class="detail-title">📋 LLM 推理过程</div>
                                <div class="reasoning-container">
                                    <div class="reasoning-text">${escapeHtml(r.reasoning)}</div>
                                </div>
                            </div>
                            ` : ''}

                            ${r.error ? `
                            <div class="detail-section">
                                <div class="detail-title">❌ 错误信息</div>
                                <div class="mono" style="color:#E4002B;">${escapeHtml(r.error)}</div>
                            </div>
                            ` : ''}
                        </div>
                    </details>
                </td>
            </tr>`;
    }).join("");

    resultsDiv.innerHTML = `
        <table class="table">
            <thead>
                <tr>
                    <th>文件名</th>
                    <th>分类标签</th>
                    <th>置信度</th>
                    <th>信号要素</th>
                    <th>患者信息</th>
                    <th>提取方式</th>
                    <th>详情</th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>`;
}

// Escape HTML
function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

// Truncate string
function truncate(str, len) {
    if (!str) return '';
    return str.length > len ? str.slice(0, len) + '...' : str;
}
