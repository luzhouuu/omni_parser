/**
 * Wanfang Pipeline Frontend
 */

// State
let isRunning = false;
let pollInterval = null;

// DOM Elements
const queryInput = document.getElementById('query');
const startYearInput = document.getElementById('start-year');
const endYearInput = document.getElementById('end-year');
const maxArticlesInput = document.getElementById('max-articles');
const resourceTypeSelect = document.getElementById('resource-type');
const drugKeywordsInput = document.getElementById('drug-keywords');
const drugFileInput = document.getElementById('drug-file');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const statusCard = document.getElementById('status-card');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const resultsCard = document.getElementById('results-card');
const resultsSummary = document.getElementById('results-summary');
const resultsDiv = document.getElementById('results');

// Step elements
const stepSearch = document.getElementById('step-search');
const stepDownload = document.getElementById('step-download');
const stepClassify = document.getElementById('step-classify');
const infoSearch = document.getElementById('info-search');
const infoDownload = document.getElementById('info-download');
const infoClassify = document.getElementById('info-classify');

// Status colors
const STATUS_COLORS = {
    pending: '#999',
    searching: '#2196F3',
    downloading: '#FF9800',
    classifying: '#9C27B0',
    completed: '#4CAF50',
    error: '#F44336',
    idle: '#999',
};

// Status text mapping
const STATUS_TEXT = {
    pending: '等待中',
    searching: '搜索中...',
    downloading: '下载中...',
    classifying: '分类中...',
    completed: '已完成',
    error: '错误',
    idle: '空闲',
};

// Label display mapping
const LABEL_MAP = {
    "Rejection": { text: "Rejection", desc: "拒绝", class: "bad" },
    "ICSR": { text: "ICSR", desc: "个例安全报告", class: "ok" },
    "Multiple_Patients": { text: "Multiple_Patients", desc: "多患者报告", class: "info" },
    "ICSR+Multiple_Patients": { text: "ICSR+Multiple_Patients", desc: "混合报告", class: "warn" },
    "Other_Safety_Signal": { text: "Other_Safety_Signal", desc: "其他安全信号", class: "warn" },
    "Error": { text: "Error", desc: "处理错误", class: "bad" }
};

// Patient mode mapping
const PATIENT_MODE_MAP = {
    "single": "单个患者",
    "multiple": "多个患者",
    "mixed": "混合",
    "unknown": "未知"
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    startBtn.addEventListener('click', startPipeline);
    stopBtn.addEventListener('click', stopPipeline);
    drugFileInput.addEventListener('change', handleDrugFileUpload);

    // Load health and check status
    loadHealth();
    checkStatus();
});

// Load health info
async function loadHealth() {
    try {
        const res = await fetch("/api/health");
        const data = await res.json();
        const env = data.env || {};
        const paths = data.paths || {};
        const toolPill = (ok) => `<span class="pill ${ok ? "ok" : "bad"}">${ok ? "正常" : "缺失"}</span>`;

        document.getElementById("health").innerHTML =
            `OpenAI ${toolPill(!!env.openai_api_key)} <code>${env.llm_model || "-"}</code> | ` +
            `万方登录 ${toolPill(!!env.wanfang_username)} | ` +
            `已下载 PDF: <code>${paths.papers_count || 0}</code>`;
    } catch (e) {
        document.getElementById("health").textContent = "健康检查失败";
    }
}

// Handle drug keywords file upload
function handleDrugFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        const content = e.target.result;
        const keywords = content.split('\n')
            .map(line => line.trim())
            .filter(line => line && !line.startsWith('#'))
            .join(', ');
        drugKeywordsInput.value = keywords;
    };
    reader.readAsText(file);
}

// Start pipeline
async function startPipeline() {
    if (isRunning) return;

    const query = queryInput.value.trim();
    if (!query) {
        alert('请输入搜索表达式');
        return;
    }

    const drugKeywords = drugKeywordsInput.value
        .split(',')
        .map(k => k.trim())
        .filter(k => k);

    const payload = {
        query: query,
        start_year: parseInt(startYearInput.value) || 2020,
        end_year: parseInt(endYearInput.value) || 2025,
        max_articles: parseInt(maxArticlesInput.value) || 0,
        resource_type: resourceTypeSelect.value || 'chinese',
        drug_keywords: drugKeywords,
    };

    try {
        const response = await fetch('/api/pipeline', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        const data = await response.json();

        if (!response.ok) {
            alert('错误: ' + (data.error || 'Pipeline 启动失败'));
            return;
        }

        isRunning = true;
        updateUI(true);
        startPolling();

    } catch (error) {
        alert('错误: ' + error.message);
    }
}

// Stop pipeline
async function stopPipeline() {
    try {
        await fetch('/api/stop', { method: 'POST' });
        isRunning = false;
        updateUI(false);
        stopPolling();
    } catch (error) {
        console.error('Stop error:', error);
    }
}

// Check current status
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        if (data.status && data.status !== 'idle') {
            isRunning = !['completed', 'error'].includes(data.status);
            updateUI(isRunning);
            updateStatus(data);

            if (isRunning) {
                startPolling();
            } else {
                loadResults();
            }
        }
    } catch (error) {
        console.error('Status check error:', error);
    }
}

// Start polling for status updates
function startPolling() {
    if (pollInterval) return;

    pollInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            updateStatus(data);

            if (data.status === 'completed' || data.status === 'error') {
                isRunning = false;
                updateUI(false);
                stopPolling();
                loadResults();
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 2000);
}

// Stop polling
function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// Update UI state
function updateUI(running) {
    startBtn.style.display = running ? 'none' : 'inline-block';
    stopBtn.style.display = running ? 'inline-block' : 'none';
    statusCard.style.display = 'block';

    // Disable inputs when running
    queryInput.disabled = running;
    startYearInput.disabled = running;
    endYearInput.disabled = running;
    maxArticlesInput.disabled = running;
    resourceTypeSelect.disabled = running;
    drugKeywordsInput.disabled = running;
    drugFileInput.disabled = running;
}

// Update status display
function updateStatus(data) {
    const status = data.status || 'idle';

    // Update progress bar
    const progress = (data.progress || 0) * 100;
    progressFill.style.width = progress + '%';
    progressText.textContent = Math.round(progress) + '%';

    // Reset all steps
    resetSteps();

    // Update steps based on status
    if (status === 'pending') {
        infoSearch.textContent = '准备中...';
    } else if (status === 'searching') {
        setStepActive(stepSearch);
        infoSearch.textContent = '搜索中...';
        if (data.search_total > 0) {
            if (data.search_count === 0) {
                infoSearch.textContent = `共 ${data.search_total} 篇（已下载）`;
            } else if (data.search_total > data.search_count) {
                infoSearch.textContent = `${data.search_count} 篇新 / ${data.search_total} 篇`;
            } else {
                infoSearch.textContent = `找到 ${data.search_count} 篇`;
            }
        } else if (data.search_count > 0) {
            infoSearch.textContent = `找到 ${data.search_count} 篇`;
        }
    } else if (status === 'downloading') {
        setStepCompleted(stepSearch);
        // Format search info
        if (data.search_total > 0 && data.search_count === 0) {
            infoSearch.textContent = `共 ${data.search_total} 篇（已下载）`;
        } else if (data.search_total > data.search_count) {
            infoSearch.textContent = `${data.search_count} 篇新 / ${data.search_total} 篇`;
        } else {
            infoSearch.textContent = `找到 ${data.search_count || 0} 篇`;
        }

        setStepActive(stepDownload);
        if (data.download_count > 0) {
            infoDownload.textContent = `已下载 ${data.download_count} 篇`;
        } else {
            infoDownload.textContent = '下载中...';
        }
    } else if (status === 'classifying') {
        setStepCompleted(stepSearch);
        infoSearch.textContent = formatSearchInfo(data);

        setStepCompleted(stepDownload);
        infoDownload.textContent = `已下载 ${data.download_count || 0} 篇`;

        setStepActive(stepClassify);
        if (data.classify_count > 0) {
            infoClassify.textContent = `已分类 ${data.classify_count} 篇`;
        } else {
            infoClassify.textContent = '分类中...';
        }
    } else if (status === 'completed') {
        setStepCompleted(stepSearch);
        infoSearch.textContent = formatSearchInfo(data);

        setStepCompleted(stepDownload);
        infoDownload.textContent = `已下载 ${data.download_count || 0} 篇`;

        setStepCompleted(stepClassify);
        infoClassify.textContent = `已分类 ${data.classify_count || 0} 篇`;
    } else if (status === 'error') {
        // Mark the current step as error
        if (data.search_total === 0 && data.search_count === 0) {
            setStepError(stepSearch);
            infoSearch.textContent = data.error_message || '搜索失败';
        } else if (data.download_count === 0) {
            setStepCompleted(stepSearch);
            infoSearch.textContent = formatSearchInfo(data);
            setStepError(stepDownload);
            infoDownload.textContent = data.error_message || '下载失败';
        } else {
            setStepCompleted(stepSearch);
            infoSearch.textContent = formatSearchInfo(data);
            setStepCompleted(stepDownload);
            infoDownload.textContent = `已下载 ${data.download_count} 篇`;
            setStepError(stepClassify);
            infoClassify.textContent = data.error_message || '分类失败';
        }
    }
}

// Format search info based on total and new counts
function formatSearchInfo(data) {
    const total = data.search_total || 0;
    const newCount = data.search_count || 0;

    if (total > 0 && newCount === 0) {
        return `共 ${total} 篇（已下载）`;
    } else if (total > newCount && newCount > 0) {
        return `${newCount} 篇新 / ${total} 篇`;
    } else if (newCount > 0) {
        return `找到 ${newCount} 篇`;
    } else {
        return '无结果';
    }
}

// Reset all steps to default state
function resetSteps() {
    [stepSearch, stepDownload, stepClassify].forEach((step, index) => {
        step.classList.remove('active', 'completed', 'error');
        const icon = step.querySelector('.step-icon');
        if (icon) icon.textContent = String(index + 1);
    });
    infoSearch.textContent = '等待开始';
    infoDownload.textContent = '等待开始';
    infoClassify.textContent = '等待开始';
}

// Set step as active
function setStepActive(step) {
    step.classList.add('active');
}

// Set step as completed
function setStepCompleted(step) {
    step.classList.remove('active');
    step.classList.add('completed');
    const icon = step.querySelector('.step-icon');
    if (icon) icon.textContent = '✓';
}

// Set step as error
function setStepError(step) {
    step.classList.remove('active');
    step.classList.add('error');
    const icon = step.querySelector('.step-icon');
    if (icon) icon.textContent = '✗';
}

// Load classification results
async function loadResults() {
    try {
        const response = await fetch('/api/results');
        const data = await response.json();

        if (!data.results || data.results.length === 0) {
            resultsCard.style.display = 'none';
            return;
        }

        resultsCard.style.display = 'block';

        // Summary
        const summary = summarizeResults(data.results);
        resultsSummary.innerHTML = `
            <span>总计: <strong>${data.count}</strong></span>
            <span class="pill ok">ICSR: ${summary.ICSR || 0}</span>
            <span class="pill info">Multiple: ${summary.Multiple_Patients || 0}</span>
            <span class="pill bad">Rejection: ${summary.Rejection || 0}</span>
            <span class="pill warn">Signal: ${summary.Other_Safety_Signal || 0}</span>
            ${summary.needsReview > 0 ? `<span class="pill bad">需复核: ${summary.needsReview}</span>` : ''}
        `;

        // Table
        resultsDiv.innerHTML = renderResults(data.results);

    } catch (error) {
        console.error('Load results error:', error);
    }
}

// Summarize results
function summarizeResults(results) {
    const summary = { needsReview: 0 };
    results.forEach(r => {
        const label = r.label || 'Unknown';
        summary[label] = (summary[label] || 0) + 1;
        if (r.needs_review === 'True') {
            summary.needsReview++;
        }
    });
    return summary;
}

// Render results table (matching novartis style)
function renderResults(results) {
    if (!results || results.length === 0) {
        return `<div class="meta">暂无结果</div>`;
    }

    const rows = results.map(r => {
        const label = r.label || "Error";
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
            { key: "drug", label: "药物", value: r.has_drug === 'True' },
            { key: "ae", label: "不良事件", value: r.has_ae === 'True' },
            { key: "causality", label: "因果关系", value: r.has_causality === 'True' },
            { key: "special", label: "特殊情况", value: r.has_special_situation === 'True' },
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
        const methodMap = {
            "pdftotext": "PDF文本",
            "pymupdf": "PyMuPDF",
            "pdf_ocr": "PDF OCR",
            "tesseract": "图片OCR",
            "txt": "文本文件",
            "none": "无法提取"
        };
        const extractMethod = methodMap[r.extract_method] || r.extract_method || "-";

        // Needs review
        const needsReview = r.needs_review === 'True';

        // Evidence (parse from string if needed)
        const parseEvidence = (str) => {
            if (!str) return [];
            if (typeof str === 'string') return str.split(';').map(s => s.trim()).filter(s => s);
            return str;
        };

        const drugEvidence = parseEvidence(r.drug_evidence);
        const aeEvidence = parseEvidence(r.ae_evidence);
        const causalityEvidence = parseEvidence(r.causality_evidence);
        const specialEvidence = parseEvidence(r.special_evidence);
        const patientEvidence = parseEvidence(r.patient_evidence);

        // Build extraction fields for details
        const extractionFields = [
            { label: "目标药物", value: r.has_drug === 'True', evidence: drugEvidence, reasoning: r.has_drug_reasoning },
            { label: "不良事件", value: r.has_ae === 'True', evidence: aeEvidence, reasoning: r.has_ae_reasoning },
            { label: "因果关系", value: r.has_causality === 'True', evidence: causalityEvidence, reasoning: r.has_causality_reasoning },
            { label: "特殊情况", value: r.has_special_situation === 'True', evidence: specialEvidence, reasoning: r.has_special_reasoning },
            { label: "患者模式", value: patientDisplay, evidence: patientEvidence, reasoning: r.patient_reasoning, isText: true },
        ];

        // Rule logic explanation
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
                <td class="mono" title="${escapeHtml(r.filename)}">${escapeHtml(truncate(r.filename, 35))}</td>
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
                                            ${f.reasoning
                                                ? `<div class="extraction-reasoning">${escapeHtml(f.reasoning)}</div>`
                                                : ''
                                            }
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

    return `
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

// Utility: escape HTML
function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

// Utility: truncate string
function truncate(str, len) {
    if (!str) return '';
    return str.length > len ? str.slice(0, len) + '...' : str;
}

// Download results CSV
function downloadResults() {
    window.location.href = '/api/download';
}
