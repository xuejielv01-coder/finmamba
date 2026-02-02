
// Poll status
function updateStatus() {
    fetch('/api/status')
        .then(res => res.json())
        .then(data => {
            const dot = document.getElementById('status-dot');
            const text = document.getElementById('status-text');
            text.textContent = data.message;

            if (data.status === 'running') {
                dot.className = 'status-dot running';
                disableButtons(true);
            } else if (data.status === 'error') {
                dot.className = 'status-dot error';
                disableButtons(false);
            } else {
                dot.className = 'status-dot active';
                disableButtons(false);
            }
        })
        .catch(err => console.error(err));
}

// Poll every 2 seconds
setInterval(updateStatus, 2000);

function disableButtons(disabled) {
    const ids = ['btn-download', 'btn-preprocess', 'btn-train'];
    ids.forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.disabled = disabled;
    });
}

function triggerAction(action) {
    fetch(`/api/action/${action}`, { method: 'POST' })
        .then(res => {
            if (!res.ok) return res.json().then(j => { throw new Error(j.message) });
            return res.json();
        })
        .then(data => {
            console.log(data);
            updateStatus(); // Update immediately
        })
        .catch(err => alert('Error triggering action: ' + err.message));
}

function handleEnter(e) {
    if (e.key === 'Enter') diagnoseStock();
}

function diagnoseStock() {
    const codeInput = document.getElementById('diagnose-input');
    const code = codeInput.value.trim();
    if (!code) return;

    const output = document.getElementById('diagnosis-output');
    output.classList.remove('hidden');
    output.innerHTML = '<div style="text-align:center; color: var(--text-secondary); padding: 1rem;">Analyzing market data...</div>';

    fetch('/api/diagnose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: code })
    })
        .then(res => res.json())
        .then(data => {
            if (data.message && !data.ts_code) {
                output.innerHTML = `<div class="negative">Error: ${data.message}</div>`;
                return;
            }

            const confidence = data.confidence ? (data.confidence * 100).toFixed(1) : '0.0';
            const directionClass = data.direction === 'UP' ? 'positive' : (data.direction === 'DOWN' ? 'negative' : '');

            output.innerHTML = `
            <div class="diagnosis-item">
                <span style="color: var(--text-secondary);">Code</span>
                <span class="data-value">${data.ts_code}</span>
            </div>
            <div class="diagnosis-item">
                <span style="color: var(--text-secondary);">Direction</span>
                <span class="data-value ${directionClass}" style="font-weight: bold;">${data.direction}</span>
            </div>
            <div class="diagnosis-item">
                <span style="color: var(--text-secondary);">Magnitude</span>
                <span class="data-value">${data.magnitude}</span>
            </div>
            <div class="diagnosis-item">
                <span style="color: var(--text-secondary);">Confidence</span>
                <span class="data-value">${confidence}%</span>
            </div>
            <div class="diagnosis-item" style="border:none; margin-top: 0.5rem; font-size: 0.75rem; color: var(--text-secondary);">
                <span>Latency</span>
                <span>${Math.round(data.latency_ms || 0)}ms</span>
            </div>
        `;
        })
        .catch(err => {
            output.innerHTML = `<div class="negative" style="padding:1rem;">Error: ${err.message}</div>`;
        });
}

function loadScan() {
    const tbody = document.querySelector('#scan-table tbody');
    tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-secondary); padding: 2rem;">Loading latest scan data...</td></tr>';

    fetch('/api/scan?top_k=10')
        .then(res => res.json())
        .then(data => {
            if (data.error || data.message) throw new Error(data.message || 'Unknown error');
            if (!Array.isArray(data)) throw new Error('Invalid response format');

            tbody.innerHTML = '';
            data.forEach(row => {
                const tr = document.createElement('tr');
                // Safely handle missing keys
                const score = row.score !== undefined ? parseFloat(row.score).toFixed(4) : '-';
                const mag = row.magnitude || '-';
                const dir = row.direction || '-';

                let dirClass = '';
                if (dir === 'UP') dirClass = 'positive';
                if (dir === 'DOWN') dirClass = 'negative';

                tr.innerHTML = `
                <td class="data-value" style="color: var(--text-primary);">${row.ts_code || row.code || '-'}</td>
                <td>${row.trade_date || '-'}</td>
                <td class="data-value">${score}</td>
                <td class="data-value">${mag}</td>
                <td class="data-value ${dirClass}">${dir}</td>
            `;
                tbody.appendChild(tr);
            });

            if (data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-secondary); padding: 2rem;">No scan results found.</td></tr>';
            }
        })
        .catch(err => {
            tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--danger-color); padding: 2rem;">Error: ${err.message}</td></tr>`;
        });
}

// Initial load
updateStatus();
loadScan(); // Try to load last scan result
