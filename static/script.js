console.log('🚀 Pundit Feynman Script Loading... [v3.2]');

window.onerror = function (msg, url, lineNo, columnNo, error) {
    alert(`JS Error: ${msg}\nLine: ${lineNo}\nCheck browser console!`);
    return false;
};

// ── DOM Refs ──
const getEl = (id) => document.getElementById(id);
const dropZone = getEl('drop-zone');
const fileInput = getEl('file-input');
const uploadSection = getEl('upload-section');
const extractStatus = getEl('extract-status');
const extractLabel = getEl('extract-label');
const streamStatus = getEl('stream-status');
const doneSection = getEl('done-section');
const errorSection = getEl('error-section');
const errorText = getEl('error-text');
const downloadBtn = getEl('download-btn');
const resetBtn = getEl('reset-btn');
const errorResetBtn = getEl('error-reset-btn');
const codeOutput = getEl('code-output');
const codeViewer = getEl('code-viewer');
const codeBadge = getEl('code-badge');
const arxivInput = getEl('arxiv-input');
const arxivBtn = getEl('arxiv-btn');
const visualizeBtn = getEl('visualize-btn');
const imageFloat = getEl('image-float');
const imagePill = getEl('image-pill');
const floatHeader = getEl('float-header');
const floatImage = getEl('float-image');
const floatSpinner = getEl('float-spinner');
const floatDownload = getEl('float-download');
const floatMinimize = getEl('float-minimize');
const floatClose = getEl('float-close');

console.log('🎨 UI elements mapped. Visualize button:', !!visualizeBtn);

// Test backend connectivity
fetch('/api/ping').then(r => r.json()).then(d => console.log('🏓 Backend connectivity:', d.status)).catch(e => console.error('❌ Backend UNREACHABLE:', e));

// ── Visual Illustration State ──
let currentJobId = null;
window._debugJobId = () => currentJobId; // Access via console: window._debugJobId()

// ── State Manager ──
function showSection(section) {
    [uploadSection, extractStatus, streamStatus, doneSection, errorSection]
        .forEach(el => el.classList.add('hidden'));
    if (section) section.classList.remove('hidden');
}

// ── Drag & Drop ──
if (dropZone) {
    dropZone.addEventListener('click', () => {
        if (fileInput) fileInput.click();
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) handleUpload(e.dataTransfer.files[0]);
    });
}

if (fileInput) {
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleUpload(e.target.files[0]);
    });
}

// ── arXiv URL Handler ──
if (arxivBtn) {
    arxivBtn.addEventListener('click', () => handleArxiv());
}
if (arxivInput) {
    arxivInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') handleArxiv();
    });
}

async function handleArxiv() {
    const url = arxivInput.value.trim();
    if (!url) return;
    if (!url.includes('arxiv.org')) {
        alert('Please enter a valid arXiv URL (e.g. https://arxiv.org/abs/2401.12345)');
        return;
    }

    showSection(extractStatus);
    extractLabel.textContent = 'Downloading & analyzing arXiv paper…';
    codeOutput.textContent = '// Downloading PDF from arXiv…';
    codeBadge.textContent = 'extracting';
    codeBadge.className = 'code-badge';

    try {
        const res = await fetch('/api/extract-arxiv', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'arXiv extraction failed' }));
            throw new Error(err.detail || 'arXiv extraction failed');
        }

        const data = await res.json();
        console.log('arXiv extraction complete:', data);
        startStream(data.job_id);

    } catch (err) {
        showError(err.message);
    }
}

// ── Upload & Extract (Step 1) ──
async function handleUpload(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        alert('Please upload a PDF file.');
        return;
    }

    // Show extraction spinner
    showSection(extractStatus);
    extractLabel.textContent = 'Uploading & analyzing paper…';
    codeOutput.textContent = '// Waiting for paper analysis to complete…';
    codeBadge.textContent = 'extracting';
    codeBadge.className = 'code-badge';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/extract', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Extraction failed' }));
            throw new Error(err.detail || 'Extraction failed');
        }

        const data = await res.json();
        console.log('Extraction complete:', data);

        // Hide visualize button from previous run if any
        visualizeBtn.classList.add('hidden');

        // Start streaming (Step 2)
        startStream(data.job_id);

    } catch (err) {
        showError(err.message);
    }
}

// ── Live Streaming (Step 2) ──
function startStream(jobId) {
    currentJobId = jobId; // Store immediately
    showSection(streamStatus);
    codeOutput.textContent = '';
    codeBadge.textContent = 'streaming';
    codeBadge.className = 'code-badge streaming';

    const source = new EventSource(`/api/generate_stream/${jobId}`);
    let hasError = false;

    source.onmessage = (event) => {
        try {
            const payload = JSON.parse(event.data);

            if (payload.done) {
                source.close();
                if (payload.success) {
                    onStreamComplete(jobId);
                } else {
                    // Pipeline finished but failed — show error state
                    showError('Pipeline failed to generate notebook. Check the code output panel for details.');
                    codeBadge.textContent = 'failed';
                    codeBadge.className = 'code-badge';
                }
                return;
            }

            if (payload.analysis_done) {
                // Show visualize button early!
                if (visualizeBtn) visualizeBtn.classList.remove('hidden');
                return;
            }

            if (payload.text) {
                // Check if it's an error message
                if (payload.text.includes('❌')) {
                    hasError = true;
                }
                codeOutput.textContent += payload.text;
                // Auto-scroll to bottom
                codeViewer.scrollTop = codeViewer.scrollHeight;
            }
        } catch (e) {
            console.error('Parse error:', e);
        }
    };

    source.onerror = (err) => {
        console.error('SSE error:', err);
        source.close();
        showError('Stream connection lost. Please try again.');
    };
}

function onStreamComplete(jobId) {
    showSection(doneSection);
    if (downloadBtn) {
        downloadBtn.href = `/api/download/${jobId}`;
        downloadBtn.download = "pundit_feynman_notebook.ipynb";
    }
    currentJobId = jobId; // Store for visualization
    if (codeBadge) {
        codeBadge.textContent = 'complete';
        codeBadge.className = 'code-badge done';
    }
}

// ── Visual Illustration Logic ──

if (visualizeBtn) {
    visualizeBtn.addEventListener('click', async (e) => {
        console.log('🖱️ Visualize button CLICKED. Event object:', e);

        if (!currentJobId) {
            console.error('❌ Cannot visualize: currentJobId is null');
            alert('Software Error: Job ID not captured yet. Please wait for analysis or refresh.');
            return;
        }

        console.log('🎨 Requesting visualization for Job:', currentJobId);

        // Disable button to prevent double-clicks
        visualizeBtn.disabled = true;
        const originalText = visualizeBtn.textContent;
        visualizeBtn.textContent = '🎨 Painting...';

        // Show float UI
        if (imageFloat) imageFloat.classList.remove('hidden');
        if (imagePill) imagePill.classList.add('hidden');
        if (floatImage) floatImage.classList.add('hidden');
        if (floatSpinner) floatSpinner.classList.remove('hidden');

        try {
            const url = `/api/visualize/${currentJobId}`;
            console.log('🌐 Fetching:', url);

            const res = await fetch(url, { method: 'POST' });
            console.log('📥 Response status:', res.status);

            if (!res.ok) {
                const errDetail = await res.json().catch(() => ({ detail: 'Network error' }));
                throw new Error(errDetail.detail || `Server error ${res.status}`);
            }

            const data = await res.json();
            console.log('🖼️ Image received! Length:', data.image.length);

            if (floatImage) {
                floatImage.src = data.image;
                floatImage.classList.remove('hidden');
            }
            if (floatSpinner) floatSpinner.classList.add('hidden');
        } catch (err) {
            console.error('❌ Visualization flow error:', err);
            alert(`Painting failed: ${err.message}`);
            if (imageFloat) imageFloat.classList.add('hidden');
        } finally {
            visualizeBtn.disabled = false;
            visualizeBtn.textContent = originalText;
            console.log('🏁 Visualize flow completed.');
        }
    });
}

// Drag Logic
let isDragging = false;
let startX, startY, initialX, initialY;

if (floatHeader && imageFloat) {
    floatHeader.addEventListener('mousedown', (e) => {
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        initialX = imageFloat.offsetLeft;
        initialY = imageFloat.offsetTop;
        imageFloat.style.transition = 'none';
    });
}

document.addEventListener('mousemove', (e) => {
    if (!isDragging || !imageFloat) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    imageFloat.style.left = (initialX + dx) + 'px';
    imageFloat.style.top = (initialY + dy) + 'px';
    imageFloat.style.bottom = 'auto'; // Remove fixed positioning
    imageFloat.style.right = 'auto';
});

document.addEventListener('mouseup', () => {
    isDragging = false;
    if (imageFloat) imageFloat.style.transition = '';
});

// Minimize/Close/Download
if (floatMinimize && imageFloat && imagePill) {
    floatMinimize.addEventListener('click', () => {
        imageFloat.classList.add('hidden');
        imagePill.classList.remove('hidden');
    });
}

if (imagePill && imageFloat) {
    imagePill.addEventListener('click', () => {
        imageFloat.classList.remove('hidden');
        imagePill.classList.add('hidden');
    });
}

if (floatClose && imageFloat && imagePill) {
    floatClose.addEventListener('click', () => {
        imageFloat.classList.add('hidden');
        imagePill.classList.add('hidden');
    });
}

if (floatDownload && floatImage) {
    floatDownload.addEventListener('click', () => {
        if (!floatImage.src) return;
        const link = document.createElement('a');
        link.href = floatImage.src;
        link.download = `pundit_feynman_illustration_${currentJobId}.png`;
        link.click();
    });
}

// ── Error & Reset ──
function showError(msg) {
    showSection(errorSection);
    if (errorText) errorText.textContent = msg;
    if (codeBadge) {
        codeBadge.textContent = 'error';
        codeBadge.className = 'code-badge';
    }
    // Cleanup float on error — with null checks!
    if (imageFloat) imageFloat.classList.add('hidden');
    if (imagePill) imagePill.classList.add('hidden');
}

function resetUI() {
    showSection(uploadSection);
    if (fileInput) fileInput.value = '';
    if (arxivInput) arxivInput.value = '';
    if (codeOutput) codeOutput.textContent = '// Upload a paper to see the generated code here…';
    if (codeBadge) {
        codeBadge.textContent = 'waiting';
        codeBadge.className = 'code-badge';
    }
    currentJobId = null;
    if (visualizeBtn) visualizeBtn.classList.add('hidden');
    // Cleanup float on reset
    if (imageFloat) imageFloat.classList.add('hidden');
    if (imagePill) imagePill.classList.add('hidden');
}

if (resetBtn) resetBtn.addEventListener('click', resetUI);
if (errorResetBtn) errorResetBtn.addEventListener('click', resetUI);
