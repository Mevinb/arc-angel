/**
 * Local Grok Chatbot UI Controller
 */
document.addEventListener('DOMContentLoaded', () => {
    let currentSessionId = null;
    let currentMode = 'classic';
    let isStreaming = false;
    let currentAbortController = null;

    // DOM Elements
    const sessionListEl = document.getElementById('session-list');
    const newChatBtn = document.getElementById('new-chat-btn');
    const messagesContainer = document.getElementById('messages-container');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const modePills = document.querySelectorAll('.mode-pill');
    const telemetryBar = document.getElementById('budget-info');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const toggleEngineBtn = document.getElementById('toggle-engine-btn');
    const openMemoryBtn = document.getElementById('open-memory-btn');
    const closeMemoryBtn = document.getElementById('close-memory-btn');
    const memoryDrawer = document.getElementById('memory-drawer');
    const factCounter = document.getElementById('fact-counter');
    const dossierList = document.getElementById('dossier-list');
    const addFactForm = document.getElementById('add-fact-form');
    const drawerTabs = document.querySelectorAll('.drawer-tab');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const episodicList = document.getElementById('episodic-list');
    const episodicSearchInput = document.getElementById('episodic-search-input');
    const episodicSearchBtn = document.getElementById('episodic-search-btn');

    // Model Selector Elements
    const modelSelect = document.getElementById('model-select');
    const modelBadge = document.getElementById('model-badge');
    const inputHint = document.querySelector('.input-hint');
    const gpuModeSelect = document.getElementById('gpu-mode-select');
    let availableModels = [];
    let currentModelId = null;
    let activeProvider = 'local';

    // ChatGPT API Key Modal Elements
    const openaiKeyBtn = document.getElementById('openai-key-btn');
    const apiKeyModal = document.getElementById('api-key-modal');
    const closeKeyModalBtn = document.getElementById('close-key-modal-btn');
    const openaiKeyInput = document.getElementById('openai-key-input');
    const openaiBaseUrlInput = document.getElementById('openai-base-url-input');
    const openaiKeyStatus = document.getElementById('openai-key-status');
    const openaiKeyError = document.getElementById('openai-key-error');
    const saveOpenaiKeyBtn = document.getElementById('save-openai-key-btn');
    const deleteOpenaiKeyBtn = document.getElementById('delete-openai-key-btn');

    const desktopLiveViewer = document.getElementById('desktop-live-viewer');
    const dlvStepBadge = document.getElementById('dlv-step-badge');
    const dlvExpandBtn = document.getElementById('dlv-expand-btn');
    const dlvCloseBtn = document.getElementById('dlv-close-btn');
    const dlvPipBtn = document.getElementById('dlv-pip-btn');
    const dlvStopBtn = document.getElementById('dlv-stop-btn');
    const dlvScreenImg = document.getElementById('dlv-screen-img');
    const dlvPlaceholder = document.getElementById('dlv-placeholder');
    const dlvScreenClick = document.getElementById('dlv-screen-click');
    const dlvThought = document.getElementById('dlv-thought');
    const dlvAction = document.getElementById('dlv-action');

    const dlvModeBtn = document.getElementById('dlv-mode-btn');
    const dlvModeText = document.getElementById('dlv-mode-text');

    function handleViewerImageClick(e, imgElem, containerElem) {
        if (!imgElem || !imgElem.naturalWidth || !imgElem.naturalHeight) return;
        const rect = imgElem.getBoundingClientRect();
        const clientX = e.clientX - rect.left;
        const clientY = e.clientY - rect.top;

        // Account for object-fit: contain letterboxing
        const elemRatio = rect.width / rect.height;
        const natRatio = imgElem.naturalWidth / imgElem.naturalHeight;

        let renderedWidth = rect.width;
        let renderedHeight = rect.height;
        let offsetX = 0;
        let offsetY = 0;

        if (elemRatio > natRatio) {
            renderedWidth = rect.height * natRatio;
            offsetX = (rect.width - renderedWidth) / 2;
        } else {
            renderedHeight = rect.width / natRatio;
            offsetY = (rect.height - renderedHeight) / 2;
        }

        const relX = clientX - offsetX;
        const relY = clientY - offsetY;

        if (relX < 0 || relX > renderedWidth || relY < 0 || relY > renderedHeight) {
            return;
        }

        const actualX = Math.round((relX / renderedWidth) * imgElem.naturalWidth);
        const actualY = Math.round((relY / renderedHeight) * imgElem.naturalHeight);

        // Visual click ripple
        if (containerElem) {
            const doc = containerElem.ownerDocument || document;
            const ripple = doc.createElement('div');
            ripple.className = 'dlv-click-indicator';
            const contRect = containerElem.getBoundingClientRect();
            ripple.style.left = `${e.clientX - contRect.left}px`;
            ripple.style.top = `${e.clientY - contRect.top}px`;
            containerElem.appendChild(ripple);
            setTimeout(() => ripple.remove(), 600);
        }

        API.sendManualAction({ type: 'click', x: actualX, y: actualY }).then(res => {
            if (res && res.screenshot_url) {
                if (dlvScreenImg) dlvScreenImg.src = res.screenshot_url;
                updatePipWindow({ screenshot_url: res.screenshot_url });
            }
        }).catch(err => console.warn('Manual action failed:', err));
    }

    // Floating Picture-in-Picture (PiP) Always-on-Top HUD Controller
    let pipWindow = null;
    let isDesktopRunning = false;

    async function openPipWindow() {
        if (pipWindow && !pipWindow.closed) {
            try { pipWindow.focus(); } catch (e) {}
            return;
        }

        try {
            if ('documentPictureInPicture' in window) {
                pipWindow = await window.documentPictureInPicture.requestWindow({
                    width: 440,
                    height: 520,
                });

                [...document.styleSheets].forEach(styleSheet => {
                    try {
                        const cssRules = [...styleSheet.cssRules].map(r => r.cssText).join('\n');
                        const style = pipWindow.document.createElement('style');
                        style.textContent = cssRules;
                        pipWindow.document.head.appendChild(style);
                    } catch (e) {
                        const link = pipWindow.document.createElement('link');
                        link.rel = 'stylesheet';
                        link.href = styleSheet.href;
                        pipWindow.document.head.appendChild(link);
                    }
                });

                const pipStyle = pipWindow.document.createElement('style');
                pipStyle.textContent = `
                    body {
                        margin: 0;
                        padding: 12px;
                        background: #09090b;
                        color: #f4f4f5;
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                        display: flex;
                        flex-direction: column;
                        height: 100vh;
                        box-sizing: border-box;
                        overflow: hidden;
                    }
                    .pip-hud { display: flex; flex-direction: column; height: 100%; gap: 10px; }
                    .pip-header { display: flex; justify-content: space-between; align-items: center; }
                    .pip-title { font-size: 13px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
                    .pip-screen-container { flex: 1; background: #000; border: 1px solid #27272a; border-radius: 8px; overflow: hidden; display: flex; align-items: center; justify-content: center; min-height: 180px; }
                    .pip-screen-container img { width: 100%; height: 100%; object-fit: contain; }
                    .pip-status-bar { background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 10px; display: flex; flex-direction: column; gap: 6px; font-size: 12px; }
                    .pip-row { display: flex; gap: 6px; }
                    .pip-label { color: #a1a1aa; font-weight: 600; min-width: 50px; }
                    .pip-thought { color: #e4e4e7; max-height: 48px; overflow-y: auto; }
                    .pip-action { color: #38bdf8; font-family: monospace; font-weight: 600; word-break: break-all; }
                    .pip-stop-btn { background: #ef4444; color: white; border: none; border-radius: 6px; padding: 6px 14px; font-weight: 700; font-size: 12px; cursor: pointer; display: flex; align-items: center; gap: 6px; }
                    .pip-stop-btn:hover { background: #dc2626; }
                `;
                pipWindow.document.head.appendChild(pipStyle);

                pipWindow.document.body.innerHTML = `
                    <div class="pip-hud">
                        <div class="pip-header">
                            <div class="pip-title">
                                <span class="dlv-pulse"></span>
                                <span>GROK Desktop Pilot</span>
                                <span id="pip-step-badge" class="dlv-step-badge">${dlvStepBadge ? dlvStepBadge.textContent : 'Step 0'}</span>
                            </div>
                            <button id="pip-stop-btn" class="pip-stop-btn" title="Emergency Stop Pilot (Esc)">
                                ⏹ Stop Pilot
                            </button>
                        </div>
                        <div class="pip-screen-container">
                            <img id="pip-screen-img" src="${(dlvScreenImg && dlvScreenImg.src) || ''}" />
                        </div>
                        <div class="pip-status-bar">
                            <div class="pip-row">
                                <span class="pip-label">Thought:</span>
                                <span id="pip-thought" class="pip-thought">${(dlvThought && dlvThought.textContent) || ''}</span>
                            </div>
                            <div class="pip-row">
                                <span class="pip-label">Action:</span>
                                <span id="pip-action" class="pip-action">${(dlvAction && dlvAction.textContent) || ''}</span>
                            </div>
                        </div>
                    </div>
                `;

                const pipImg = pipWindow.document.getElementById('pip-screen-img');
                if (pipImg) {
                    pipImg.style.cursor = 'crosshair';
                    pipImg.addEventListener('click', (e) => {
                        handleViewerImageClick(e, pipImg, pipImg.parentElement);
                    });
                }

                pipWindow.document.getElementById('pip-stop-btn').addEventListener('click', async () => {
                    await API.stopDesktop();
                    if (stopDesktopBtn) stopDesktopBtn.classList.add('hidden');
                    closePipWindow();
                });

                pipWindow.addEventListener('pagehide', () => {
                    pipWindow = null;
                });
            } else {
                const w = 440, h = 520;
                const left = window.screen.width - w - 20;
                const top = 40;
                pipWindow = window.open(
                    '',
                    'GrokDesktopHUD',
                    `width=${w},height=${h},top=${top},left=${left},menubar=no,toolbar=no,location=no,status=no`
                );
                if (pipWindow) {
                    pipWindow.document.write(`
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <title>GROK Desktop Pilot HUD</title>
                            <style>
                                body { margin: 0; padding: 12px; background: #09090b; color: #f4f4f5; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; flex-direction: column; height: 95vh; }
                                .pip-hud { display: flex; flex-direction: column; height: 100%; gap: 10px; }
                                .pip-header { display: flex; justify-content: space-between; align-items: center; }
                                .pip-title { font-size: 13px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
                                .pip-screen-container { flex: 1; background: #000; border: 1px solid #27272a; border-radius: 8px; overflow: hidden; display: flex; align-items: center; justify-content: center; }
                                .pip-screen-container img { width: 100%; height: 100%; object-fit: contain; }
                                .pip-status-bar { background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 10px; font-size: 12px; }
                                .pip-stop-btn { background: #ef4444; color: white; border: none; border-radius: 6px; padding: 6px 14px; font-weight: 700; cursor: pointer; }
                                .pip-stop-btn:hover { background: #dc2626; }
                            </style>
                        </head>
                        <body>
                            <div class="pip-hud">
                                <div class="pip-header">
                                    <div class="pip-title">GROK Desktop Pilot <span id="pip-step-badge">[${(dlvStepBadge && dlvStepBadge.textContent) || 'Step 0'}]</span></div>
                                    <button id="pip-stop-btn" class="pip-stop-btn">⏹ Stop</button>
                                </div>
                                <div class="pip-screen-container">
                                    <img id="pip-screen-img" src="${(dlvScreenImg && dlvScreenImg.src) || ''}" />
                                </div>
                                <div class="pip-status-bar">
                                    <div><strong>Thought:</strong> <span id="pip-thought">${(dlvThought && dlvThought.textContent) || ''}</span></div>
                                    <div style="margin-top:4px;"><strong>Action:</strong> <span id="pip-action" style="color:#38bdf8;font-family:monospace;">${(dlvAction && dlvAction.textContent) || ''}</span></div>
                                </div>
                            </div>
                        </body>
                        </html>
                    `);
                    const popImg = pipWindow.document.getElementById('pip-screen-img');
                    if (popImg) {
                        popImg.style.cursor = 'crosshair';
                        popImg.addEventListener('click', (e) => {
                            handleViewerImageClick(e, popImg, popImg.parentElement);
                        });
                    }

                    pipWindow.document.getElementById('pip-stop-btn').addEventListener('click', async () => {
                        await API.stopDesktop();
                        if (stopDesktopBtn) stopDesktopBtn.classList.add('hidden');
                        closePipWindow();
                    });
                    pipWindow.addEventListener('beforeunload', () => { pipWindow = null; });
                }
            }
        } catch (err) {
            console.warn('PiP window open failed:', err);
        }
    }

    function closePipWindow() {
        if (pipWindow) {
            try { pipWindow.close(); } catch (e) {}
            pipWindow = null;
        }
    }

    function updatePipWindow(desktopStep) {
        if (!pipWindow || pipWindow.closed || !pipWindow.document) return;
        try {
            const stepBadge = pipWindow.document.getElementById('pip-step-badge');
            const thoughtEl = pipWindow.document.getElementById('pip-thought');
            const actionEl = pipWindow.document.getElementById('pip-action');
            const imgEl = pipWindow.document.getElementById('pip-screen-img');

            if (stepBadge && desktopStep.step !== undefined) {
                stepBadge.textContent = `Step ${desktopStep.step}`;
            }
            if (thoughtEl && desktopStep.thought) {
                thoughtEl.textContent = desktopStep.thought;
            }
            if (actionEl && desktopStep.action) {
                actionEl.textContent = desktopStep.action;
            }
            if (imgEl && desktopStep.screenshot_url) {
                imgEl.src = desktopStep.screenshot_url;
            }
        } catch (e) {}
    }

    if (dlvPipBtn) {
        dlvPipBtn.addEventListener('click', () => {
            if (pipWindow && !pipWindow.closed) {
                closePipWindow();
            } else {
                openPipWindow();
            }
        });
    }

    if (dlvStopBtn) {
        dlvStopBtn.addEventListener('click', async () => {
            await API.stopDesktop();
            if (stopDesktopBtn) stopDesktopBtn.classList.add('hidden');
            closePipWindow();
        });
    }

    // Global Emergency Stop Listener (Escape key)
    window.addEventListener('keydown', async (e) => {
        if (e.key === 'Escape' && isDesktopRunning) {
            await API.stopDesktop();
            if (stopDesktopBtn) stopDesktopBtn.classList.add('hidden');
            closePipWindow();
        }
    });

    if (dlvExpandBtn) {
        dlvExpandBtn.addEventListener('click', () => {
            if (desktopLiveViewer) desktopLiveViewer.classList.toggle('expanded');
        });
    }
    if (dlvScreenImg) {
        dlvScreenImg.addEventListener('click', (e) => {
            e.stopPropagation();
            handleViewerImageClick(e, dlvScreenImg, dlvScreenClick);
        });
    }
    if (dlvCloseBtn) {
        dlvCloseBtn.addEventListener('click', () => {
            if (desktopLiveViewer) desktopLiveViewer.classList.add('hidden');
        });
    }

    async function syncDisplayModeUI() {
        if (!dlvModeBtn || !dlvModeText) return;
        try {
            const mode = await API.getDisplayMode();
            if (mode.isolated_display) {
                dlvModeBtn.classList.add('sandbox-active');
                dlvModeText.textContent = '📦 Sandbox (:99)';
                dlvModeBtn.title = 'Isolated Virtual Display (:99) active. Click to switch to Physical Screen.';
            } else {
                dlvModeBtn.classList.remove('sandbox-active');
                dlvModeText.textContent = '🖥️ Physical';
                dlvModeBtn.title = 'Physical Screen active. Click to switch to Isolated Sandbox (:99).';
            }
        } catch (e) {}
    }

    if (dlvModeBtn) {
        dlvModeBtn.addEventListener('click', async () => {
            try {
                const current = await API.getDisplayMode();
                const targetMode = !current.isolated_display;
                if (targetMode && !current.available) {
                    alert(
                        `Isolated Sandbox requires Xvfb and xdotool.\n\n` +
                        `Status: ${current.reason}\n\n` +
                        `To install:\nsudo apt install -y xvfb xdotool openbox`
                    );
                }
                await API.setDisplayMode(targetMode);
                await syncDisplayModeUI();
            } catch (err) {
                console.error('Failed to toggle display mode:', err);
            }
        });
        syncDisplayModeUI();
    }

    function isLikelyDesktop(text) {
        const t = (text || '').toLowerCase().trim();
        return t.startsWith('/desktop') || /\b(open|launch|start|run|search|click|cursor)\b.*\b(brave|chrome|browser|files|calculator|terminal)\b/.test(t);
    }

    // Image Attachment Elements & State
    let currentAttachedImage = null; // { dataUrl: string, name: string }
    const attachBtn = document.getElementById('attach-btn');
    const imageFileInput = document.getElementById('image-file-input');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreviewThumb = document.getElementById('image-preview-thumb');
    const imagePreviewName = document.getElementById('image-preview-name');
    const removeImageBtn = document.getElementById('remove-image-btn');

    function setAttachedImage(dataUrl, name = 'image.png') {
        currentAttachedImage = { dataUrl, name };
        if (imagePreviewThumb) imagePreviewThumb.src = dataUrl;
        if (imagePreviewName) imagePreviewName.innerText = name;
        if (imagePreviewContainer) imagePreviewContainer.classList.remove('hidden');
        chatInput.focus();
    }

    function clearAttachedImage() {
        currentAttachedImage = null;
        if (imagePreviewThumb) imagePreviewThumb.src = '';
        if (imagePreviewName) imagePreviewName.innerText = '';
        if (imagePreviewContainer) imagePreviewContainer.classList.add('hidden');
        if (imageFileInput) imageFileInput.value = '';
    }

    if (removeImageBtn) {
        removeImageBtn.addEventListener('click', clearAttachedImage);
    }

    if (attachBtn && imageFileInput) {
        attachBtn.addEventListener('click', () => {
            imageFileInput.click();
        });

        imageFileInput.addEventListener('change', (e) => {
            const file = e.target.files && e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (evt) => {
                    setAttachedImage(evt.target.result, file.name);
                };
                reader.readAsDataURL(file);
            }
        });
    }

    // Clipboard Paste Listener (Ctrl+V)
    document.addEventListener('paste', (e) => {
        if (e.clipboardData && e.clipboardData.items) {
            for (const item of e.clipboardData.items) {
                if (item.type.startsWith('image/')) {
                    const file = item.getAsFile();
                    if (file) {
                        const reader = new FileReader();
                        reader.onload = (evt) => {
                            setAttachedImage(evt.target.result, `clipboard_${Date.now().toString().slice(-4)}.png`);
                        };
                        reader.readAsDataURL(file);
                        break;
                    }
                }
            }
        }
    });

    // Drag and Drop Listeners
    ['dragenter', 'dragover'].forEach(eventName => {
        document.addEventListener(eventName, (e) => {
            if (e.dataTransfer && Array.from(e.dataTransfer.types).includes('Files')) {
                e.preventDefault();
                chatForm.classList.add('drag-over');
            }
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        document.addEventListener(eventName, (e) => {
            e.preventDefault();
            chatForm.classList.remove('drag-over');
        });
    });

    document.addEventListener('drop', (e) => {
        e.preventDefault();
        chatForm.classList.remove('drag-over');
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (evt) => {
                    setAttachedImage(evt.target.result, file.name);
                };
                reader.readAsDataURL(file);
            }
        }
    });

    // Configure Marked.js
    if (window.marked) {
        marked.setOptions({
            highlight: function(code, lang) {
                if (window.hljs && lang && hljs.getLanguage(lang)) {
                    return hljs.highlight(code, { language: lang }).value;
                }
                return window.hljs ? hljs.highlightAuto(code).value : code;
            },
            breaks: true
        });
    }

    // Auto-resize textarea
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
    });

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // Mode Selector
    modePills.forEach(pill => {
        pill.addEventListener('click', () => {
            modePills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            currentMode = pill.dataset.mode;
        });
    });

    // Model Management
    async function loadModels() {
        try {
            availableModels = await API.getModels();
            if (!modelSelect) return;
            modelSelect.innerHTML = '';
            availableModels.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.id;
                const isCloud = m.provider === 'openai';
                const icon = isCloud ? '☁️' : '⚡';
                const vision = m.has_vision ? '👁️ ' : '';
                const lock = m.locked ? ' — 🔑 key needed' : '';
                opt.innerText = `${vision}${icon} ${m.name}${m.badge ? ` (${m.badge})` : ''}${lock}`;
                if (m.is_active) {
                    opt.selected = true;
                    currentModelId = m.id;
                }
                modelSelect.appendChild(opt);
            });
            if (currentModelId) {
                updateModelUI(currentModelId);
            }
        } catch (e) {
            console.error('Failed to load models:', e);
        }
    }

    function updateModelUI(modelId) {
        const model = availableModels.find(m => m.id === modelId);
        if (!model) return;
        const isCloud = model.provider === 'openai';

        // Update vision / provider badge
        if (modelBadge) {
            if (isCloud) {
                modelBadge.classList.remove('hidden');
                modelBadge.innerText = model.has_vision ? 'CLOUD • VISION' : 'CLOUD';
            } else if (model.has_vision) {
                modelBadge.classList.remove('hidden');
                modelBadge.innerText = 'VISION';
            } else {
                modelBadge.classList.add('hidden');
            }
        }

        // Update input hint
        if (inputHint) {
            if (isCloud) {
                inputHint.innerText = `${model.name} • ChatGPT Cloud${model.has_vision ? ' • Vision' : ''} • Multi-Tier Memory Active`;
            } else if (model.has_vision) {
                inputHint.innerText = `${model.name} • Native Multimodal Vision Active • Multi-Tier Memory Active`;
            } else {
                inputHint.innerText = `${model.name} • Uncensored Text Active • Multi-Tier Memory Active`;
            }
        }

        // GPU offload dropdown is only meaningful for local models
        if (gpuModeSelect) {
            gpuModeSelect.disabled = isCloud;
            gpuModeSelect.title = isCloud ? 'GPU offload applies to local models only' : 'GPU Offload Layers';

            if (!isCloud) {
                gpuModeSelect.innerHTML = '';
                const recOpt = document.createElement('option');
                recOpt.value = model.recommended_ngl;
                recOpt.innerText = `GPU (${model.recommended_ngl}L)`;
                recOpt.selected = true;
                gpuModeSelect.appendChild(recOpt);

                if (model.total_layers !== model.recommended_ngl) {
                    const fullOpt = document.createElement('option');
                    fullOpt.value = model.total_layers;
                    fullOpt.innerText = `Full GPU (${model.total_layers}L)`;
                    gpuModeSelect.appendChild(fullOpt);
                }

                const cpuOpt = document.createElement('option');
                cpuOpt.value = '0';
                cpuOpt.innerText = 'CPU';
                gpuModeSelect.appendChild(cpuOpt);
            } else {
                gpuModeSelect.innerHTML = '<option value="0">Cloud</option>';
            }
        }
    }

    if (modelSelect) {
        modelSelect.addEventListener('change', async (e) => {
            const selectedId = e.target.value;
            const selectedModel = availableModels.find(m => m.id === selectedId);

            // Cloud model chosen without a key -> open the key modal instead
            if (selectedModel && selectedModel.locked) {
                modelSelect.value = currentModelId;
                openKeyModal();
                return;
            }

            currentModelId = selectedId;
            updateModelUI(selectedId);

            // Show switching state
            const isCloud = selectedModel && selectedModel.provider === 'openai';
            statusDot.className = 'dot dot-offline';
            statusText.innerText = 'Switching Model...';
            toggleEngineBtn.innerText = 'Loading...';
            toggleEngineBtn.disabled = true;

            const ngl = selectedModel ? selectedModel.recommended_ngl : null;

            try {
                await API.selectModel(selectedId, ngl, !isCloud);
            } catch (err) {
                console.error('Failed to switch model:', err);
            } finally {
                toggleEngineBtn.disabled = false;
                setTimeout(updateSystemStatus, 1500);
            }
        });
    }

    // Check Status
    async function updateSystemStatus() {
        try {
            const status = await API.getStatus();
            activeProvider = status.provider || 'local';
            if (status.active_model_id && status.active_model_id !== currentModelId) {
                currentModelId = status.active_model_id;
                if (modelSelect) modelSelect.value = currentModelId;
                updateModelUI(currentModelId);
            }

            const isCloud = activeProvider === 'openai';

            if (status.engine_ready) {
                statusDot.className = 'dot dot-online';
                statusText.innerText = isCloud ? 'ChatGPT Online' : 'Engine Online';
                toggleEngineBtn.innerText = isCloud ? 'Cloud' : 'Stop';
                toggleEngineBtn.disabled = isCloud;
            } else if (isCloud) {
                statusDot.className = 'dot dot-offline';
                statusText.innerText = 'API Key Missing';
                toggleEngineBtn.innerText = 'Cloud';
                toggleEngineBtn.disabled = true;
            } else if (status.process_status && (status.process_status.running || status.process_status.is_loading)) {
                statusDot.className = 'dot dot-offline';
                statusText.innerText = status.process_status.status_message || 'Loading Model...';
                toggleEngineBtn.innerText = 'Stop';
                toggleEngineBtn.disabled = false;
            } else {
                statusDot.className = 'dot dot-offline';
                statusText.innerText = (status.process_status && status.process_status.status_message) || 'Engine Offline';
                toggleEngineBtn.innerText = 'Start Engine';
                toggleEngineBtn.disabled = false;
            }

            factCounter.innerText = status.dossier_facts_count || '0';
        } catch (e) {
            statusDot.className = 'dot dot-offline';
            statusText.innerText = 'Server Offline';
        }
    }

    // Engine Toggle
    toggleEngineBtn.addEventListener('click', async () => {
        const status = await API.getStatus();
        if ((status.provider || 'local') === 'openai') {
            return; // Cloud provider needs no local engine control
        }
        if (status.process_status && status.process_status.running) {
            await API.stopServer();
        } else {
            const gpuLayers = gpuModeSelect ? parseInt(gpuModeSelect.value, 10) : null;
            toggleEngineBtn.innerText = 'Starting...';
            await API.startServer(gpuLayers, currentModelId);
        }
        setTimeout(updateSystemStatus, 1500);
    });

    // ChatGPT / OpenAI API Key Modal
    function showKeyError(msg) {
        if (!openaiKeyError) return;
        openaiKeyError.innerText = msg;
        openaiKeyError.classList.remove('hidden');
    }

    function clearKeyError() {
        if (!openaiKeyError) return;
        openaiKeyError.innerText = '';
        openaiKeyError.classList.add('hidden');
    }

    async function refreshKeyStatus() {
        try {
            const s = await API.getOpenAIConfig();
            if (!openaiKeyStatus) return;
            if (s.has_key) {
                const src = s.key_source === 'env' ? 'environment variable OPENAI_API_KEY' : 'saved key file';
                openaiKeyStatus.className = 'key-status-line online';
                openaiKeyStatus.innerHTML = `✅ Key active (${src}) — <strong>${s.key_hint || ''}</strong><br>` +
                    `Base URL: <code>${s.base_url}</code> • ${s.models.length} ChatGPT models available`;
            } else {
                openaiKeyStatus.className = 'key-status-line offline';
                openaiKeyStatus.innerHTML = '⚠️ No OpenAI API key configured. Paste your key below to unlock the ☁️ ChatGPT models.';
            }
            if (s.base_url && openaiBaseUrlInput && !openaiBaseUrlInput.value) {
                openaiBaseUrlInput.placeholder = s.base_url;
            }
        } catch (e) {
            if (openaiKeyStatus) {
                openaiKeyStatus.className = 'key-status-line offline';
                openaiKeyStatus.innerText = 'Could not read OpenAI config: ' + e.message;
            }
        }
    }

    async function openKeyModal() {
        clearKeyError();
        if (openaiKeyInput) openaiKeyInput.value = '';
        if (apiKeyModal) apiKeyModal.classList.remove('hidden');
        await refreshKeyStatus();
    }

    function closeKeyModal() {
        if (apiKeyModal) apiKeyModal.classList.add('hidden');
    }

    if (openaiKeyBtn) openaiKeyBtn.addEventListener('click', openKeyModal);
    if (closeKeyModalBtn) closeKeyModalBtn.addEventListener('click', closeKeyModal);
    if (apiKeyModal) {
        apiKeyModal.addEventListener('click', (e) => {
            if (e.target === apiKeyModal) closeKeyModal();
        });
    }

    if (saveOpenaiKeyBtn) {
        saveOpenaiKeyBtn.addEventListener('click', async () => {
            const key = (openaiKeyInput ? openaiKeyInput.value : '').trim();
            const baseUrl = (openaiBaseUrlInput ? openaiBaseUrlInput.value : '').trim();
            if (!key) {
                showKeyError('Please paste your OpenAI API key (starts with "sk-").');
                return;
            }
            clearKeyError();
            saveOpenaiKeyBtn.disabled = true;
            saveOpenaiKeyBtn.innerText = 'Validating…';
            try {
                await API.saveOpenAIKey(key, baseUrl || null);
                await refreshKeyStatus();
                await loadModels();
                await updateSystemStatus();
                if (openaiKeyInput) openaiKeyInput.value = '';
            } catch (err) {
                showKeyError(err.message);
            } finally {
                saveOpenaiKeyBtn.disabled = false;
                saveOpenaiKeyBtn.innerText = 'Save & Validate';
            }
        });
    }

    if (deleteOpenaiKeyBtn) {
        deleteOpenaiKeyBtn.addEventListener('click', async () => {
            clearKeyError();
            try {
                const res = await API.deleteOpenAIKey();
                await refreshKeyStatus();
                await loadModels();
                await updateSystemStatus();
                if (res.has_key) {
                    showKeyError(res.message);
                }
            } catch (err) {
                showKeyError(err.message);
            }
        });
    }

    // Load Sessions
    async function loadSessions() {
        const sessions = await API.getSessions();
        sessionListEl.innerHTML = '';
        if (sessions.length === 0) {
            const newSess = await API.createSession();
            currentSessionId = newSess.id;
            loadSessions();
            return;
        }

        if (!currentSessionId && sessions.length > 0) {
            currentSessionId = sessions[0].id;
        }

        sessions.forEach(sess => {
            const item = document.createElement('div');
            item.className = `session-item ${sess.id === currentSessionId ? 'active' : ''}`;
            item.innerHTML = `
                <span class="session-title">${sess.title || 'Conversation'}</span>
                <button class="session-del-btn" title="Delete">&times;</button>
            `;

            item.addEventListener('click', (e) => {
                if (e.target.classList.contains('session-del-btn')) {
                    e.stopPropagation();
                    deleteSession(sess.id);
                } else {
                    switchSession(sess.id);
                }
            });

            sessionListEl.appendChild(item);
        });

        if (currentSessionId) {
            loadMessages(currentSessionId);
        }
    }

    async function switchSession(id) {
        currentSessionId = id;
        loadSessions();
        loadMessages(id);
    }

    async function deleteSession(id) {
        await API.deleteSession(id);
        if (currentSessionId === id) {
            currentSessionId = null;
        }
        loadSessions();
    }

    newChatBtn.addEventListener('click', async () => {
        const newSess = await API.createSession('New Conversation', currentMode);
        currentSessionId = newSess.id;
        loadSessions();
        showWelcomeHero();
    });

    function showWelcomeHero() {
        messagesContainer.innerHTML = `
            <div class="welcome-hero">
                <div class="hero-logo">/</div>
                <h2>Uncensored Grok Local</h2>
                <p>Truthful, candid, and direct. Retains biographical facts and secrets forever, even within low-context limits.</p>
                <div class="hero-suggestions">
                    <button class="suggestion-chip" data-prompt="Let's be completely candid. What do you think about people giving up on their lifelong dreams?">"What do you think about giving up on dreams?"</button>
                    <button class="suggestion-chip" data-prompt="My name is Asin, 21. What are your core thoughts on brutal honesty vs comforting lies?">"My name is Asin. Honest thoughts on brutal honesty?"</button>
                    <button class="suggestion-chip" data-prompt="Roast me playfully, but tell me one genuinely truthful piece of advice for dealing with parental pressure.">"Roast me, but give real advice on parents."</button>
                </div>
            </div>
        `;
        attachSuggestionHandlers();
    }

    function attachSuggestionHandlers() {
        document.querySelectorAll('.suggestion-chip').forEach(btn => {
            btn.addEventListener('click', () => {
                chatInput.value = btn.dataset.prompt;
                chatForm.dispatchEvent(new Event('submit'));
            });
        });
    }

    // Load Messages
    async function loadMessages(sessionId) {
        const messages = await API.getMessages(sessionId);
        messagesContainer.innerHTML = '';
        if (messages.length === 0) {
            showWelcomeHero();
            return;
        }

        messages.forEach(msg => {
            renderMessage(msg.role, msg.content, null, msg.image_url);
        });
        scrollToBottom();
    }

    function renderMessage(role, content, meta = null, imageUrl = null) {
        const row = document.createElement('div');
        row.className = `message-row ${role}`;

        let inner = '';
        if (role === 'assistant') {
            inner += `<div class="message-avatar">/</div>`;
        }

        inner += `<div class="message-bubble">`;
        if (imageUrl) {
            inner += `<div class="message-attachment"><img src="${imageUrl}" alt="Uploaded image" onclick="window.open('${imageUrl}', '_blank')"></div>`;
        }

        const parsedContent = window.marked ? marked.parse(content) : content;
        inner += `${parsedContent}`;

        if (meta) {
            inner += `<div class="message-meta">${meta}</div>`;
        }

        inner += `</div>`;
        row.innerHTML = inner;
        messagesContainer.appendChild(row);
        scrollToBottom();
        return row;
    }

    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Send Message / SSE Streaming
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        const attachedImg = currentAttachedImage ? currentAttachedImage.dataUrl : null;
        if ((!text && !attachedImg) || isStreaming) return;

        const promptText = text || (attachedImg ? "Analyze and describe this image without filters." : "");
        const previewImg = attachedImg;

        // Clear input and attachment state
        chatInput.value = '';
        chatInput.style.height = 'auto';
        clearAttachedImage();

        // Remove welcome hero if visible
        const hero = messagesContainer.querySelector('.welcome-hero');
        if (hero) hero.remove();

        // Render user message with attached image
        renderMessage('user', promptText, null, previewImg);

        // Create empty assistant response row
        const assistantRow = document.createElement('div');
        assistantRow.className = 'message-row assistant';
        assistantRow.innerHTML = `
            <div class="message-avatar">/</div>
            <div class="message-bubble"><span class="cursor-blink">▍</span></div>
        `;
        messagesContainer.appendChild(assistantRow);
        scrollToBottom();

        const bubbleEl = assistantRow.querySelector('.message-bubble');
        let fullResponse = '';
        let lastTokSec = 0;
        let lastTotalTok = 0;

        isStreaming = true;
        sendBtn.disabled = true;

        currentAbortController = API.streamChat(
            currentSessionId,
            promptText,
            currentMode,
            attachedImg,
            (budget, serverImageUrl) => {
                // Telemetry callback
                telemetryBar.innerText = `Context: ${budget.total_estimated_tokens}t (Dossier: ${budget.dossier_tokens}t • Episodic: ${budget.episodic_tokens}t)`;
                if (currentMode === 'desktop' || promptText.startsWith('/desktop') || isLikelyDesktop(promptText)) {
                    if (stopDesktopBtn) stopDesktopBtn.classList.remove('hidden');
                    if (desktopLiveViewer) desktopLiveViewer.classList.remove('hidden');
                }
            },
            (token, tokSec, totalTok) => {
                // Token stream chunk
                fullResponse += token;
                lastTokSec = tokSec;
                lastTotalTok = totalTok;
                bubbleEl.innerHTML = (window.marked ? marked.parse(fullResponse) : fullResponse) + '<span class="cursor-blink">▍</span>';
                scrollToBottom();
            },
            (errorMessage) => {
                bubbleEl.innerHTML = `<span style="color: var(--accent-red)">⚠️ ${errorMessage}</span>`;
                isStreaming = false;
                sendBtn.disabled = false;
                if (stopDesktopBtn) stopDesktopBtn.classList.add('hidden');
            },
            () => {
                // Done
                isStreaming = false;
                sendBtn.disabled = false;
                if (stopDesktopBtn) stopDesktopBtn.classList.add('hidden');
                const metaHtml = `<span>⚡ ${lastTokSec} tok/s • ${lastTotalTok} tokens</span> <button class="btn-tiny copy-btn" title="Copy response">Copy</button>`;
                bubbleEl.innerHTML = (window.marked ? marked.parse(fullResponse) : fullResponse) + `<div class="message-meta">${metaHtml}</div>`;
                
                const copyBtn = bubbleEl.querySelector('.copy-btn');
                if (copyBtn) {
                    copyBtn.addEventListener('click', () => {
                        navigator.clipboard.writeText(fullResponse);
                        copyBtn.innerText = 'Copied!';
                        setTimeout(() => { copyBtn.innerText = 'Copy'; }, 2000);
                    });
                }

                updateSystemStatus();
                loadDossier();
                loadSessions(); // Refresh titles
            },
            (desktopStep) => {
                isDesktopRunning = true;
                // Live Desktop Pilot Event
                if (desktopLiveViewer) {
                    desktopLiveViewer.classList.remove('hidden');
                    if (stopDesktopBtn) stopDesktopBtn.classList.remove('hidden');
                    if (desktopStep.step !== undefined) {
                        dlvStepBadge.textContent = `Step ${desktopStep.step}`;
                    }
                    if (desktopStep.thought) {
                        dlvThought.textContent = desktopStep.thought;
                    }
                    if (desktopStep.action) {
                        dlvAction.textContent = desktopStep.action;
                    }
                    if (desktopStep.screenshot_url) {
                        if (dlvPlaceholder) dlvPlaceholder.style.display = 'none';
                        const img = new Image();
                        img.onload = () => {
                            dlvScreenImg.src = desktopStep.screenshot_url;
                            dlvScreenImg.classList.add('loaded');
                        };
                        img.src = desktopStep.screenshot_url;
                    }
                    const pulseEl = desktopLiveViewer.querySelector('.dlv-pulse');
                    if (pulseEl) {
                        if (desktopStep.status === 'completed') {
                            pulseEl.style.backgroundColor = 'var(--accent-green)';
                            pulseEl.style.animation = 'none';
                        } else if (desktopStep.status === 'error' || desktopStep.status === 'aborted') {
                            pulseEl.style.backgroundColor = 'var(--accent-red)';
                            pulseEl.style.animation = 'none';
                        } else {
                            pulseEl.style.backgroundColor = 'var(--accent-green)';
                            pulseEl.style.animation = 'dlvPulseAnim 1.4s infinite';
                        }
                    }
                }

                // Update PiP Always-on-Top HUD
                updatePipWindow(desktopStep);

                // Auto-open PiP on first step if supported and not opened yet
                if (desktopStep.step === 1 && !pipWindow) {
                    openPipWindow();
                }

                if (desktopStep.status === 'completed' || desktopStep.status === 'error' || desktopStep.status === 'aborted') {
                    isDesktopRunning = false;
                }
            }
        );
    });

    // Emergency Stop Desktop Pilot
    const stopDesktopBtn = document.getElementById('stop-desktop-btn');
    if (stopDesktopBtn) {
        stopDesktopBtn.addEventListener('click', async () => {
            await API.stopDesktop();
            stopDesktopBtn.classList.add('hidden');
        });
    }

    // Export Conversation
    const exportChatBtn = document.getElementById('export-chat-btn');
    if (exportChatBtn) {
        exportChatBtn.addEventListener('click', async () => {
            if (!currentSessionId) return;
            const data = await API.exportSession(currentSessionId);
            const blob = new Blob([data.markdown], { type: 'text/markdown' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${(data.title || 'grok_conversation').replace(/[^a-zA-Z0-9_-]/g, '_')}.md`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }

    // Keyboard Shortcuts (Esc to close drawer/modal, Ctrl+M to toggle)
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (!memoryDrawer.classList.contains('hidden')) memoryDrawer.classList.add('hidden');
            if (apiKeyModal && !apiKeyModal.classList.contains('hidden')) apiKeyModal.classList.add('hidden');
        } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'm') {
            e.preventDefault();
            memoryDrawer.classList.toggle('hidden');
            if (!memoryDrawer.classList.contains('hidden')) {
                loadDossier();
                loadEpisodic();
            }
        }
    });

    // Memory Inspector Drawer
    openMemoryBtn.addEventListener('click', () => {
        memoryDrawer.classList.remove('hidden');
        loadDossier();
        loadEpisodic();
    });

    closeMemoryBtn.addEventListener('click', () => {
        memoryDrawer.classList.add('hidden');
    });

    drawerTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            drawerTabs.forEach(t => t.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
        });
    });

    async function loadDossier() {
        const facts = await API.getDossier();
        factCounter.innerText = facts.length;
        dossierList.innerHTML = '';
        if (facts.length === 0) {
            dossierList.innerHTML = '<p class="tab-desc">No facts extracted yet. Chat with Grok and it will automatically remember your details!</p>';
            return;
        }

        facts.forEach(f => {
            const card = document.createElement('div');
            card.className = 'fact-card';
            card.innerHTML = `
                <div class="fact-info">
                    <span class="fact-badge">${f.category}</span>
                    <div class="fact-key">${f.key.replace(/_/g, ' ').toUpperCase()}</div>
                    <div class="fact-val">${f.value}</div>
                </div>
                <button class="fact-delete-btn" title="Delete Fact">&times;</button>
            `;
            card.querySelector('.fact-delete-btn').addEventListener('click', async () => {
                await API.deleteFact(f.key);
                loadDossier();
                updateSystemStatus();
            });
            dossierList.appendChild(card);
        });
    }

    addFactForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const key = document.getElementById('fact-key').value.trim();
        const value = document.getElementById('fact-value').value.trim();
        const category = document.getElementById('fact-category').value;
        if (!key || !value) return;

        await API.setFact(key, value, category);
        addFactForm.reset();
        loadDossier();
        updateSystemStatus();
    });

    async function loadEpisodic(query = '') {
        const memories = await API.searchEpisodic(query);
        episodicList.innerHTML = '';
        if (memories.length === 0) {
            episodicList.innerHTML = '<p class="tab-desc">No memories found.</p>';
            return;
        }

        memories.forEach(m => {
            const card = document.createElement('div');
            card.className = 'memory-card';
            card.innerHTML = `
                <div class="memory-card-header">
                    <span>${m.created_at || 'Memory'}</span>
                    <button class="fact-delete-btn" data-id="${m.id}">&times;</button>
                </div>
                <div class="memory-content">${m.content}</div>
            `;
            card.querySelector('.fact-delete-btn').addEventListener('click', async () => {
                await API.deleteEpisodic(m.id);
                loadEpisodic(query);
            });
            episodicList.appendChild(card);
        });
    }

    episodicSearchBtn.addEventListener('click', () => {
        loadEpisodic(episodicSearchInput.value.trim());
    });

    // Initial setup
    loadModels();
    updateSystemStatus();
    loadSessions();
    setInterval(updateSystemStatus, 4000);
});
