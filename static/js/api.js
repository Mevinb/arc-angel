/**
 * API client for Local Grok Chatbot
 */
const API = {
    async getStatus() {
        const res = await fetch('/api/status');
        return res.json();
    },

    async getModels() {
        const res = await fetch('/api/models');
        return res.json();
    },

    async selectModel(modelId, gpuLayers = null, restart = true) {
        const res = await fetch('/api/models/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId, gpu_layers: gpuLayers, restart: restart })
        });
        return res.json();
    },

    async startServer(gpuLayers = null, modelId = null) {
        const res = await fetch('/api/server/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ gpu_layers: gpuLayers, model_id: modelId })
        });
        return res.json();
    },

    async stopServer() {
        const res = await fetch('/api/server/stop', { method: 'POST' });
        return res.json();
    },

    /**
     * OpenAI (ChatGPT) API key management
     */
    async getOpenAIConfig() {
        const res = await fetch('/api/openai/status');
        return res.json();
    },

    async saveOpenAIKey(apiKey, baseUrl = null) {
        const res = await fetch('/api/openai/key', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_key: apiKey, base_url: baseUrl })
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
        return data;
    },

    async deleteOpenAIKey() {
        const res = await fetch('/api/openai/key', { method: 'DELETE' });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
        return data;
    },

    async getSessions() {
        const res = await fetch('/api/sessions');
        return res.json();
    },

    async createSession(title = "New Conversation", mode = "classic") {
        const res = await fetch('/api/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, mode })
        });
        return res.json();
    },

    async deleteSession(sessionId) {
        const res = await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
        return res.json();
    },

    async renameSession(sessionId, title) {
        const res = await fetch(`/api/sessions/${sessionId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title })
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
        return data;
    },

    async getMessages(sessionId) {
        const res = await fetch(`/api/sessions/${sessionId}/messages`);
        return res.json();
    },

    async deleteMessage(sessionId, messageId) {
        const res = await fetch(`/api/sessions/${sessionId}/messages/${messageId}`, { method: 'DELETE' });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
        return data;
    },

    async exportSession(sessionId) {
        const res = await fetch(`/api/sessions/${sessionId}/export`);
        return res.json();
    },

    async getDossier() {
        const res = await fetch('/api/memory/dossier');
        return res.json();
    },

    async setFact(key, value, category = "general") {
        const res = await fetch('/api/memory/dossier', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key, value, category })
        });
        return res.json();
    },

    async deleteFact(key) {
        const res = await fetch(`/api/memory/dossier/${encodeURIComponent(key)}`, {
            method: 'DELETE'
        });
        return res.json();
    },

    async searchEpisodic(query = '') {
        const url = query ? `/api/memory/episodic?q=${encodeURIComponent(query)}` : '/api/memory/episodic';
        const res = await fetch(url);
        return res.json();
    },

    async deleteEpisodic(id) {
        const res = await fetch(`/api/memory/episodic/${id}`, { method: 'DELETE' });
        return res.json();
    },

    /**
     * Attachment upload (documents / code). Returns {url, name, type, size, chars}.
     */
    async uploadAttachment(file) {
        const res = await fetch('/api/attachments', {
            method: 'POST',
            headers: { 'Content-Type': 'application/octet-stream', 'X-Filename': encodeURIComponent(file.name) },
            body: file
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
        return data;
    },

    /**
     * Generic Server-Sent Events reader over a POST response body.
     * Handlers: an object mapping event `type` -> callback(data).
     */
    _readSSE(response, handlers, controller) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        const dispatch = (data) => {
            const fn = handlers[data.type] || handlers['*'];
            if (fn) fn(data);
        };

        const pump = () => {
            reader.read().then(({ done, value }) => {
                if (done) {
                    if (handlers._done) handlers._done();
                    return;
                }
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();
                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed || !trimmed.startsWith('data: ')) continue;
                    try {
                        dispatch(JSON.parse(trimmed.substring(6)));
                    } catch (err) {
                        console.error('SSE JSON parse error', err, trimmed);
                    }
                }
                pump();
            }).catch(err => {
                if (err.name !== 'AbortError' && handlers._error) handlers._error(err.message);
            });
        };
        pump();
    },

    /**
     * SSE Streaming Chat with optional image support.
     * `handlers` may be an object {telemetry, token, error, done, desktop_step,
     * task_update, subgoal_update, _done, _error}. For backwards compatibility,
     * positional callbacks are also accepted.
     */
    streamChat(sessionId, message, mode, imageData, ...rest) {
        let imgData = null;
        let handlers = {};

        // Detect legacy positional callback signature
        if (typeof imageData === 'function') {
            // imageData was omitted
            const [onTelemetry, onToken, onError, onDone, onDesktopStep, onTaskUpdate, onSubgoalUpdate] = [imageData, ...rest];
            handlers = {
                telemetry: onTelemetry, token: onToken, error: onError, done: onDone,
                desktop_step: onDesktopStep, task_update: onTaskUpdate, subgoal_update: onSubgoalUpdate,
                _done: () => { if (onDone) onDone(); }, _error: onError
            };
        } else {
            imgData = imageData;
            const last = rest[rest.length - 1];
            if (last && typeof last === 'object' && !Array.isArray(last) && (last.token || last._done || last.telemetry || last.task_update)) {
                handlers = last;
            } else if (typeof rest[0] === 'function') {
                const [onTelemetry, onToken, onError, onDone, onDesktopStep, onTaskUpdate, onSubgoalUpdate] = rest;
                handlers = {
                    telemetry: onTelemetry, token: onToken, error: onError, done: onDone,
                    desktop_step: onDesktopStep, task_update: onTaskUpdate, subgoal_update: onSubgoalUpdate,
                    _done: () => { if (onDone) onDone(); }, _error: onError
                };
            }
        }

        const controller = new AbortController();
        const body = { session_id: sessionId, message, mode, image_data: imgData };
        if (handlers.attachment) body.attachment = handlers.attachment;
        if (handlers.regenerate) body.regenerate = true;
        if (handlers.edit_message_id) body.edit_message_id = handlers.edit_message_id;

        fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: controller.signal
        }).then(response => {
            if (!response.ok) throw new Error(`HTTP error ${response.status}`);
            this._readSSE(response, handlers, controller);
        }).catch(err => {
            if (handlers._error) handlers._error(err.message);
            else if (handlers.error) handlers.error(err.message);
        });

        return controller;
    },

    async stopDesktop() {
        const res = await fetch('/api/desktop/stop', { method: 'POST' });
        return res.json();
    },

    async getDesktopStatus() {
        const res = await fetch('/api/desktop/status');
        return res.json();
    },

    async sendManualAction(action) {
        const res = await fetch('/api/desktop/manual_action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(action)
        });
        return res.json();
    },

    async getDisplayMode() {
        const res = await fetch('/api/desktop/display_mode');
        return res.json();
    },

    async setDisplayMode(isolated) {
        const res = await fetch('/api/desktop/display_mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ isolated_display: isolated })
        });
        return res.json();
    },

    /**
     * Hierarchical Long-Horizon Task Orchestrator endpoints
     */
    async listTasks(limit = 30) {
        const res = await fetch(`/api/orchestrator/tasks?limit=${limit}`);
        return res.json();
    },

    async getTask(taskId) {
        const res = await fetch(`/api/orchestrator/tasks/${taskId}`);
        return res.json();
    },

    async startTask(taskId) {
        const res = await fetch(`/api/orchestrator/tasks/${taskId}/start`, { method: 'POST' });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
        return data;
    },

    async retryTask(taskId) {
        const res = await fetch(`/api/orchestrator/tasks/${taskId}/retry`, { method: 'POST' });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
        return data;
    },

    async getTaskEvents(taskId, sinceId = 0) {
        const res = await fetch(`/api/orchestrator/tasks/${taskId}/events?since_id=${sinceId}`);
        return res.json();
    },

    async listTaskFiles(taskId, relPath = '') {
        const res = await fetch(`/api/orchestrator/tasks/${taskId}/files?rel_path=${encodeURIComponent(relPath)}`);
        return res.json();
    },

    taskFileUrl(taskId, relPath) {
        return `/api/orchestrator/tasks/${taskId}/file?rel_path=${encodeURIComponent(relPath)}`;
    },

    /**
     * Live task event stream (SSE via GET).
     */
    streamTaskEvents(taskId, handlers = {}) {
        const controller = new AbortController();
        fetch(`/api/orchestrator/tasks/${taskId}/events/stream`, { signal: controller.signal })
            .then(response => {
                if (!response.ok) throw new Error(`HTTP error ${response.status}`);
                this._readSSE(response, handlers, controller);
            })
            .catch(err => {
                if (err.name !== 'AbortError' && handlers._error) handlers._error(err.message);
            });
        return controller;
    },

    async pauseTask(taskId) {
        const res = await fetch('/api/orchestrator/pause', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: taskId })
        });
        return res.json();
    },

    async resumeTask(taskId) {
        const res = await fetch('/api/orchestrator/resume', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: taskId })
        });
        return res.json();
    },

    async abortTask(taskId) {
        const res = await fetch('/api/orchestrator/abort', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: taskId })
        });
        return res.json();
    }
};
