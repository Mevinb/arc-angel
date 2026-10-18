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

    async getMessages(sessionId) {
        const res = await fetch(`/api/sessions/${sessionId}/messages`);
        return res.json();
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
     * SSE Streaming Chat with optional image support
     */
    streamChat(sessionId, message, mode, imageData, onTelemetry, onToken, onError, onDone, onDesktopStep) {
        // If 4th arg is a function, imageData was omitted
        let imgData = null;
        let cbTelemetry = onTelemetry;
        let cbToken = onToken;
        let cbError = onError;
        let cbDone = onDone;
        let cbDesktopStep = onDesktopStep;
        if (typeof imageData === 'function') {
            imgData = null;
            cbTelemetry = imageData;
            cbToken = onTelemetry;
            cbError = onToken;
            cbDone = onError;
            cbDesktopStep = onDone;
        } else {
            imgData = imageData;
        }

        const controller = new AbortController();

        fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                session_id: sessionId, 
                message, 
                mode,
                image_data: imgData 
            }),
            signal: controller.signal
        }).then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            function readChunk() {
                reader.read().then(({ done, value }) => {
                    if (done) {
                        if (cbDone) cbDone();
                        return;
                    }

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop(); // keep last incomplete line

                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (!trimmed || !trimmed.startsWith('data: ')) continue;
                        
                        try {
                            const data = JSON.parse(trimmed.substring(6));
                            if (data.type === 'telemetry') {
                                if (cbTelemetry) cbTelemetry(data.budget, data.image_url);
                            } else if (data.type === 'token') {
                                if (cbToken) cbToken(data.token, data.tok_sec, data.total_tokens);
                            } else if (data.type === 'desktop_step') {
                                if (cbDesktopStep) cbDesktopStep(data);
                            } else if (data.type === 'error') {
                                if (cbError) cbError(data.message);
                            } else if (data.type === 'done') {
                                if (cbDone) cbDone(data.full_response);
                            }
                        } catch (err) {
                            console.error('SSE JSON parse error', err, trimmed);
                        }
                    }

                    readChunk();
                }).catch(err => {
                    if (err.name !== 'AbortError') {
                        onError(err.message);
                    }
                });
            }

            readChunk();
        }).catch(err => {
            onError(err.message);
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
    }
};
