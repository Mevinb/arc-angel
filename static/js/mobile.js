/* GROK Pocket — paired mobile companion. */
(() => {
    'use strict';

    const API = '/api/v2';
    const TOKEN_KEY = 'grok.mobile.token';
    const state = { token: '', device: null, pairing: false };

    const $ = (sel) => document.querySelector(sel);
    const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
    const fmt = (value) => {
        if (!value) return '—';
        const d = new Date(value);
        return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString();
    };
    const pill = (status) => `<span class="m-pill ${esc(status || 'unknown')}">${esc(status || 'unknown')}</span>`;

    function storedToken() {
        try { return localStorage.getItem(TOKEN_KEY) || ''; } catch { return ''; }
    }

    function storeToken(token) {
        state.token = token || '';
        try {
            if (token) localStorage.setItem(TOKEN_KEY, token);
            else localStorage.removeItem(TOKEN_KEY);
        } catch { /* private browsing can reject storage */ }
    }

    function toast(message, kind = '') {
        const node = $('#m-toast');
        node.textContent = message;
        node.className = `m-toast ${kind}`;
        node.hidden = false;
        clearTimeout(toast._timer);
        toast._timer = setTimeout(() => { node.hidden = true; }, 4200);
    }

    function setBusy(button, busy) {
        if (!button) return;
        button.disabled = busy;
        if (busy) {
            button.dataset.label = button.textContent;
            button.textContent = 'Working…';
        } else if (button.dataset.label) {
            button.textContent = button.dataset.label;
        }
    }

    async function request(path, options = {}, auth = true) {
        const headers = { ...(options.headers || {}) };
        if (auth && state.token) headers['X-Mobile-Token'] = state.token;
        const opts = { ...options, headers };
        if (opts.body !== undefined && typeof opts.body !== 'string') {
            headers['Content-Type'] = 'application/json';
            opts.body = JSON.stringify(opts.body);
        }
        const response = await fetch(`${API}${path}`, opts);
        const text = await response.text();
        let data = null;
        if (text) {
            try { data = JSON.parse(text); } catch { data = text; }
        }
        if (!response.ok) {
            const detail = data && typeof data === 'object' ? data.detail : data;
            const error = new Error(detail || `Request failed (${response.status})`);
            error.status = response.status;
            if (response.status === 401 && auth) {
                showPair('This pairing expired. Create a new pairing in Command Center → Settings.');
            }
            throw error;
        }
        return data;
    }

    function showPair(message) {
        stopRefresh();
        state.device = null;
        $('#app-view').hidden = true;
        $('#pair-view').hidden = false;
        $('#m-refresh').hidden = true;
        if (message !== undefined) {
            const error = $('#pair-error');
            error.textContent = message;
            error.hidden = !message;
        }
    }

    function showApp() {
        $('#pair-view').hidden = true;
        $('#app-view').hidden = false;
        $('#m-refresh').hidden = false;
        $('#pair-error').hidden = true;
    }

    function cleanPairingUrl() {
        try { history.replaceState(null, '', '/mobile'); } catch { /* ignore */ }
    }

    async function pair(deviceId, code) {
        state.pairing = true;
        try {
            const data = await request('/mobile/pair', {
                method: 'POST', body: { device_id: deviceId.trim(), code: code.trim() },
            }, false);
            storeToken(data.token);
            cleanPairingUrl();
            toast(`Paired ${data.name || 'phone'}.`, 'ok');
            await loadApp();
        } finally {
            state.pairing = false;
        }
    }

    function renderBrief(brief, safety) {
        const data = brief?.data || {};
        const tasks = data.upcoming_tasks || [];
        const bills = data.bills_due_soon || [];
        const habits = (data.habits || []).filter((h) => !h.done);
        const parts = [];
        $('#m-date').textContent = String(data.date || new Date().toISOString().slice(0, 10)).toUpperCase();
        $('#m-brief-time').textContent = new Date().toLocaleTimeString();

        if (data.pending_approvals) {
            parts.push(`<div class="m-list-item"><div class="m-item-title"><span>${esc(data.pending_approvals)} approval${data.pending_approvals === 1 ? '' : 's'} waiting</span><span class="m-pill awaiting_approval">review</span></div><div class="m-item-meta">Open Command Center to approve or deny.</div></div>`);
        }
        if (data.notifications_unread) {
            parts.push(`<div class="m-list-item"><div class="m-item-title"><span>${esc(data.notifications_unread)} unread notification${data.notifications_unread === 1 ? '' : 's'}</span></div></div>`);
        }
        if (tasks.length) {
            parts.push(tasks.slice(0, 5).map((task) => `
                <div class="m-list-item">
                    <div class="m-item-title"><span>${esc(task.title)}</span>${pill(task.status)}</div>
                    <div class="m-item-meta">${esc(task.due_at ? `due ${fmt(task.due_at)}` : task.id)}</div>
                </div>`).join(''));
        }
        if (bills.length) {
            parts.push(bills.slice(0, 4).map((bill) => `
                <div class="m-list-item">
                    <div class="m-item-title"><span>${esc(bill.name)}</span><span class="m-pill paused">bill</span></div>
                    <div class="m-item-meta">${esc(bill.amount)} · due ${esc(bill.due_on)}</div>
                </div>`).join(''));
        }
        if (habits.length) {
            parts.push(`<div class="m-list-item"><div class="m-item-title"><span>Habits still open</span><span class="m-pill queued">${habits.length}</span></div><div class="m-item-meta">${habits.slice(0, 6).map((h) => esc(h.name)).join(' · ')}</div></div>`);
        }
        if (!parts.length) {
            parts.push('<div class="m-empty">Nothing needs you right now. Start something below or enjoy the quiet.</div>');
        }
        $('#m-brief').innerHTML = parts.join('');

        const chip = $('#m-safety');
        chip.className = 'm-chip';
        if (safety?.kill_switch) {
            chip.textContent = 'Kill switch';
            chip.classList.add('danger');
        } else if (safety?.offline_mode) {
            chip.textContent = 'Offline mode';
            chip.classList.add('warn');
        } else {
            chip.textContent = 'Safety active';
        }
    }

    async function loadBrief() {
        const [brief, safety] = await Promise.all([
            request('/mobile/brief'),
            request('/safety', {}, false).catch(() => null),
        ]);
        renderBrief(brief, safety);
    }

    async function loadApp() {
        const [device] = await Promise.all([request('/mobile/me')]);
        state.device = device;
        showApp();
        $('#m-device-name').textContent = device.name || 'Paired phone';
        $('#m-device-meta').textContent = `${device.platform || 'mobile'} · paired ${fmt(device.paired_at)}`;
        await loadBrief();
    }

    function resultHtml(kind, data) {
        if (kind === 'note') {
            const note = data?.note || data;
            return `<strong>Note saved.</strong><div class="m-item-meta">${esc(note?.title || 'Knowledge note')} · ${esc(note?.id || '')}</div>`;
        }
        const task = data?.task || data;
        return `<strong>Task queued.</strong><div class="m-item-meta">${esc(task?.title || 'Atlas task')} · ${esc(task?.id || '')} ${task?.status ? pill(task.status) : ''}</div><a href="/platform#tasks">Open in Command Center →</a>`;
    }

    let refreshTimer = null;
    function startRefresh() {
        stopRefresh();
        refreshTimer = setInterval(() => {
            if (!document.hidden && state.token) loadBrief().catch((err) => console.debug('brief refresh failed', err));
        }, 60000);
    }
    function stopRefresh() {
        if (refreshTimer) clearInterval(refreshTimer);
        refreshTimer = null;
    }

    function wire() {
        $('#pair-form').addEventListener('submit', async (event) => {
            event.preventDefault();
            const button = event.target.querySelector('button[type="submit"]');
            if (state.pairing) return;
            setBusy(button, true);
            $('#pair-error').hidden = true;
            try {
                await pair($('#pair-device').value, $('#pair-code').value);
            } catch (err) {
                const node = $('#pair-error');
                node.textContent = err.message;
                node.hidden = false;
            } finally {
                setBusy(button, false);
            }
        });

        $('#command-form').addEventListener('submit', async (event) => {
            event.preventDefault();
            const text = $('#command-text').value.trim();
            if (!text) return;
            const button = event.target.querySelector('button[type="submit"]');
            setBusy(button, true);
            try {
                const data = await request('/mobile/command', { method: 'POST', body: { text } });
                const node = $('#command-result');
                node.innerHTML = resultHtml(data.kind, data);
                node.hidden = false;
                $('#command-text').value = '';
                toast('Sent to Atlas.', 'ok');
                await loadBrief();
            } catch (err) {
                toast(err.message, 'error');
            } finally {
                setBusy(button, false);
            }
        });

        $('#share-form').addEventListener('submit', async (event) => {
            event.preventDefault();
            const text = $('#share-text').value.trim();
            if (!text) return;
            const button = event.target.querySelector('button[type="submit"]');
            setBusy(button, true);
            try {
                const data = await request('/mobile/share', {
                    method: 'POST',
                    body: { title: $('#share-title').value.trim(), text, auto_start: true },
                });
                const node = $('#share-result');
                node.innerHTML = resultHtml('task', data);
                node.hidden = false;
                $('#share-title').value = '';
                $('#share-text').value = '';
                toast('Shared item queued.', 'ok');
                await loadBrief();
            } catch (err) {
                toast(err.message, 'error');
            } finally {
                setBusy(button, false);
            }
        });

        $('#m-refresh').addEventListener('click', async () => {
            try {
                await loadBrief();
                toast('Brief refreshed.', 'ok');
            } catch (err) {
                toast(err.message, 'error');
            }
        });

        $('#m-signout').addEventListener('click', async () => {
            if (!confirm('Unpair this phone? You will need a new pairing code.')) return;
            const deviceId = state.device?.id;
            try {
                if (deviceId) await request(`/mobile/devices/${encodeURIComponent(deviceId)}`, { method: 'DELETE' });
            } catch (err) {
                console.debug('revoke failed', err);
            }
            storeToken('');
            $('#m-brief').innerHTML = '';
            showPair('This phone is no longer paired.');
            toast('Unpaired.', 'ok');
        });

        window.addEventListener('online', () => { $('#m-status').textContent = 'Local'; });
        window.addEventListener('offline', () => { $('#m-status').textContent = 'Browser offline'; });
    }

    async function boot() {
        state.token = storedToken();
        wire();
        const params = new URLSearchParams(location.search);
        const deviceId = params.get('device') || '';
        const code = params.get('code') || '';
        if (deviceId) $('#pair-device').value = deviceId;
        if (code) $('#pair-code').value = code;

        if (state.token) {
            try {
                await loadApp();
                startRefresh();
                return;
            } catch (err) {
                console.debug('stored pairing unusable', err);
                storeToken('');
            }
        }
        if (deviceId && code) {
            $('#pair-error').hidden = true;
            try {
                await pair(deviceId, code);
                startRefresh();
                return;
            } catch (err) {
                const node = $('#pair-error');
                node.textContent = err.message;
                node.hidden = false;
            }
        }
        showPair();
    }

    document.addEventListener('DOMContentLoaded', () => {
        boot().catch((err) => {
            showPair(err.message || 'Unable to start the mobile companion.');
        });
    });
})();
