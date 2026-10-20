/* GROK Command Center — v2 API client and views. */
(() => {
    'use strict';

    const $ = (sel, root = document) => root.querySelector(sel);
    const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
    const state = {
        view: 'overview',
        agents: [],
        tasks: [],
        selectedTask: null,
        taskDetail: null,
        research: null,
        poll: null,
    };

    const TITLES = {
        overview: 'Overview', tasks: 'Tasks & Runs', approvals: 'Approval inbox',
        knowledge: 'Knowledge vault', research: 'Research', connectors: 'Connectors',
        automations: 'Automations', files: 'Files & Code', audit: 'Audit trail',
        settings: 'Settings',
    };

    const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
    const attr = (value) => esc(value);
    const fmt = (value) => {
        if (!value) return '—';
        const d = new Date(value);
        return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString();
    };
    const short = (value, n = 120) => {
        const text = String(value ?? '').replace(/\s+/g, ' ').trim();
        return text.length > n ? text.slice(0, n) + '…' : text;
    };
    const money = (n) => `$${Number(n || 0).toFixed(4)}`;
    const num = (n) => Number(n || 0).toLocaleString();
    const pill = (status) => `<span class="pf-pill ${attr(status)}">${esc(status || 'unknown')}</span>`;
    const empty = (msg) => `<div class="pf-empty">${esc(msg)}</div>`;
    const tags = (list) => (list || []).map((t) => `<span class="pf-tag">${esc(t)}</span>`).join('');

    function toast(message, kind = '') {
        const node = $('#pf-toast');
        node.textContent = message;
        node.className = `pf-toast ${kind}`;
        clearTimeout(toast._timer);
        toast._timer = setTimeout(() => node.classList.add('hidden'), 4200);
    }

    function showError(error) {
        const node = $('#pf-error');
        node.textContent = error?.message || String(error);
        node.classList.remove('hidden');
        clearTimeout(showError._timer);
        showError._timer = setTimeout(() => node.classList.add('hidden'), 7000);
        console.error(error);
    }

    function clearError() {
        $('#pf-error').classList.add('hidden');
    }

    async function api(path, options = {}) {
        const opts = { ...options, headers: { ...(options.headers || {}) } };
        if (opts.body !== undefined && typeof opts.body !== 'string') {
            opts.headers['Content-Type'] = 'application/json';
            opts.body = JSON.stringify(opts.body);
        }
        const response = await fetch(`/api/v2${path}`, opts);
        const text = await response.text();
        let data = null;
        if (text) {
            try { data = JSON.parse(text); } catch { data = text; }
        }
        if (!response.ok) {
            const detail = data && typeof data === 'object' ? data.detail : data;
            throw new Error(detail || `HTTP ${response.status}`);
        }
        return data;
    }

    function setPoll(fn, ms = 2200) {
        if (state.poll) clearInterval(state.poll);
        state.poll = setInterval(() => {
            if (document.hidden) return;
            fn().catch((err) => console.debug('poll failed', err));
        }, ms);
    }

    function stopPoll() {
        if (state.poll) clearInterval(state.poll);
        state.poll = null;
    }

    async function updateShell() {
        const clock = $('#pf-clock');
        if (clock) clock.textContent = new Date().toLocaleTimeString();
        try {
            const [safety, approvals] = await Promise.all([
                api('/safety'), api('/approvals?status=pending&limit=100'),
            ]);
            const badge = $('#pf-approval-count');
            if (badge) badge.textContent = String(approvals.length);
            const node = $('#pf-safety-pill');
            const text = $('#pf-safety-text');
            node.classList.toggle('danger', !!safety.kill_switch);
            node.classList.toggle('warn', !safety.kill_switch && (!!safety.offline_mode || !!safety.lock_enabled));
            text.textContent = safety.kill_switch ? 'KILL SWITCH ENGAGED'
                : safety.offline_mode ? 'Offline mode' : 'Local · safety active';
        } catch (err) {
            console.debug('shell status unavailable', err);
        }
    }

    async function navigate(view, push = true) {
        if (!TITLES[view]) view = 'overview';
        state.view = view;
        state.research = null;
        stopPoll();
        if (push && location.hash !== `#${view}`) history.replaceState(null, '', `#${view}`);
        $('#pf-title').textContent = TITLES[view];
        $$('.pf-nav-item').forEach((b) => b.classList.toggle('active', b.dataset.view === view));
        clearError();
        $('#pf-view').innerHTML = '<div class="pf-empty">Loading…</div>';
        try {
            if (view === 'overview') await renderOverview();
            else if (view === 'tasks') await renderTasks();
            else if (view === 'approvals') await renderApprovals();
            else if (view === 'knowledge') await renderKnowledge();
            else if (view === 'research') await renderResearch();
            else if (view === 'connectors') await renderConnectors();
            else if (view === 'automations') await renderAutomations();
            else if (view === 'files') await renderFiles();
            else if (view === 'audit') await renderAudit();
            else if (view === 'settings') await renderSettings();
        } catch (err) {
            showError(err);
            $('#pf-view').innerHTML = `<div class="pf-alert pf-alert-error">${esc(err.message)}</div>`;
        }
        updateShell();
    }

    // ------------------------------------------------------------------ overview
    async function renderOverview() {
        const [overview, activity, approvals] = await Promise.all([
            api('/overview'), api('/activity?limit=9'), api('/approvals?status=pending&limit=8'),
        ]);
        const d = overview.dashboard || {};
        const taskCounts = d.tasks || {};
        const active = ['queued', 'running', 'paused', 'awaiting_approval', 'waiting', 'interrupted']
            .reduce((n, key) => n + Number(taskCounts[key] || 0), 0);
        const runs = d.runs || {};
        const agents = overview.agents || [];
        const profiles = overview.profiles || {};
        const activeProfile = profiles.active || {};
        const recentRuns = overview.recent_runs || [];
        const runsRows = recentRuns.length ? recentRuns : [];
        const approvalHtml = approvals.length ? approvals.map((a) => `
            <div class="pf-list-item">
                <div class="pf-card-head" style="margin-bottom:6px">
                    <span class="pf-list-title">${esc(a.tool_name || a.action || 'Approval needed')}</span>
                    ${pill(a.risk_class || 'approval')}
                </div>
                <div class="pf-list-body">${esc(short(a.reason || a.summary || 'Review this action', 160))}</div>
                <div class="pf-actions" style="margin-top:8px">
                    <button class="pf-btn pf-btn-primary pf-btn-sm" data-action="approve" data-id="${attr(a.id)}">Approve</button>
                    <button class="pf-btn pf-btn-danger pf-btn-sm" data-action="deny" data-id="${attr(a.id)}">Deny</button>
                </div>
            </div>`).join('') : empty('No pending approvals.');
        const activityHtml = activity.length ? activity.map((a) => `
            <div class="pf-list-item">
                <div class="pf-list-title">${esc(a.title)}</div>
                <div class="pf-list-meta">${esc(fmt(a.created_at))} · ${esc(a.kind)} · ${esc(a.severity)}</div>
            </div>`).join('') : empty('No activity yet.');

        $('#pf-view').innerHTML = `
            <div class="pf-grid pf-grid-4">
                <div class="pf-card"><div class="pf-stat-label">Active work</div><div class="pf-stat-value">${num(active)}</div><div class="pf-stat-note">${num(taskCounts.completed || 0)} completed</div></div>
                <div class="pf-card"><div class="pf-stat-label">Pending approvals</div><div class="pf-stat-value">${num(approvals.length)}</div><div class="pf-stat-note">Human control stays explicit</div></div>
                <div class="pf-card"><div class="pf-stat-label">Knowledge sources</div><div class="pf-stat-value">${num(d.knowledge?.sources || 0)}</div><div class="pf-stat-note">${num(d.knowledge?.chars || 0)} chars indexed</div></div>
                <div class="pf-card"><div class="pf-stat-label">Spend</div><div class="pf-stat-value">${money(d.cost_usd)}</div><div class="pf-stat-note">${num(d.tokens || 0)} tokens</div></div>
            </div>

            <div class="pf-grid pf-grid-2">
                <section class="pf-card">
                    <div class="pf-card-head"><div><h2>Give Atlas a job</h2><div class="pf-card-sub">The orchestrator plans, uses tools, checkpoints, and stops for approvals.</div></div>${pill('runs ' + num(runs.running || 0))}</div>
                    <form id="quick-task-form" class="pf-form">
                        <div class="pf-field"><label for="q-goal">Goal</label><textarea class="pf-textarea" id="q-goal" required placeholder="e.g. Research the best local embedding models and write a cited comparison into my workspace."></textarea></div>
                        <div class="pf-form-row">
                            <div class="pf-field"><label for="q-agent">Agent</label><select class="pf-select" id="q-agent">${agents.map((a) => `<option value="${attr(a.id)}" ${a.id === 'orchestrator' ? 'selected' : ''}>${esc(a.name)} — ${esc(a.role)}</option>`).join('')}</select></div>
                            <div class="pf-field"><label for="q-steps">Max steps</label><input class="pf-input" id="q-steps" type="number" min="1" max="100" value="24"></div>
                        </div>
                        <label class="pf-check"><input type="checkbox" id="q-start" checked> Start immediately</label>
                        <div class="pf-actions"><button class="pf-btn pf-btn-primary" type="submit">Create task</button><span class="pf-dim pf-small">Runs in an isolated workspace.</span></div>
                    </form>
                </section>
                <section class="pf-card">
                    <div class="pf-card-head"><h3>Operating mode</h3><span class="pf-pill ok">local-first</span></div>
                    <dl class="pf-kv">
                        <dt>Provider</dt><dd>${esc(activeProfile.provider || 'local')} · ${esc(activeProfile.model || 'active model')}</dd>
                        <dt>Tool surface</dt><dd>${num((overview.tools || []).length)} typed tools</dd>
                        <dt>Agent gallery</dt><dd>${num(agents.length)} agents</dd>
                        <dt>Safety</dt><dd id="overview-safety">Checking…</dd>
                        <dt>Runs</dt><dd>${num(runs.running || 0)} running · ${num(runs.failed || 0)} failed</dd>
                    </dl>
                    <div class="pf-actions" style="margin-top:14px">
                        <button class="pf-btn" data-action="go" data-view="approvals">Open approvals</button>
                        <button class="pf-btn" data-action="go" data-view="settings">Safety settings</button>
                    </div>
                </section>
            </div>

            <div class="pf-grid pf-grid-3">
                <section class="pf-card">
                    <div class="pf-card-head"><h3>Needs you</h3><span class="pf-badge">${approvals.length}</span></div>
                    <div class="pf-list">${approvalHtml}</div>
                </section>
                <section class="pf-card">
                    <div class="pf-card-head"><h3>Recent runs</h3><button class="pf-btn pf-btn-sm" data-action="go" data-view="tasks">All tasks</button></div>
                    <div class="pf-list">${runsRows.length ? runsRows.map((r) => `
                        <div class="pf-list-item selectable" data-action="open-task" data-id="${attr(r.task_id || '')}">
                            <div class="pf-card-head" style="margin:0"><span class="pf-list-title">${esc(r.title || r.task_id || r.id)}</span>${pill(r.status)}</div>
                            <div class="pf-list-meta">${esc(fmt(r.created_at))} · ${esc(r.id)}</div>
                        </div>`).join('') : empty('No runs yet. Start one above.')}</div>
                </section>
                <section class="pf-card">
                    <div class="pf-card-head"><h3>Activity</h3><button class="pf-btn pf-btn-sm" data-action="go" data-view="audit">Audit</button></div>
                    <div class="pf-list">${activityHtml}</div>
                </section>
            </div>`;

        api('/safety').then((s) => {
            const node = $('#overview-safety');
            if (node) node.textContent = s.kill_switch ? 'Kill switch engaged' : s.offline_mode ? 'Offline mode' : 'Active · approvals on';
        }).catch(() => {});
    }

    // --------------------------------------------------------------------- tasks
    async function renderTasks() {
        if (!state.agents.length) {
            state.agents = await api('/agents');
        }
        const data = await api('/tasks?limit=100');
        state.tasks = data.tasks || [];
        const options = state.agents.map((a) => `<option value="${attr(a.id)}" ${a.id === 'orchestrator' ? 'selected' : ''}>${esc(a.name)} — ${esc(a.role)}</option>`).join('');
        $('#pf-view').innerHTML = `
            <section class="pf-card">
                <div class="pf-card-head"><div><h2>New task</h2><div class="pf-card-sub">A goal plus an agent is enough; budgets are enforced by the runtime.</div></div></div>
                <form id="task-create-form" class="pf-form">
                    <div class="pf-form-row">
                        <div class="pf-field"><label for="t-title">Title (optional)</label><input class="pf-input" id="t-title" placeholder="Short label"></div>
                        <div class="pf-field"><label for="t-agent">Agent</label><select class="pf-select" id="t-agent">${options}</select></div>
                        <div class="pf-field"><label for="t-steps">Max steps</label><input class="pf-input" id="t-steps" type="number" min="1" max="100" value="24"></div>
                    </div>
                    <div class="pf-field"><label for="t-goal">Goal</label><textarea class="pf-textarea" id="t-goal" required placeholder="What should the agent accomplish? Include constraints and where artifacts should go."></textarea></div>
                    <div class="pf-actions"><label class="pf-check"><input type="checkbox" id="t-start" checked> Start immediately</label><button class="pf-btn pf-btn-primary" type="submit">Create task</button></div>
                </form>
            </section>
            <div class="pf-split">
                <section class="pf-card">
                    <div class="pf-card-head"><h3>Task board <span class="pf-dim">${state.tasks.length}</span></h3><button class="pf-btn pf-btn-sm" data-action="refresh-tasks">Refresh</button></div>
                    <div class="pf-list" id="task-list">${taskListHtml()}</div>
                </section>
                <section class="pf-card" id="task-detail-card">${empty('Select a task to inspect its run, steps, events and controls.')}</section>
            </div>`;
        if (state.selectedTask) {
            await loadTaskDetail(state.selectedTask);
        }
    }

    function taskListHtml() {
        if (!state.tasks.length) return empty('No tasks yet.');
        return state.tasks.map((t) => `
            <div class="pf-list-item selectable ${state.selectedTask === t.id ? 'active' : ''}" data-action="select-task" data-id="${attr(t.id)}">
                <div class="pf-card-head" style="margin:0;gap:8px"><span class="pf-list-title">${esc(t.title)}</span>${pill(t.status)}</div>
                <div class="pf-list-meta"><span>${esc(t.agent_id || 'agent')}</span><span>${esc(t.id)}</span><span>${esc(fmt(t.updated_at))}</span></div>
                ${t.error ? `<div class="pf-list-body pf-error-text">${esc(t.error)}</div>` : ''}
            </div>`).join('');
    }

    async function loadTaskDetail(taskId) {
        state.selectedTask = taskId;
        const task = await api(`/tasks/${taskId}`);
        const run = task.latest_run ? await api(`/runs/${task.latest_run.id}`) : null;
        const events = run ? await api(`/runs/${run.id}/events?limit=80`) : [];
        state.taskDetail = { task, run, events };
        const card = $('#task-detail-card');
        if (!card) return;
        card.innerHTML = taskDetailHtml(task, run, events);
        $$('#task-list .pf-list-item').forEach((n) => n.classList.toggle('active', n.dataset.id === taskId));
        setPoll(async () => {
            if (state.view !== 'tasks' || !state.selectedTask) return;
            const fresh = await api(`/tasks/${state.selectedTask}`);
            const freshRun = fresh.latest_run ? await api(`/runs/${fresh.latest_run.id}`) : null;
            const freshEvents = freshRun ? await api(`/runs/${freshRun.id}/events?limit=80`) : [];
            state.taskDetail = { task: fresh, run: freshRun, events: freshEvents };
            const node = $('#task-detail-card');
            if (node) node.innerHTML = taskDetailHtml(fresh, freshRun, freshEvents);
            const list = $('#task-list');
            if (list) {
                state.tasks = (await api('/tasks?limit=100')).tasks || [];
                list.innerHTML = taskListHtml();
            }
        }, 2200);
    }

    function taskDetailHtml(task, run, events) {
        const status = task.status;
        const controls = [];
        if (status === 'queued' || status === 'waiting') controls.push(`<button class="pf-btn pf-btn-primary pf-btn-sm" data-action="task-start" data-id="${attr(task.id)}">Start</button>`);
        if (status === 'running') controls.push(`<button class="pf-btn pf-btn-sm" data-action="task-pause" data-id="${attr(task.id)}">Pause</button>`);
        if (status === 'paused' || status === 'interrupted') controls.push(`<button class="pf-btn pf-btn-primary pf-btn-sm" data-action="task-resume" data-id="${attr(task.id)}">Resume</button>`);
        if (['failed', 'completed', 'partially_completed', 'cancelled'].includes(status)) controls.push(`<button class="pf-btn pf-btn-sm" data-action="task-retry" data-id="${attr(task.id)}">Retry</button>`);
        if (!['completed', 'cancelled'].includes(status)) controls.push(`<button class="pf-btn pf-btn-danger pf-btn-sm" data-action="task-cancel" data-id="${attr(task.id)}">Cancel</button>`);
        controls.push(`<button class="pf-btn pf-btn-sm" data-action="task-open-approvals">Approvals</button>`);
        controls.push(`<button class="pf-btn pf-btn-danger pf-btn-sm" data-action="task-delete" data-id="${attr(task.id)}">Delete</button>`);

        if (!run) {
            return `<div class="pf-card-head"><div><h2>${esc(task.title)}</h2><div class="pf-card-sub">${esc(task.goal)}</div></div>${pill(status)}</div>
                <div class="pf-actions" style="margin:12px 0">${controls.join('')}</div>
                <div class="pf-empty">No run has been created for this task.</div>`;
        }
        const steps = run.steps || [];
        const maxSteps = Number(run.budget?.max_steps || 24);
        const pct = Math.min(100, Math.round((steps.length / Math.max(1, maxSteps)) * 100));
        const eventsHtml = (events || []).slice(-70).reverse().map((e) => `
            <div class="pf-event"><span>${esc(fmt(e.created_at).split(', ')[1] || fmt(e.created_at))}</span><span class="pf-event-type">${esc(e.type)}</span><span>${esc(short(JSON.stringify(e.payload || {}), 180))}</span></div>`).join('');
        const stepsHtml = steps.length ? steps.map((s) => `
            <div class="pf-step"><span class="pf-step-idx">${esc(s.idx)}</span><div><div class="pf-step-name">${esc(s.title || s.tool_name || 'Step')}</div><div class="pf-step-out">${esc(short(JSON.stringify(s.output || {}), 160))}</div></div>${pill(s.status)}</div>`).join('') : empty('No steps recorded.');
        const result = run.result || task.result;
        return `
            <div class="pf-card-head"><div><h2>${esc(task.title)}</h2><div class="pf-card-sub">${esc(task.id)} · ${esc(task.agent_id || 'agent')}</div></div>${pill(status)}</div>
            <div class="pf-actions" style="margin:10px 0 14px">${controls.join('')}</div>
            <dl class="pf-kv">
                <dt>Goal</dt><dd>${esc(task.goal)}</dd>
                <dt>Run</dt><dd>${esc(run.id)} · attempt ${esc(run.attempt)}</dd>
                <dt>Workspace</dt><dd class="pf-mono">${esc(task.workspace_dir || '—')}</dd>
                <dt>Usage</dt><dd>${num(run.tokens_in)} in / ${num(run.tokens_out)} out · ${money(run.cost_usd)}</dd>
                <dt>Budget</dt><dd>${steps.length}/${maxSteps} steps · ${num(run.budget?.max_tool_calls || 0)} tool calls max</dd>
                <dt>Started</dt><dd>${esc(fmt(run.started_at))}</dd>
                ${run.error ? `<dt>Error</dt><dd class="pf-error-text">${esc(run.error)}</dd>` : ''}
            </dl>
            <div class="pf-meter" title="Step budget"><i style="width:${pct}%"></i></div>
            ${run.checkpoint?.pending_approval ? `<div class="pf-alert pf-alert-warn" style="margin-top:12px">Waiting for approval: <strong>${esc(run.checkpoint.pending_approval.tool)}</strong> <button class="pf-btn pf-btn-sm" data-action="go" data-view="approvals">Review</button></div>` : ''}
            <div class="pf-grid pf-grid-2" style="margin-top:14px">
                <div><div class="pf-card-head"><h3>Steps</h3></div><div class="pf-steps">${stepsHtml}</div></div>
                <div><div class="pf-card-head"><h3>Run events</h3></div><div class="pf-events">${eventsHtml || empty('No events.')}</div></div>
            </div>
            ${result ? `<div class="pf-card-head" style="margin-top:14px"><h3>Result</h3></div><pre class="pf-code">${esc(result)}</pre>` : ''}`;
    }

    // ----------------------------------------------------------------- approvals
    async function renderApprovals() {
        const all = await api('/approvals?limit=100');
        const pending = all.filter((a) => a.status === 'pending');
        const resolved = all.filter((a) => a.status !== 'pending').slice(0, 20);
        const card = (a) => `
            <div class="pf-list-item">
                <div class="pf-card-head" style="margin-bottom:6px"><div><span class="pf-list-title">${esc(a.tool_name || a.action || 'Approval')}</span><div class="pf-list-meta">${esc(fmt(a.created_at))} · ${esc(a.agent_id || 'system')}</div></div>${pill(a.status)}</div>
                <div class="pf-list-body">${esc(a.reason || a.summary || 'Review before allowing this action.')}</div>
                <div class="pf-list-meta" style="margin-top:6px">${a.risk_class ? pill(a.risk_class) : ''}${a.run_id ? `<span class="pf-mono">${esc(a.run_id)}</span>` : ''}</div>
                ${a.payload ? `<details style="margin-top:8px"><summary class="pf-small pf-muted">Payload</summary><pre class="pf-code">${esc(JSON.stringify(a.payload, null, 2))}</pre></details>` : ''}
                ${a.status === 'pending' ? `<div class="pf-actions" style="margin-top:9px"><button class="pf-btn pf-btn-primary pf-btn-sm" data-action="approve" data-id="${attr(a.id)}">Approve &amp; continue</button><button class="pf-btn pf-btn-danger pf-btn-sm" data-action="deny" data-id="${attr(a.id)}">Deny</button></div>` : `<div class="pf-list-meta" style="margin-top:7px">${esc(a.resolution_note || 'Resolved')}</div>`}
            </div>`;
        $('#pf-view').innerHTML = `
            <div class="pf-grid pf-grid-2">
                <section class="pf-card"><div class="pf-card-head"><div><h2>Pending</h2><div class="pf-card-sub">Agents pause until you decide. Denials are respected by the runtime.</div></div><span class="pf-badge">${pending.length}</span></div><div class="pf-list">${pending.length ? pending.map(card).join('') : empty('Inbox zero. No actions are waiting.')}</div></section>
                <section class="pf-card"><div class="pf-card-head"><h3>Recently resolved</h3><button class="pf-btn pf-btn-sm" data-action="refresh-view">Refresh</button></div><div class="pf-list">${resolved.length ? resolved.map(card).join('') : empty('No resolved approvals yet.')}</div></section>
            </div>`;
    }

    async function resolveApproval(id, approved) {
        await api(`/approvals/${id}/resolve`, { method: 'POST', body: { approved, note: approved ? 'Approved in Command Center' : 'Denied in Command Center' } });
        toast(approved ? 'Approved.' : 'Denied.', 'ok');
        await renderApprovals();
        updateShell();
    }

    // ----------------------------------------------------------------- knowledge
    async function renderKnowledge() {
        const [stats, data] = await Promise.all([api('/knowledge/stats'), api('/knowledge/sources?limit=50')]);
        const sources = data.sources || [];
        $('#pf-view').innerHTML = `
            <div class="pf-grid pf-grid-4">
                <div class="pf-card"><div class="pf-stat-label">Sources</div><div class="pf-stat-value">${num(stats.sources)}</div></div>
                <div class="pf-card"><div class="pf-stat-label">Chunks</div><div class="pf-stat-value">${num(stats.chunks)}</div></div>
                <div class="pf-card"><div class="pf-stat-label">Characters</div><div class="pf-stat-value">${num(stats.chars)}</div></div>
                <div class="pf-card"><div class="pf-stat-label">Embeddings</div><div class="pf-stat-value">${num(stats.embedded_chunks)}</div><div class="pf-stat-note">${num(stats.embedding_dim || 384)}-dim local vectors</div></div>
            </div>
            <div class="pf-grid pf-grid-2">
                <section class="pf-card">
                    <div class="pf-card-head"><div><h2>Ask your vault</h2><div class="pf-card-sub">Hybrid retrieval over notes, files, web pages and transcripts.</div></div></div>
                    <form id="knowledge-ask-form" class="pf-form">
                        <div class="pf-field"><label for="k-question">Question</label><textarea class="pf-textarea" id="k-question" required placeholder="What have I saved about…?"></textarea></div>
                        <div class="pf-actions"><label class="pf-check"><input type="checkbox" id="k-synthesize" checked> Synthesize an answer</label><button class="pf-btn pf-btn-primary" type="submit">Search vault</button></div>
                    </form>
                    <div id="knowledge-results" style="margin-top:14px"></div>
                </section>
                <section class="pf-card">
                    <div class="pf-card-head"><h3>Add a note</h3><button class="pf-btn pf-btn-sm" data-action="reindex-knowledge">Reindex</button></div>
                    <form id="knowledge-add-form" class="pf-form">
                        <div class="pf-field"><label for="k-title">Title</label><input class="pf-input" id="k-title" required placeholder="Short title"></div>
                        <div class="pf-field"><label for="k-content">Content</label><textarea class="pf-textarea" id="k-content" required placeholder="Durable note, quote, or reference…"></textarea></div>
                        <div class="pf-form-row"><div class="pf-field"><label for="k-tags">Tags (comma separated)</label><input class="pf-input" id="k-tags" placeholder="work, idea"></div><div class="pf-field"><label for="k-profile">Profile</label><input class="pf-input" id="k-profile" value="personal"></div></div>
                        <button class="pf-btn pf-btn-primary" type="submit">Save to vault</button>
                    </form>
                </section>
            </div>
            <section class="pf-card"><div class="pf-card-head"><h3>Recent sources</h3><span class="pf-dim">${sources.length}</span></div><div class="pf-list">${sources.length ? sources.map((s) => `
                <div class="pf-list-item"><div class="pf-card-head" style="margin:0"><span class="pf-list-title">${esc(s.title)}</span>${pill(s.source_type)}</div><div class="pf-list-meta">${esc(fmt(s.created_at))} · ${num(s.chars || s.char_count || 0)} chars ${tags(s.tags)}</div></div>`).join('') : empty('The vault is empty. Add a note or connect a source.')}</div></section>`;
    }

    async function knowledgeSearch(question, synthesize) {
        const data = await api('/knowledge/ask', { method: 'POST', body: { question, synthesize, answer_style: 'concise' } });
        const sources = data.sources || data.results || [];
        const answer = data.answer || data.response || '';
        $('#knowledge-results').innerHTML = `
            ${answer ? `<div class="pf-card" style="background:#0f1510"><div class="pf-card-head"><h3>Answer</h3>${pill('grounded')}</div><div class="pf-report">${esc(answer)}</div></div>` : ''}
            <div class="pf-list" style="margin-top:10px">${sources.length ? sources.map((s) => `
                <div class="pf-list-item"><div class="pf-list-title">${esc(s.title || 'Source')}</div><div class="pf-list-body">${esc(short(s.content || s.snippet || '', 320))}</div><div class="pf-list-meta">${esc(s.source_type || 'source')} ${s.source_id ? `· <span class="pf-mono">${esc(s.source_id)}</span>` : ''}</div></div>`).join('') : empty('No relevant sources found.')}</div>`;
    }

    // ------------------------------------------------------------------ research
    async function renderResearch() {
        const projects = await api('/research?limit=50');
        $('#pf-view').innerHTML = `
            <section class="pf-card">
                <div class="pf-card-head"><div><h2>Launch a research brief</h2><div class="pf-card-sub">Plans queries, fetches sources, extracts claims and produces a cited report in the vault.</div></div></div>
                <form id="research-create-form" class="pf-form">
                    <div class="pf-field"><label for="r-question">Research question</label><textarea class="pf-textarea" id="r-question" required placeholder="What should be investigated, compared, or verified?"></textarea></div>
                    <div class="pf-form-row"><div class="pf-field"><label for="r-max">Max sources</label><input class="pf-input" id="r-max" type="number" min="1" max="30" value="8"></div><div class="pf-field"><label for="r-profile">Model profile</label><select class="pf-select" id="r-profile"><option value="quality">Quality</option><option value="balanced">Balanced</option><option value="fast">Fast</option><option value="local_only">Local only</option></select></div></div>
                    <label class="pf-check"><input type="checkbox" id="r-start" checked> Start now</label>
                    <button class="pf-btn pf-btn-primary" type="submit">Create project</button>
                </form>
            </section>
            <div class="pf-split">
                <section class="pf-card"><div class="pf-card-head"><h3>Projects</h3><span class="pf-badge">${projects.length}</span></div><div class="pf-list">${projects.length ? projects.map((p) => `
                    <div class="pf-list-item selectable" data-action="select-research" data-id="${attr(p.id)}"><div class="pf-card-head" style="margin:0"><span class="pf-list-title">${esc(p.title || p.question)}</span>${pill(p.status)}</div><div class="pf-list-meta">${esc(fmt(p.created_at))} · ${esc(p.id)}</div></div>`).join('') : empty('No research projects yet.')}</div></section>
                <section class="pf-card" id="research-detail">${empty('Select a project to inspect its sources and report.')}</section>
            </div>`;
    }

    async function loadResearch(projectId) {
        const project = await api(`/research/${projectId}`);
        state.research = project;
        const sources = project.sources || project.notes || [];
        $('#research-detail').innerHTML = `
            <div class="pf-card-head"><div><h2>${esc(project.title || project.question)}</h2><div class="pf-card-sub">${esc(project.id)} · ${esc(fmt(project.created_at))}</div></div>${pill(project.status)}</div>
            <div class="pf-actions" style="margin:10px 0"><button class="pf-btn pf-btn-primary pf-btn-sm" data-action="research-task" data-id="${attr(project.id)}">Turn into agent task</button><button class="pf-btn pf-btn-sm" data-action="refresh-research" data-id="${attr(project.id)}">Refresh</button></div>
            <dl class="pf-kv"><dt>Question</dt><dd>${esc(project.question)}</dd><dt>Sources</dt><dd>${sources.length}</dd><dt>Report</dt><dd>${project.report_path ? `<span class="pf-mono">${esc(project.report_path)}</span>` : 'Not written yet'}</dd>${project.error ? `<dt>Error</dt><dd class="pf-error-text">${esc(project.error)}</dd>` : ''}</dl>
            <div class="pf-card-head" style="margin-top:14px"><h3>Sources</h3></div>
            <div class="pf-list">${sources.length ? sources.map((s) => `<div class="pf-list-item"><div class="pf-list-title">${esc(s.title || s.url || 'Source')}</div><div class="pf-list-body">${esc(short(s.content || s.snippet || '', 300))}</div><div class="pf-list-meta">${s.url ? `<a href="${attr(s.url)}" target="_blank" rel="noreferrer">${esc(short(s.url, 80))}</a>` : ''} ${s.credibility ? `· credibility ${esc(s.credibility)}` : ''}</div></div>`).join('') : empty('Sources will appear as the research runs.')}</div>
            <div class="pf-card-head" style="margin-top:14px"><h3>Report</h3></div>
            <div class="pf-report">${esc(project.report || 'The cited report will appear here when synthesis finishes.')}</div>`;
        if (['queued', 'running'].includes(project.status)) {
            setPoll(async () => {
                if (state.view !== 'research' || !state.research) return;
                const fresh = await api(`/research/${state.research.id}`);
                state.research = fresh;
                if (['completed', 'failed'].includes(fresh.status)) {
                    stopPoll();
                    await loadResearch(fresh.id);
                } else {
                    await loadResearch(fresh.id);
                }
            }, 3000);
        }
    }

    // ---------------------------------------------------------------- connectors
    async function renderConnectors() {
        const [catalog, connections] = await Promise.all([api('/connectors'), api('/connectors/connections')]);
        const connectedIds = new Set(connections.map((c) => c.connector_id));
        $('#pf-view').innerHTML = `
            <div class="pf-grid pf-grid-2">
                <section class="pf-card"><div class="pf-card-head"><div><h2>Connections</h2><div class="pf-card-sub">Credentials stay encrypted locally. Sync is read-only; external actions require approval.</div></div><span class="pf-badge">${connections.length}</span></div>
                    <div class="pf-list">${connections.length ? connections.map((c) => `
                        <div class="pf-list-item"><div class="pf-card-head" style="margin:0"><span class="pf-list-title">${esc(c.account_label)} · ${esc(c.connector_id)}</span>${pill(c.status)}</div>
                        <div class="pf-list-meta">${esc(c.id)} · ${esc(fmt(c.updated_at))} ${c.error ? `<span class="pf-error-text">${esc(short(c.error, 90))}</span>` : ''}</div>
                        <div class="pf-actions" style="margin-top:8px"><button class="pf-btn pf-btn-sm" data-action="connector-test" data-id="${attr(c.id)}">Test</button><button class="pf-btn pf-btn-sm" data-action="connector-sync" data-id="${attr(c.id)}">Sync</button><button class="pf-btn pf-btn-danger pf-btn-sm" data-action="connector-disconnect" data-id="${attr(c.id)}">Disconnect</button></div></div>`).join('') : empty('No connections yet.')}</div>
                </section>
                <section class="pf-card"><div class="pf-card-head"><div><h2>Available connectors</h2><div class="pf-card-sub">OAuth providers need client credentials in the environment.</div></div></div>
                    <div class="pf-list">${catalog.map((c) => `
                        <div class="pf-list-item"><div class="pf-card-head" style="margin:0"><span class="pf-list-title">${esc(c.name)}</span><span class="pf-pill ${c.configured ? 'ok' : 'waiting'}">${c.configured ? 'configured' : 'needs setup'}</span></div>
                        <div class="pf-list-body">${esc(c.description)}</div>
                        <div class="pf-list-meta">${esc(c.category)} · ${esc(c.auth)} ${connectedIds.has(c.id) ? '· connected' : ''}</div>
                        <div class="pf-actions" style="margin-top:8px"><button class="pf-btn pf-btn-primary pf-btn-sm" data-action="connector-connect" data-id="${attr(c.id)}">${c.auth === 'oauth' ? 'Connect OAuth' : 'Add credentials'}</button></div></div>`).join('')}</div>
                </section>
            </div>`;
    }

    function openConnectorModal(connectorId, catalog) {
        const meta = catalog.find((c) => c.id === connectorId);
        if (!meta) return;
        const isRss = connectorId === 'rss';
        const isWebhook = connectorId === 'webhook';
        const secretField = isRss
            ? '<div class="pf-field"><label for="cx-feed">Feed URL</label><input class="pf-input" id="cx-feed" required placeholder="https://example.com/feed.xml"></div>'
            : isWebhook
                ? '<div class="pf-field"><label for="cx-url">Webhook target URL</label><input class="pf-input" id="cx-url" required placeholder="https://example.com/hook"></div>'
                : '<div class="pf-field"><label for="cx-token">Access token</label><input class="pf-input" id="cx-token" type="password" required placeholder="Paste token"></div>';
        openModal(`Connect ${meta.name}`, `
            <form id="connector-credential-form" class="pf-form">
                <div class="pf-field"><label for="cx-label">Account label</label><input class="pf-input" id="cx-label" value="Primary"></div>
                ${secretField}
                <div class="pf-alert pf-alert-warn">Credentials are encrypted with the local master key and never written to chat or audit logs.</div>
                <button class="pf-btn pf-btn-primary" type="submit">Save connection</button>
            </form>`);
        $('#connector-credential-form').addEventListener('submit', async (event) => {
            event.preventDefault();
            const credentials = {};
            const metadata = {};
            if (isRss) { credentials.feed_url = $('#cx-feed').value.trim(); metadata.feed_url = credentials.feed_url; }
            else if (isWebhook) { credentials.url = $('#cx-url').value.trim(); metadata.url = credentials.url; }
            else credentials.access_token = $('#cx-token').value.trim();
            try {
                await api(`/connectors/${connectorId}/connect`, {
                    method: 'POST',
                    body: { account_label: $('#cx-label').value.trim() || 'Primary', credentials, metadata },
                });
                closeModal();
                toast('Connection saved.', 'ok');
                await renderConnectors();
            } catch (err) { showError(err); }
        });
    }

    async function startOAuth(connectorId) {
        const redirect = `${location.origin}/api/v2/connectors/oauth/callback`;
        const data = await api(`/connectors/${connectorId}/oauth-url`, { method: 'POST', body: { redirect_uri: redirect } });
        const popup = window.open(data.url, 'grok-oauth', 'width=560,height=720');
        if (!popup) {
            openModal('OAuth link', `<p class="pf-muted">Your browser blocked the popup. Open this link:</p><p><a href="${attr(data.url)}" target="_blank" rel="noreferrer">${esc(data.url)}</a></p>`);
        } else {
            toast('Complete the provider consent window.');
        }
    }

    // --------------------------------------------------------------- automations
    async function renderAutomations() {
        const [automations] = await Promise.all([
            api('/automations'),
            state.agents.length ? Promise.resolve() : api('/agents').then((agents) => { state.agents = agents; }),
        ]);
        $('#pf-view').innerHTML = `
            <section class="pf-card"><div class="pf-card-head"><div><h2>New automation</h2><div class="pf-card-sub">Schedules are checked every 30 seconds. Webhooks are HMAC-signed.</div></div></div>
                <form id="automation-create-form" class="pf-form">
                    <div class="pf-form-row"><div class="pf-field"><label for="a-name">Name</label><input class="pf-input" id="a-name" required placeholder="Morning brief"></div><div class="pf-field"><label for="a-trigger">Trigger</label><select class="pf-select" id="a-trigger"><option value="schedule">Schedule</option><option value="webhook">Webhook</option></select></div><div class="pf-field"><label for="a-every">Every (minutes)</label><input class="pf-input" id="a-every" type="number" min="1" value="1440"></div></div>
                    <div class="pf-field"><label for="a-goal">Task goal</label><textarea class="pf-textarea" id="a-goal" placeholder="What should the agent do when this fires?"></textarea></div>
                    <div class="pf-form-row"><div class="pf-field"><label for="a-action">Action</label><select class="pf-select" id="a-action"><option value="create_task">Create and start task</option><option value="notify">Notify me</option></select></div><div class="pf-field"><label for="a-agent">Agent</label><select class="pf-select" id="a-agent">${state.agents.length ? state.agents.map((a) => `<option value="${attr(a.id)}">${esc(a.name)}</option>`).join('') : '<option value="orchestrator">Atlas</option>'}</select></div></div>
                    <button class="pf-btn pf-btn-primary" type="submit">Create automation</button>
                </form>
            </section>
            <section class="pf-card"><div class="pf-card-head"><h3>Automations</h3><span class="pf-badge">${automations.length}</span></div><div class="pf-list">${automations.length ? automations.map((a) => `
                <div class="pf-list-item"><div class="pf-card-head" style="margin:0"><span class="pf-list-title">${esc(a.name)}</span><span class="pf-pill ${a.enabled ? 'ok' : 'cancelled'}">${a.enabled ? 'enabled' : 'paused'}</span></div>
                <div class="pf-list-meta">${esc(a.trigger_type)} · ${esc(a.action_type)} · next ${esc(fmt(a.next_run_at))} · ${num(a.run_count)} runs ${a.last_status ? `· last ${esc(a.last_status)}` : ''}</div>
                ${a.trigger_type === 'webhook' ? `<div class="pf-list-meta"><span class="pf-mono">${esc(location.origin + '/api/v2/hooks/' + a.id)}</span><button class="pf-btn pf-btn-sm" data-action="copy" data-text="${attr(location.origin + '/api/v2/hooks/' + a.id)}">Copy</button></div>` : ''}
                <div class="pf-actions" style="margin-top:8px"><button class="pf-btn pf-btn-sm" data-action="automation-run" data-id="${attr(a.id)}">Run now</button><button class="pf-btn pf-btn-sm" data-action="automation-toggle" data-id="${attr(a.id)}">${a.enabled ? 'Pause' : 'Enable'}</button><button class="pf-btn pf-btn-danger pf-btn-sm" data-action="automation-delete" data-id="${attr(a.id)}">Delete</button></div></div>`).join('') : empty('No automations configured.')}</div></section>`;
    }

    // --------------------------------------------------------------------- files
    async function renderFiles() {
        $('#pf-view').innerHTML = `
            <div class="pf-grid pf-grid-2">
                <section class="pf-card"><div class="pf-card-head"><div><h2>Workspace</h2><div class="pf-card-sub">Read-only inspection; approved roots only.</div></div><button class="pf-btn pf-btn-sm" data-action="index-code">Index code</button></div>
                    <form id="files-root-form" class="pf-form"><div class="pf-form-row"><div class="pf-field"><label for="f-root">Root directory</label><input class="pf-input" id="f-root" placeholder="Default workspace"></div><button class="pf-btn pf-btn-primary" type="submit">Open</button></div></form>
                    <div id="file-tree" class="pf-tree" style="margin-top:12px">${empty('Open a workspace to inspect files.')}</div>
                </section>
                <section class="pf-card"><div class="pf-card-head"><h3>File preview</h3><span id="file-meta" class="pf-dim pf-small"></span></div><pre id="file-output" class="pf-code">Select a file.</pre>
                    <div class="pf-card-head" style="margin-top:14px"><h3>Code search</h3></div><form id="code-search-form" class="pf-form"><div class="pf-form-row"><input class="pf-input" id="c-query" required placeholder="symbol, text, filename…"><button class="pf-btn pf-btn-primary" type="submit">Search</button></div></form><div id="code-results" class="pf-list" style="margin-top:10px"></div>
                </section>
            </div>`;
        loadFileTree();
    }

    async function loadFileTree(root = '') {
        const query = root ? `?root=${encodeURIComponent(root)}` : '';
        const data = await api(`/files/tree${query}`);
        const entries = data.entries || [];
        const node = $('#file-tree');
        if (!node) return;
        node.innerHTML = entries.length ? entries.map((e) => `<div class="pf-tree-item ${e.is_dir ? 'dir' : ''}" ${e.is_dir ? '' : `data-action="file-read" data-path="${attr(e.path)}" data-root="${attr(data.root)}"`}>${e.is_dir ? '▸' : '·'} ${esc(e.path)}</div>`).join('') : empty('Directory is empty.');
        node.dataset.root = data.root;
    }

    // --------------------------------------------------------------------- audit
    async function renderAudit() {
        const [verify, events] = await Promise.all([api('/audit/verify'), api('/audit?limit=100')]);
        $('#pf-view').innerHTML = `
            <div class="pf-grid pf-grid-3">
                <div class="pf-card"><div class="pf-stat-label">Chain integrity</div><div class="pf-stat-value ${verify.ok ? 'pf-ok-text' : 'pf-error-text'}">${verify.ok ? 'VERIFIED' : 'BROKEN'}</div><div class="pf-stat-note">${num(verify.events)} events · ${esc(short(verify.head, 18))}</div></div>
                <div class="pf-card"><div class="pf-stat-label">Audit events</div><div class="pf-stat-value">${num(events.length)}</div><div class="pf-stat-note">most recent 100</div></div>
                <div class="pf-card"><div class="pf-stat-label">Export</div><div class="pf-stat-value" style="font-size:16px;margin-top:12px">Machine-readable trail</div><button class="pf-btn pf-btn-sm" data-action="audit-export" style="margin-top:10px">Download JSON</button></div>
            </div>
            <section class="pf-card"><div class="pf-card-head"><h3>Event trail</h3><button class="pf-btn pf-btn-sm" data-action="refresh-view">Refresh</button></div><div class="pf-events" style="max-height:620px">${events.length ? events.map((e) => `
                <div class="pf-event" style="grid-template-columns:145px 170px minmax(0,1fr)"><span>${esc(fmt(e.created_at))}</span><span class="pf-event-type">${esc(e.category)}.${esc(e.action)}</span><span>${esc(short(e.summary || e.status || '', 220))}</span></div>`).join('') : empty('No audit events yet.')}</div></section>`;
    }

    // ------------------------------------------------------------------ settings
    async function renderSettings() {
        const [safety, backups, devices, meta] = await Promise.all([
            api('/safety'), api('/backups'), api('/mobile/devices'), api('/meta'),
        ]);
        $('#pf-view').innerHTML = `
            <div class="pf-grid pf-grid-2">
                <section class="pf-card"><div class="pf-card-head"><div><h2>Safety controls</h2><div class="pf-card-sub">The kill switch stops supervised runs and the automation scheduler.</div></div>${safety.kill_switch ? '<span class="pf-pill failed">engaged</span>' : '<span class="pf-pill ok">clear</span>'}</div>
                    <dl class="pf-kv"><dt>Kill switch</dt><dd>${safety.kill_switch ? 'Engaged' : 'Clear'}</dd><dt>Offline mode</dt><dd>${safety.offline_mode ? 'Enabled — no network sync' : 'Disabled'}</dd><dt>Redaction</dt><dd>${safety.redaction_enabled ? 'Enabled' : 'Disabled'}</dd><dt>App lock</dt><dd>${safety.lock_enabled ? 'Configured' : 'Not set'}</dd><dt>Vault secrets</dt><dd>${num(safety.secret_count)}</dd></dl>
                    <div class="pf-actions" style="margin-top:14px">${safety.kill_switch ? '<button class="pf-btn pf-btn-primary" data-action="safety-release">Release kill switch</button>' : '<button class="pf-btn pf-btn-danger" data-action="safety-kill">Engage kill switch</button>'}<button class="pf-btn" data-action="safety-offline">${safety.offline_mode ? 'Disable' : 'Enable'} offline mode</button></div>
                    <div class="pf-actions" style="margin-top:10px">${safety.lock_enabled ? '<button class="pf-btn" data-action="app-lock-set">Change PIN</button><button class="pf-btn pf-btn-danger" data-action="app-lock-remove">Remove app lock</button>' : '<button class="pf-btn" data-action="app-lock-set">Set app-lock PIN</button>'}</div>
                </section>
                <section class="pf-card"><div class="pf-card-head"><div><h2>Local backups</h2><div class="pf-card-sub">Consistent SQLite snapshots plus workspace files.</div></div><button class="pf-btn pf-btn-primary pf-btn-sm" data-action="backup-create">Create backup</button></div>
                    <div class="pf-list">${backups.length ? backups.map((b) => `<div class="pf-list-item"><div class="pf-card-head" style="margin:0"><span class="pf-list-title">${esc(b.label)}</span><span class="pf-pill ${b.encrypted ? 'ok' : ''}">${b.encrypted ? 'encrypted' : 'plain'}</span></div><div class="pf-list-meta">${esc(fmt(b.created_at))} · ${num(b.size_bytes)} bytes · <span class="pf-mono">${esc(b.id)}</span></div><div class="pf-actions" style="margin-top:7px"><button class="pf-btn pf-btn-danger pf-btn-sm" data-action="backup-delete" data-id="${attr(b.id)}">Delete</button></div></div>`).join('') : empty('No backups yet.')}</div>
                </section>
            </div>
            <div class="pf-grid pf-grid-2">
                <section class="pf-card"><div class="pf-card-head"><div><h2>Mobile companion</h2><div class="pf-card-sub">Pair by one-time code. Tokens can be revoked per device.</div></div></div>
                    <form id="mobile-pair-form" class="pf-form"><div class="pf-form-row"><div class="pf-field"><label for="m-name">Device name</label><input class="pf-input" id="m-name" value="My phone"></div><button class="pf-btn pf-btn-primary" type="submit">Create pairing</button></div></form>
                    <div id="mobile-pair-result" style="margin-top:12px"></div>
                    <div class="pf-card-head" style="margin-top:16px"><h3>Paired devices</h3></div><div class="pf-list">${devices.length ? devices.map((d) => `<div class="pf-list-item"><div class="pf-card-head" style="margin:0"><span class="pf-list-title">${esc(d.name)}</span>${pill(d.paired_at ? 'paired' : 'pending')}</div><div class="pf-list-meta">${esc(d.platform)} · last seen ${esc(fmt(d.last_seen))}</div><button class="pf-btn pf-btn-danger pf-btn-sm" data-action="mobile-revoke" data-id="${attr(d.id)}" style="margin-top:7px">Revoke</button></div>`).join('') : empty('No devices paired.')}</div>
                </section>
                <section class="pf-card"><div class="pf-card-head"><div><h2>Data &amp; privacy</h2><div class="pf-card-sub">Export metadata as JSON or clean old activity.</div></div></div>
                    <dl class="pf-kv"><dt>Database</dt><dd class="pf-mono">${esc(meta.database || '')}</dd><dt>API</dt><dd>${esc(meta.version)} · ${esc(meta.api_prefix)}</dd><dt>Approved roots</dt><dd>${(meta.allowed_roots || []).map((p) => `<span class="pf-mono">${esc(p)}</span>`).join('<br>')}</dd></dl>
                    <div class="pf-actions" style="margin-top:14px"><button class="pf-btn" data-action="privacy-export">Export JSON</button><button class="pf-btn" data-action="privacy-cleanup">Clean old activity</button></div>
                    <div class="pf-alert pf-alert-warn" style="margin-top:14px">Destructive wipe is available only through the API with an explicit typed confirmation. The Command Center intentionally does not expose it as a single click.</div>
                </section>
            </div>`;
    }

    // -------------------------------------------------------------------- modal
    function openModal(title, html) {
        $('#pf-modal-title').textContent = title;
        $('#pf-modal-body').innerHTML = html;
        $('#pf-modal').classList.remove('hidden');
    }

    function closeModal() {
        $('#pf-modal').classList.add('hidden');
        $('#pf-modal-body').innerHTML = '';
    }

    // ------------------------------------------------------------------- events
    async function handleAction(el) {
        const action = el.dataset.action;
        const id = el.dataset.id;
        if (action === 'go') { await navigate(el.dataset.view); return; }
        if (action === 'refresh-view') { await navigate(state.view, false); return; }
        if (action === 'open-task') {
            state.selectedTask = id;
            await navigate('tasks');
            if (id) await loadTaskDetail(id);
            return;
        }
        if (action === 'select-task') { await loadTaskDetail(id); return; }
        if (action === 'refresh-tasks') { await navigate('tasks', false); return; }
        if (action === 'task-start') { await api(`/tasks/${id}/start`, { method: 'POST', body: {} }); toast('Task started.', 'ok'); await loadTaskDetail(id); updateShell(); return; }
        if (action === 'task-pause') { await api(`/tasks/${id}/pause`, { method: 'POST' }); toast('Run paused.', 'ok'); await loadTaskDetail(id); return; }
        if (action === 'task-resume') { await api(`/tasks/${id}/resume`, { method: 'POST' }); toast('Run resumed.', 'ok'); await loadTaskDetail(id); return; }
        if (action === 'task-retry') { await api(`/tasks/${id}/retry`, { method: 'POST', body: {} }); toast('New run queued.', 'ok'); await loadTaskDetail(id); return; }
        if (action === 'task-cancel') { await api(`/tasks/${id}/cancel`, { method: 'POST' }); toast('Task cancelled.', 'ok'); await loadTaskDetail(id); updateShell(); return; }
        if (action === 'task-delete') {
            if (!confirm('Delete this task and its run history?')) return;
            await api(`/tasks/${id}`, { method: 'DELETE' });
            state.selectedTask = null; state.taskDetail = null;
            toast('Task deleted.', 'ok'); await navigate('tasks', false); return;
        }
        if (action === 'task-open-approvals') { await navigate('approvals'); return; }
        if (action === 'approve') { await resolveApproval(id, true); return; }
        if (action === 'deny') { await resolveApproval(id, false); return; }
        if (action === 'reindex-knowledge') { const r = await api('/knowledge/reindex', { method: 'POST' }); toast(`Reindexed ${r.sources || 0} sources.`, 'ok'); return; }
        if (action === 'select-research') { await loadResearch(id); return; }
        if (action === 'refresh-research') { await loadResearch(id); return; }
        if (action === 'research-task') { const r = await api(`/research/${id}/task`, { method: 'POST', body: { auto_start: true } }); toast(`Task ${r.id} created.`, 'ok'); return; }
        if (action === 'connector-connect') {
            const catalog = await api('/connectors');
            const meta = catalog.find((c) => c.id === id);
            if (meta?.auth === 'oauth') await startOAuth(id);
            else openConnectorModal(id, catalog);
            return;
        }
        if (action === 'connector-test') { const r = await api(`/connections/${id}/test`, { method: 'POST' }); toast(r.ok ? 'Connection OK.' : `Connection failed: ${r.error}`, r.ok ? 'ok' : 'error'); await renderConnectors(); return; }
        if (action === 'connector-sync') { const r = await api(`/connections/${id}/sync`, { method: 'POST', body: { limit: 25 } }); toast(`Synced ${r.saved || 0} new items.`, 'ok'); await renderConnectors(); return; }
        if (action === 'connector-disconnect') { if (!confirm('Disconnect and delete cached items?')) return; await api(`/connections/${id}`, { method: 'DELETE' }); toast('Disconnected.', 'ok'); await renderConnectors(); return; }
        if (action === 'automation-run') { const r = await api(`/automations/${id}/run`, { method: 'POST' }); toast(`Automation ${r.status}: ${r.detail || ''}`, r.status === 'completed' ? 'ok' : 'error'); await renderAutomations(); return; }
        if (action === 'automation-toggle') { await api(`/automations/${id}/toggle`, { method: 'POST' }); await renderAutomations(); return; }
        if (action === 'automation-delete') { if (!confirm('Delete this automation?')) return; await api(`/automations/${id}`, { method: 'DELETE' }); await renderAutomations(); return; }
        if (action === 'file-read') {
            const path = el.dataset.path; const root = el.dataset.root;
            const data = await api(`/files/read?path=${encodeURIComponent(path)}&root=${encodeURIComponent(root)}`);
            $('#file-output').textContent = data.content;
            $('#file-meta').textContent = `${data.path} · ${num(data.size)} bytes${data.binary ? ' · binary' : ''}`;
            return;
        }
        if (action === 'index-code') {
            const root = $('#file-tree')?.dataset.root || '';
            const r = await api('/code/index', { method: 'POST', body: root ? { root } : {} });
            toast(`Indexed ${r.files || 0} files.`, 'ok'); return;
        }
        if (action === 'audit-export') {
            const response = await fetch('/api/v2/audit/export');
            if (!response.ok) throw new Error(`Export failed: HTTP ${response.status}`);
            const blob = await response.blob();
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = `grok-audit-${new Date().toISOString().slice(0, 10)}.json`;
            link.click(); URL.revokeObjectURL(link.href); return;
        }
        if (action === 'backup-create') { await api('/backups', { method: 'POST', body: {} }); toast('Backup created.', 'ok'); await renderSettings(); return; }
        if (action === 'backup-delete') { if (!confirm('Delete this backup file?')) return; await api(`/backups/${id}`, { method: 'DELETE' }); await renderSettings(); return; }
        if (action === 'safety-kill') { if (!confirm('Engage the kill switch and stop all supervised runs?')) return; await api('/safety/kill', { method: 'POST', body: { reason: 'Command Center emergency stop' } }); toast('Kill switch engaged.', 'ok'); await renderSettings(); updateShell(); return; }
        if (action === 'safety-release') { await api('/safety/release', { method: 'POST' }); toast('Kill switch released.', 'ok'); await renderSettings(); updateShell(); return; }
        if (action === 'safety-offline') { const s = await api('/safety'); await api('/safety/offline', { method: 'POST', body: { enabled: !s.offline_mode } }); await renderSettings(); updateShell(); return; }
        if (action === 'app-lock-set') {
            openModal('App lock', `
                <p class="pf-muted">A local PIN protects the chat and Command Center on this machine. Mobile devices keep using their own pairing tokens.</p>
                <form id="app-lock-form" class="pf-form">
                    <div class="pf-form-row"><div class="pf-field"><label for="lock-pin">New PIN</label><input class="pf-input" id="lock-pin" type="password" inputmode="numeric" autocomplete="new-password" minlength="4" required></div></div>
                    <div class="pf-form-row"><div class="pf-field"><label for="lock-confirm">Confirm PIN</label><input class="pf-input" id="lock-confirm" type="password" inputmode="numeric" autocomplete="new-password" minlength="4" required></div></div>
                    <div class="pf-actions"><button class="pf-btn pf-btn-primary" type="submit">Save PIN</button></div>
                </form>`);
            return;
        }
        if (action === 'app-lock-remove') {
            if (!confirm('Remove the app lock? The chat will open without a PIN.')) return;
            await api('/safety/lock', { method: 'POST', body: { pin: null } });
            toast('App lock removed.', 'ok');
            await renderSettings();
            updateShell();
            return;
        }
        if (action === 'mobile-revoke') { await api(`/mobile/devices/${id}`, { method: 'DELETE' }); toast('Device revoked.', 'ok'); await renderSettings(); return; }
        if (action === 'privacy-export') { const r = await api('/privacy/export', { method: 'POST', body: {} }); toast(`Export written: ${r.path || 'complete'}`, 'ok'); return; }
        if (action === 'privacy-cleanup') { if (!confirm('Delete activity older than 180 days and backups older than 30 days?')) return; const r = await api('/privacy/cleanup', { method: 'POST', body: {} }); toast(`Cleaned ${JSON.stringify(r)}`, 'ok'); return; }
        if (action === 'copy') { await navigator.clipboard.writeText(el.dataset.text || ''); toast('Copied.', 'ok'); return; }
        console.debug('Unhandled action', action);
    }

    document.addEventListener('click', async (event) => {
        const nav = event.target.closest('[data-view].pf-nav-item');
        if (nav) { event.preventDefault(); await navigate(nav.dataset.view); return; }
        const target = event.target.closest('[data-action]');
        if (!target) return;
        event.preventDefault();
        const original = target.textContent;
        target.disabled = true;
        try {
            await handleAction(target);
        } catch (err) {
            showError(err);
            toast(err.message, 'error');
        } finally {
            target.disabled = false;
            if (!target.isConnected) return;
            if (target.dataset.action === 'refresh-view' || target.dataset.action === 'refresh-tasks') target.textContent = original;
        }
    });

    document.addEventListener('submit', async (event) => {
        const form = event.target;
        if (!form.id) return;
        event.preventDefault();
        try {
            if (form.id === 'quick-task-form') {
                const goal = $('#q-goal').value.trim();
                if (!goal) return;
                const task = await api('/tasks', { method: 'POST', body: {
                    title: goal.split('\n')[0].slice(0, 120), goal,
                    agent_id: $('#q-agent').value, auto_start: $('#q-start').checked,
                    budget: { max_steps: Number($('#q-steps').value || 24) },
                } });
                toast(`Task ${task.id} created.`, 'ok');
                state.selectedTask = task.id;
                await navigate('tasks');
                await loadTaskDetail(task.id);
                return;
            }
            if (form.id === 'app-lock-form') {
                const pin = $('#lock-pin').value;
                const confirmPin = $('#lock-confirm').value;
                if (pin.length < 4) throw new Error('PIN must be at least 4 characters.');
                if (pin !== confirmPin) throw new Error('PINs do not match.');
                await api('/safety/lock', { method: 'POST', body: { pin } });
                closeModal();
                toast('App lock enabled.', 'ok');
                await renderSettings();
                updateShell();
                return;
            }
            if (form.id === 'task-create-form') {
                const goal = $('#t-goal').value.trim();
                if (!goal) return;
                const task = await api('/tasks', { method: 'POST', body: {
                    title: $('#t-title').value.trim() || goal.split('\n')[0].slice(0, 120),
                    goal, agent_id: $('#t-agent').value, auto_start: $('#t-start').checked,
                    budget: { max_steps: Number($('#t-steps').value || 24) },
                } });
                state.selectedTask = task.id;
                toast('Task created.', 'ok');
                await navigate('tasks', false);
                await loadTaskDetail(task.id);
                return;
            }
            if (form.id === 'knowledge-ask-form') {
                const question = $('#k-question').value.trim();
                if (!question) return;
                await knowledgeSearch(question, $('#k-synthesize').checked);
                return;
            }
            if (form.id === 'knowledge-add-form') {
                await api('/knowledge/sources', { method: 'POST', body: {
                    title: $('#k-title').value.trim(), content: $('#k-content').value,
                    tags: $('#k-tags').value.split(',').map((s) => s.trim()).filter(Boolean),
                    profile: $('#k-profile').value.trim() || 'personal', source_type: 'note',
                } });
                toast('Saved to vault.', 'ok');
                await renderKnowledge();
                return;
            }
            if (form.id === 'research-create-form') {
                const question = $('#r-question').value.trim();
                if (!question) return;
                const project = await api('/research', { method: 'POST', body: {
                    question, max_sources: Number($('#r-max').value || 8),
                    profile: $('#r-profile').value, auto_start: $('#r-start').checked,
                } });
                toast('Research started.', 'ok');
                await renderResearch();
                await loadResearch(project.id);
                return;
            }
            if (form.id === 'automation-create-form') {
                const actionType = $('#a-action').value;
                const actionConfig = actionType === 'create_task'
                    ? { goal: $('#a-goal').value.trim() || $('#a-name').value.trim(), auto_start: true }
                    : { title: $('#a-name').value.trim(), body: $('#a-goal').value.trim() || 'Automation fired.' };
                const triggerType = $('#a-trigger').value;
                await api('/automations', { method: 'POST', body: {
                    name: $('#a-name').value.trim(), description: $('#a-goal').value.trim(),
                    trigger_type: triggerType,
                    trigger_config: triggerType === 'schedule'
                        ? { every_minutes: Number($('#a-every').value || 1440) } : {},
                    action_type: actionType, action_config: actionConfig,
                    agent_id: $('#a-agent')?.value || 'orchestrator', enabled: true,
                } });
                toast('Automation created.', 'ok');
                await renderAutomations();
                return;
            }
            if (form.id === 'files-root-form') {
                await loadFileTree($('#f-root').value.trim());
                return;
            }
            if (form.id === 'code-search-form') {
                const q = $('#c-query').value.trim();
                if (!q) return;
                const root = $('#file-tree')?.dataset.root || '';
                const results = await api(`/code/search?q=${encodeURIComponent(q)}${root ? `&root=${encodeURIComponent(root)}` : ''}`);
                $('#code-results').innerHTML = results.length ? results.map((r) => `<div class="pf-list-item"><div class="pf-list-title">${esc(r.path)}:${esc(r.line || 0)}</div><div class="pf-list-body">${esc(short(r.preview || r.snippet || '', 200))}</div></div>`).join('') : empty('No matches.');
                return;
            }
            if (form.id === 'mobile-pair-form') {
                const pairing = await api('/mobile/pairing', { method: 'POST', body: { name: $('#m-name').value.trim() || 'My phone', platform: 'mobile' } });
                const pairingCode = pairing.pairing_code || pairing.code || '';
                const link = `${location.origin}/mobile?device=${encodeURIComponent(pairing.device_id)}&code=${encodeURIComponent(pairingCode)}`;
                $('#mobile-pair-result').innerHTML = `<div class="pf-alert pf-alert-ok"><strong>Pairing code: ${esc(pairingCode)}</strong><br>Device: <span class="pf-mono">${esc(pairing.device_id)}</span><br>Open on your phone:<br><a href="${attr(link)}" target="_blank" rel="noreferrer">${esc(link)}</a></div>`;
                return;
            }
        } catch (err) {
            showError(err);
            toast(err.message, 'error');
        }
    });

    // --------------------------------------------------------------------- boot
    document.addEventListener('DOMContentLoaded', async () => {
        $('#pf-refresh').addEventListener('click', () => navigate(state.view, false));
        $('#pf-stop-all').addEventListener('click', async () => {
            if (!confirm('Engage the emergency kill switch and stop all supervised runs?')) return;
            try {
                await api('/safety/kill', { method: 'POST', body: { reason: 'Stop all runs from Command Center' } });
                toast('Kill switch engaged.', 'ok');
                updateShell();
            } catch (err) { showError(err); }
        });
        $('#pf-modal-close').addEventListener('click', closeModal);
        $('#pf-modal').addEventListener('click', (event) => {
            if (event.target.id === 'pf-modal') closeModal();
        });
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') closeModal();
        });
        window.addEventListener('focus', () => {
            if (state.view === 'connectors') renderConnectors().catch(() => {});
        });
        const initial = (location.hash || '#overview').slice(1);
        await navigate(initial, false);
        setInterval(updateShell, 15000);
        updateShell();
    });
})();
