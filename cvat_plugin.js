// ==UserScript==
// @name         CVAT SAM3 AI Tracker
// @namespace    http://tampermonkey.net/
// @version      2.9
// @description  Unified Object ID, Auto-Select, Auto-Save, Range Tools (Propagate/Delete/Change Label)
// @author       Zhixin MA
// @include      *
// @grant        none
// ==/UserScript==

(function() {
    'use strict';

    if (!window.location.hostname.includes('10.4.4.24') && !window.location.hostname.includes('localhost')) return;

    const API_BASE = 'http://10.4.4.24:8081/api';
    let isLabelsSynced = false;

    setInterval(() => {
        try {
            const isJobPage = window.location.href.includes('/jobs/');
            const panelExists = document.getElementById('sam3-plugin-panel');

            if (isJobPage && !panelExists) {
                initPlugin();
                isLabelsSynced = false;
            } else if (!isJobPage && panelExists) {
                panelExists.remove();
            }

            if (panelExists && !isLabelsSynced) {
                const jobId = getJobId();
                if (jobId) {
                    isLabelsSynced = true;
                    autoSyncLabels(jobId);
                }
            }
        } catch (error) {
            console.error("SAM3 Plugin Error:", error);
        }
    }, 1500);

    function initPlugin() {
        const panel = document.createElement('div');
        panel.id = 'sam3-plugin-panel';
        panel.style.cssText = `
            position: fixed; top: 100px; right: 30px; width: 340px;
            background: #ffffff; border: 1px solid #ddd; border-radius: 8px;
            z-index: 999999; font-family: sans-serif;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15); font-size: 13px; color: #333;
            overflow: hidden;
        `;

        const inputStyle = "background: #f8f9fa; border: 1px solid #ced4da; border-radius: 4px; padding: 5px; width: 50px;";
        const selectStyle = "background: #e8f5e9; border: 1px solid #4caf50; border-radius: 4px; padding: 5px; color: #2e7d32; font-weight: bold; width: 100%; box-sizing: border-box;";
        const btnStyle = "background: #e3f2fd; border: 1px solid #2196f3; border-radius: 4px; padding: 6px 12px; color: #1565c0; cursor: pointer; font-weight: bold; transition: 0.2s;";
        const refreshIcon = "🔄";

        panel.innerHTML = `
            <div id="sam-drag-bar" style="background: #f1f3f4; padding: 8px 15px; cursor: grab; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; align-items: center; user-select: none;">
                <strong style="color: #5f6368; font-size: 14px;">🤖 SAM3 Tracker</strong>
                <span style="color: #9aa0a6; font-size: 12px;">☰ Drag</span>
            </div>
            <div style="padding: 15px;">
                <div style="border-bottom: 2px solid #eee; padding-bottom: 12px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <div><strong>Job ID:</strong> <span id="sam-job-id" style="font-size: 16px; color: #d32f2f; font-weight:bold; margin-right: 5px;">Detecting...</span></div>
                        <button id="sam-btn-sync" style="${btnStyle} font-size: 12px; padding: 4px 8px;">Sync Labels</button>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: #1565c0;">Target Object:</strong>
                        <select id="sam-global-obj-id" style="${selectStyle} width: 65%;"><option value="">(Auto-Syncing...)</option></select>
                    </div>
                </div>

                <div style="border-bottom: 1px dashed #ccc; padding-bottom: 12px; margin-bottom: 12px;">
                    <button id="sam-btn-next" style="${btnStyle} width: 100%;">Propagate to next frame</button>
                </div>

                <div style="border-bottom: 1px dashed #ccc; padding-bottom: 12px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong>Query:</strong> <input type="text" id="sam-query" style="background: #fff; border: 1px solid #ccc; border-radius: 4px; padding: 5px; width: 140px;" placeholder="e.g. white hat">
                        <button id="sam-btn-search" style="${btnStyle}">Search</button>
                    </div>
                    <div style="font-size: 11px; color: #888; margin-top: 6px; text-align: right;">*Applies to current frame</div>
                </div>

                <div>
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                        <span>Start:</span> <input type="number" id="sam-start" style="${inputStyle}">
                        <button id="sam-btn-set-start" style="cursor:pointer; background:none; border:none;">${refreshIcon}</button>
                        <span style="margin-left:5px;">End:</span> <input type="number" id="sam-end" style="${inputStyle}">
                        <button id="sam-btn-set-end" style="cursor:pointer; background:none; border:none;">${refreshIcon}</button>
                    </div>

                    <div style="display: flex; gap: 8px; margin-bottom: 8px;">
                        <button id="sam-btn-range" style="${btnStyle} flex: 2;">Propagate Range</button>
                        <button id="sam-btn-delete-range" style="${btnStyle} flex: 1; background: #ffebee; color: #c62828; border-color: #ef9a9a;" title="Delete all annotations for Target Object in range">Delete</button>
                    </div>

                    <div style="display: flex; gap: 8px; align-items: center; border-top: 1px dashed #eee; padding-top: 8px;">
                        <span style="font-weight: bold; color: #666; font-size: 12px;">To:</span>
                        <select id="sam-target-obj-id" style="${selectStyle} background: #fff8e1; border-color: #ffca28; color: #e65100; flex: 2;"><option value="">Select label...</option></select>
                        <button id="sam-btn-change-label" style="${btnStyle} flex: 1; background: #fff3e0; color: #e65100; border-color: #ffcc80;" title="Change Target Object to this new label">Change</button>
                    </div>
                </div>

                <div id="sam-status" style="margin-top: 15px; font-size: 12px; font-weight: bold; color: #666; text-align: center; background: #f8f9fa; padding: 6px; border-radius: 4px; border: 1px solid #e9ecef;">Ready.</div>
            </div>
        `;

        document.body.appendChild(panel);
        makeDraggable(panel, document.getElementById('sam-drag-bar'));
        bindEvents();
    }

    function makeDraggable(elmnt, dragHeader) {
        let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
        dragHeader.onmousedown = (e) => {
            e.preventDefault(); pos3 = e.clientX; pos4 = e.clientY;
            document.onmouseup = () => { document.onmouseup = null; document.onmousemove = null; dragHeader.style.cursor = "grab"; };
            document.onmousemove = (e) => {
                e.preventDefault(); pos1 = pos3 - e.clientX; pos2 = pos4 - e.clientY; pos3 = e.clientX; pos4 = e.clientY;
                elmnt.style.top = (elmnt.offsetTop - pos2) + "px"; elmnt.style.left = (elmnt.offsetLeft - pos1) + "px";
            };
            dragHeader.style.cursor = "grabbing";
        };
    }

    function setStatus(msg, color="#666") { const el = document.getElementById('sam-status'); if(el) { el.innerText = msg; el.style.color = color; } }
    function getJobId() { const match = window.location.href.match(/jobs\/(\d+)/); return match ? parseInt(match[1], 10) : null; }
    function getCurrentFrame() { const input = document.querySelector('.cvat-player-frame-selector input'); return input && !isNaN(parseInt(input.value)) ? parseInt(input.value, 10) : 0; }

    async function saveCvatAnnotations() {
        setStatus("💾 Saving annotations to CVAT...", "#f57c00");
        try {
            const saveBtn = document.querySelector('.cvat-header-save-button') ||
                            Array.from(document.querySelectorAll('span')).find(el => el.innerText.trim() === 'Save')?.closest('button') ||
                            document.querySelector('i[aria-label="save"]')?.closest('button');

            if (saveBtn) {
                if (!saveBtn.disabled && !saveBtn.classList.contains('ant-btn-disabled') && !saveBtn.classList.contains('cvat-header-button-disabled')) {
                    saveBtn.click();
                }
            } else {
                document.dispatchEvent(new KeyboardEvent('keydown', { key: 's', code: 'KeyS', ctrlKey: true, bubbles: true }));
            }
        } catch(e) { console.warn("[SAM3] Auto-save trigger failed", e); }
        await new Promise(resolve => setTimeout(resolve, 1500));
    }

    function forceReloadAtFrame(frame) {
        const currentObj = document.getElementById('sam-global-obj-id').value;
        if(currentObj) localStorage.setItem('sam3_saved_obj', currentObj);

        setStatus(`🔄 Reloading CVAT database at frame ${frame}...`, "#f57c00");
        const url = new URL(window.location.href);
        url.searchParams.set('frame', frame);
        window.location.href = url.toString();
    }

    async function autoSyncLabels(jobId) {
        setStatus("🔄 Auto-fetching labels...", "#1565c0");
        try {
            const res = await fetch(`${API_BASE}/get_labels`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({job_id: jobId}) });
            const data = await res.json();

            if(data.status === "success" && data.labels.length > 0) {
                let optionsHTML = '';
                data.labels.forEach(lbl => { optionsHTML += `<option value="${lbl.id}" data-name="${lbl.name.toLowerCase()}">${lbl.name} (ID: ${lbl.id})</option>`; });

                const selectGlobal = document.getElementById('sam-global-obj-id');
                const selectTarget = document.getElementById('sam-target-obj-id'); // 新增的目标下拉框

                selectGlobal.innerHTML = optionsHTML;
                selectTarget.innerHTML = '<option value="">Select label...</option>' + optionsHTML; // 默认空选项

                const savedObj = localStorage.getItem('sam3_saved_obj');
                if (savedObj) {
                    selectGlobal.value = savedObj;
                } else {
                    localStorage.setItem('sam3_saved_obj', selectGlobal.value);
                }

                selectGlobal.addEventListener('change', (e) => {
                    localStorage.setItem('sam3_saved_obj', e.target.value);
                });

                setStatus(`✅ Loaded ${data.labels.length} object classes.`, "#2e7d32");
            }
        } catch(e) {}
    }

    function watchActiveLabel() {
        try {
            const selectGlobal = document.getElementById('sam-global-obj-id');
            if (!selectGlobal || selectGlobal.options.length === 0) return;

            let activeText = "";
            let isExplicitlyActive = false;

            const activeSidebarItem = document.querySelector('.cvat-objects-sidebar-state-item-active') || document.querySelector('.cvat-object-item-active');
            if (activeSidebarItem) {
                const labelEl = activeSidebarItem.querySelector('.ant-select-selection-item') || activeSidebarItem.querySelector('.cvat-object-item-label-selector');
                if (labelEl && labelEl.innerText) {
                    activeText = labelEl.innerText.trim().toLowerCase();
                    isExplicitlyActive = true;
                }
            }

            if (!activeText) {
                const anySidebarItem = document.querySelector('.cvat-objects-sidebar-state-item') || document.querySelector('.cvat-object-item');
                if (anySidebarItem) {
                    const labelEl = anySidebarItem.querySelector('.ant-select-selection-item') || anySidebarItem.querySelector('.cvat-object-item-label-selector');
                    if (labelEl && labelEl.innerText) {
                        activeText = labelEl.innerText.trim().toLowerCase();
                    }
                }
            }

            if (activeText) {
                for (let i = 0; i < selectGlobal.options.length; i++) {
                    const optName = selectGlobal.options[i].getAttribute('data-name');
                    if (optName && (activeText.includes(optName) || optName.includes(activeText))) {
                        if (selectGlobal.value !== selectGlobal.options[i].value) {
                            if (isExplicitlyActive || !localStorage.getItem('sam3_saved_obj')) {
                                selectGlobal.value = selectGlobal.options[i].value;
                                localStorage.setItem('sam3_saved_obj', selectGlobal.value);
                            }
                        }
                        break;
                    }
                }
            }
        } catch (e) {}
    }

    function bindEvents() {
        setInterval(() => {
            const jobId = getJobId();
            const el = document.getElementById('sam-job-id');
            if(el) el.innerText = jobId ? jobId : "No Job";
            watchActiveLabel();
        }, 800);

        document.getElementById('sam-btn-sync').onclick = () => { const j = getJobId(); if(j) autoSyncLabels(j); };

        document.getElementById('sam-btn-set-start').onclick = () => document.getElementById('sam-start').value = getCurrentFrame();
        document.getElementById('sam-btn-set-end').onclick = () => document.getElementById('sam-end').value = getCurrentFrame();

        document.getElementById('sam-btn-next').onclick = async () => {
            const j = getJobId(), f = getCurrentFrame(), o = document.getElementById('sam-global-obj-id').value;
            if(!o) return alert("Select an object first.");
            await saveCvatAnnotations();
            setStatus("🚀 Propagating to next frame...", "#1565c0");
            try {
                const res = await fetch(`${API_BASE}/propagate_next`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({job_id: j, mask_frame_index: f, object_id: parseInt(o)}) });
                if(res.ok) forceReloadAtFrame(f + 1);
                else throw new Error();
            } catch(e) { setStatus("❌ Connection Error", "#d32f2f"); }
        };

        document.getElementById('sam-btn-search').onclick = async () => {
            const j = getJobId(), q = document.getElementById('sam-query').value, o = document.getElementById('sam-global-obj-id').value;
            const f = getCurrentFrame();

            if(!o || !q) return alert("Missing text prompt or object.");
            await saveCvatAnnotations();

            setStatus(`🔍 Searching "${q}" on Frame ${f}...`, "#1565c0");
            try {
                const res = await fetch(`${API_BASE}/segment_text`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({job_id: j, mask_frame_index: f, text_prompt: q, object_id: parseInt(o)}) });
                if(res.ok) forceReloadAtFrame(f);
                else throw new Error();
            } catch(e) { setStatus("❌ Connection Error", "#d32f2f"); }
        };

        document.getElementById('sam-btn-range').onclick = async () => {
            const j = getJobId(), s = document.getElementById('sam-start').value, e = document.getElementById('sam-end').value, o = document.getElementById('sam-global-obj-id').value;
            if(!o || !s || !e) return alert("Invalid Start/End frames.");
            await saveCvatAnnotations();
            setStatus(`🎬 Propagating ${s} to ${e}...`, "#1565c0");
            try {
                const res = await fetch(`${API_BASE}/propagate_range`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({job_id: j, start_frame: parseInt(s), end_frame: parseInt(e), object_id: parseInt(o)}) });
                if(res.ok) forceReloadAtFrame(e);
                else throw new Error();
            } catch(err) { setStatus("❌ Connection Error", "#d32f2f"); }
        };

        document.getElementById('sam-btn-delete-range').onclick = async () => {
            const j = getJobId(), s = document.getElementById('sam-start').value, e = document.getElementById('sam-end').value, o = document.getElementById('sam-global-obj-id').value;
            if(!o || !s || !e) return alert("Invalid Start/End frames.");

            if(!confirm(`⚠️ WARNING!\n\nAre you sure you want to DELETE ALL shapes for this Target Object from frame ${s} to ${e}?`)) return;

            await saveCvatAnnotations();

            setStatus(`🗑️ Deleting frames ${s} to ${e}...`, "#c62828");
            try {
                const res = await fetch(`${API_BASE}/delete_range`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({job_id: j, start_frame: parseInt(s), end_frame: parseInt(e), object_id: parseInt(o)}) });
                if(res.ok) forceReloadAtFrame(s);
                else throw new Error();
            } catch(err) { setStatus("❌ Connection Error", "#d32f2f"); }
        };

        document.getElementById('sam-btn-change-label').onclick = async () => {
            const j = getJobId(), s = document.getElementById('sam-start').value, e = document.getElementById('sam-end').value;
            const o_src = document.getElementById('sam-global-obj-id').value;
            const o_tgt = document.getElementById('sam-target-obj-id').value;

            if(!o_src || !o_tgt) return alert("Please select both Target Object and a new Label.");
            if(o_src === o_tgt) return alert("Target Object and new Label cannot be the same.");
            if(!s || !e) return alert("Invalid Start/End frames.");

            if(!confirm(`⚠️ Confirm Label Change:\n\nChange all annotations from current Target Object to the new label between frames ${s} and ${e}?`)) return;

            await saveCvatAnnotations();

            setStatus(`🔄 Changing labels ${s} to ${e}...`, "#e65100");
            try {
                const res = await fetch(`${API_BASE}/change_label_range`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({job_id: j, start_frame: parseInt(s), end_frame: parseInt(e), object_src: parseInt(o_src), object_tgt: parseInt(o_tgt)}) });
                if(res.ok) forceReloadAtFrame(s);
                else throw new Error();
            } catch(err) { setStatus("❌ Connection Error", "#d32f2f"); }
        };
    }
})();