document.addEventListener('DOMContentLoaded', () => {

    // ---------------------------------------------------------------------------
    // Constants
    // ---------------------------------------------------------------------------

    // Auto-detect environment: localhost = local dev, anything else = Vercel (production)
    const URL_BASE = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
        ? 'http://127.0.0.1:8000'
        : 'https://scythe410-dsgp-007.hf.space';

    // ---------------------------------------------------------------------------
    // Vehicle Make/Model Dropdown
    // ---------------------------------------------------------------------------

    const makeInput    = document.getElementById('make');
    const modelInput   = document.getElementById('model');
    const makeList     = document.getElementById('make-list');
    const modelList    = document.getElementById('model-list');
    const modelWarning = document.getElementById('model-warning');
    let vehicleMapping = {};

    if (makeInput && modelInput) {
        fetch(`${URL_BASE}/vehicle_options`)
            .then(r => r.json())
            .then(data => {
                if (data.status !== 'success') return;
                vehicleMapping = data.options;
                Object.keys(vehicleMapping).forEach(make => {
                    const opt = document.createElement('option');
                    opt.value = make;
                    makeList.appendChild(opt);
                });
            })
            .catch(err => console.error('[vehicle_options]', err));

        makeInput.addEventListener('input', () => {
            modelInput.value = '';
            modelList.innerHTML = '';
            const models = vehicleMapping[makeInput.value];
            if (models) {
                models.forEach(model => {
                    const opt = document.createElement('option');
                    opt.value = model;
                    modelList.appendChild(opt);
                });
                modelWarning.style.display = 'none';
                modelInput.disabled = false;
            } else {
                modelInput.disabled = true;
                modelWarning.style.display = 'none';
            }
        });

        document.getElementById('model-wrapper').addEventListener('click', () => {
            if (modelInput.disabled) modelWarning.style.display = 'block';
        });
    }

    // ---------------------------------------------------------------------------
    // Image Upload & Preview
    // ---------------------------------------------------------------------------

    const uploadBox        = document.getElementById('multi-upload-box');
    const fileInput        = document.getElementById('car-images');
    const previewContainer = document.getElementById('image-preview-container');
    let queuedFiles        = [];

    uploadBox.addEventListener('click',     (e) => { if (e.target !== fileInput) fileInput.click(); });
    uploadBox.addEventListener('dragover',  (e) => { e.preventDefault(); uploadBox.classList.add('dragover'); });
    uploadBox.addEventListener('dragleave', ()  => uploadBox.classList.remove('dragover'));
    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
    fileInput.addEventListener('change', () => handleFiles(fileInput.files));

    function handleFiles(files) {
        Array.from(files)
            .filter(f => f.type.startsWith('image/'))
            .forEach(file => {
                queuedFiles.push(file);
                const reader = new FileReader();
                reader.onload = (e) => {
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.className = 'preview-img';
                    previewContainer.appendChild(img);
                };
                reader.readAsDataURL(file);
            });

        if (queuedFiles.length > 0) {
            const label = uploadBox.querySelector('p');
            label.innerText = `${queuedFiles.length} photo(s) queued for multi-angle evaluation`;
            label.style.color = 'var(--primary-blue)';
        }
    }

    // ---------------------------------------------------------------------------
    // UI Helpers
    // ---------------------------------------------------------------------------

    const evalBtn = document.getElementById('evaluate-btn');
    const spinner = document.getElementById('spinner');

    function resetEvalButton() {
        evalBtn.disabled = false;
        evalBtn.innerHTML = `Evaluate Vehicle <span id="spinner" style="display:none; margin-left:10px;"><ion-icon name="sync-outline" class="spin"></ion-icon></span>`;
    }

    function hideModal() {
        document.getElementById('invalid-image-modal').style.display = 'none';
    }

    /** Shows a styled amber warning banner when vehicle specs are statistically unusual. */
    function showAnomalyWarning(reason) {
        if (document.getElementById('anomaly-banner')) return;
        const banner = document.createElement('div');
        banner.id = 'anomaly-banner';
        banner.style.cssText = [
            'position:fixed', 'top:20px', 'left:50%', 'transform:translateX(-50%)',
            'z-index:9999', 'display:flex', 'align-items:center', 'gap:12px',
            'background:rgba(255,100,0,0.12)', 'backdrop-filter:blur(12px)',
            'border:1.5px solid rgba(255,100,0,0.45)', 'border-radius:14px',
            'padding:14px 20px', 'max-width:500px', 'width:90%',
            'box-shadow:0 8px 32px rgba(255,80,0,0.15)',
            'animation:slideDown 0.35s ease',
        ].join(';');
        banner.innerHTML = `
            <ion-icon name="alert-circle-outline" style="font-size:22px;color:#F97316;flex-shrink:0;"></ion-icon>
            <div style="flex:1;">
                <p style="margin:0;font-weight:700;font-size:14px;color:#7C2D12;">Unusual Vehicle Specs Detected</p>
                <p style="margin:4px 0 0;font-size:12px;color:#92400E;line-height:1.4;">${reason} The price estimate may be less accurate — please double-check your inputs.</p>
            </div>
            <button onclick="this.parentElement.remove()" style="background:none;border:none;cursor:pointer;color:#92400E;font-size:18px;line-height:1;flex-shrink:0;">&#x2715;</button>
        `;
        document.body.appendChild(banner);
        setTimeout(() => banner.remove(), 15000);
    }

    /** Shows a styled in-page warning banner when Gemini quota is exceeded. */
    function showGeminiWarning() {
        // Avoid duplicates
        if (document.getElementById('gemini-quota-banner')) return;

        const banner = document.createElement('div');
        banner.id = 'gemini-quota-banner';
        banner.style.cssText = [
            'position:fixed', 'top:20px', 'left:50%', 'transform:translateX(-50%)',
            'z-index:9999', 'display:flex', 'align-items:center', 'gap:12px',
            'background:rgba(255,180,0,0.15)', 'backdrop-filter:blur(12px)',
            'border:1.5px solid rgba(255,180,0,0.5)', 'border-radius:14px',
            'padding:14px 20px', 'max-width:480px', 'width:90%',
            'box-shadow:0 8px 32px rgba(255,160,0,0.2)',
            'animation:slideDown 0.35s ease',
        ].join(';');

        banner.innerHTML = `
            <ion-icon name="warning-outline" style="font-size:22px;color:#F59E0B;flex-shrink:0;"></ion-icon>
            <div style="flex:1;">
                <p style="margin:0;font-weight:700;font-size:14px;color:#92400E;">Gemini API Quota Exceeded</p>
                <p style="margin:4px 0 0;font-size:12px;color:#78350F;line-height:1.4;">Image validation &amp; AI descriptions are currently unavailable. Non-vehicle images may pass through. Please update the API key in <code style="background:rgba(0,0,0,0.08);padding:1px 5px;border-radius:4px;">.env</code> and restart the server.</p>
            </div>
            <button onclick="this.parentElement.remove()" style="background:none;border:none;cursor:pointer;color:#92400E;font-size:18px;line-height:1;flex-shrink:0;">&#x2715;</button>
        `;

        // Inject keyframe if not already present
        if (!document.getElementById('gemini-banner-style')) {
            const style = document.createElement('style');
            style.id = 'gemini-banner-style';
            style.textContent = '@keyframes slideDown { from { opacity:0; transform:translateX(-50%) translateY(-16px); } to { opacity:1; transform:translateX(-50%) translateY(0); } }';
            document.head.appendChild(style);
        }

        document.body.appendChild(banner);

        // Auto-dismiss after 12 seconds
        setTimeout(() => banner.remove(), 12000);
    }

    // ---------------------------------------------------------------------------
    // Invalid Image Modal
    // ---------------------------------------------------------------------------

    /**
     * Populates and shows the invalid-image modal for a batch of rejected images.
     * Returns a Promise that resolves with 'retry' or 'ignore' based on user action.
     */
    async function showInvalidImagesModal(invalidResults) {
        const grid = document.getElementById('invalid-images-grid');
        grid.innerHTML = '';

        invalidResults.forEach(res => {
            const previewUrl = res.localImage ? URL.createObjectURL(res.localImage) : '';
            res.tempUrl = previewUrl;

            const imgBox = document.createElement('div');
            imgBox.style.cssText = 'position:relative; width:80px; height:80px; border-radius:50%; overflow:hidden; border:3px solid #FF0055; box-shadow:0 5px 15px rgba(255,0,85,0.2); flex-shrink:0;';
            imgBox.innerHTML = `
                <img src="${previewUrl}" style="width:100%;height:100%;object-fit:cover;">
                <div style="position:absolute;bottom:0;right:0;background:#FF0055;border-radius:50%;width:22px;height:22px;display:flex;justify-content:center;align-items:center;">
                    <ion-icon name="alert" style="color:white;font-size:12px;"></ion-icon>
                </div>
            `;
            grid.appendChild(imgBox);
        });

        document.getElementById('invalid-image-text').innerText =
            `${invalidResults.length} of your uploads do not appear to be vehicle photos.`;
        document.getElementById('invalid-image-modal').style.display = 'flex';

        const decision = await new Promise(resolve => {
            document.getElementById('modal-btn-retry').onclick  = () => { hideModal(); resolve('retry'); };
            document.getElementById('modal-btn-ignore').onclick = () => { hideModal(); resolve('ignore'); };
        });

        // Free blob URL memory
        invalidResults.forEach(res => { if (res.tempUrl) URL.revokeObjectURL(res.tempUrl); });

        return decision;
    }

    // ---------------------------------------------------------------------------
    // Dashboard Rendering
    // ---------------------------------------------------------------------------

    /** Builds a single damage or clean-signature card and appends it to container. */
    function renderDamageCard(res, container) {
        const localBlobSrc = res.localImage ? URL.createObjectURL(res.localImage) : 'dummy.jpg';
        const imgSrc = (res.image && res.image.startsWith('data:image')) ? res.image : localBlobSrc;
        const card = document.createElement('div');
        card.className = 'damage-item';

        if (res.has_damage) {
            const itemCost = res.estimated_cost_lkr || res.cost || 0;
            let detectionText = '';
            if (res.detections && res.detections.length > 0) {
                // Show each damage type once with its highest confidence — no duplicates
                const bestConf = {};
                res.detections.forEach(d => {
                    if (!bestConf[d.class] || d.confidence > bestConf[d.class]) bestConf[d.class] = d.confidence;
                });
                const labels = Object.entries(bestConf).map(([cls, conf]) => `${cls} ${(conf * 100).toFixed(0)}%`).join(' | ');
                detectionText = `<div style="font-size:12px;margin-top:6px;padding:4px 8px;background:#E0EBFF;color:var(--primary-blue);border-radius:4px;display:inline-block;">YOLO: ${labels}</div>`;
            }
            card.innerHTML = `
                <div style="position:relative;">
                    <img src="${imgSrc}" class="damage-item-img" onclick="window.open(this.src)" style="cursor:zoom-in;">
                    <ion-icon name="scan-outline" style="position:absolute;bottom:4px;left:4px;color:white;font-size:16px;pointer-events:none;"></ion-icon>
                </div>
                <div style="flex:1;">
                    <h4 style="margin-bottom:4px;font-weight:700;font-size:15px;">${res.repair_action || 'Surface Anomaly Detected'}</h4>
                    <p style="font-size:13px;color:var(--text-muted);line-height:1.4;">${res.vlm_reasoning || res.message || 'Inspection documented.'}</p>
                    ${detectionText}
                </div>
                <div style="font-weight:800;color:var(--primary-blue);font-size:15px;margin-top:2px;">Rs. ${itemCost.toLocaleString()}</div>
            `;
            container.appendChild(card);
            return itemCost;
        } else {
            card.innerHTML = `
                <img src="${imgSrc}" class="damage-item-img">
                <div style="flex:1;">
                    <h4 style="margin-bottom:4px;font-weight:700;font-size:15px;color:#2EBE6C;">Clean Signature</h4>
                    <p style="font-size:13px;color:var(--text-muted);line-height:1.4;">No visible damage metrics localized by YOLO engine on this axis.</p>
                </div>
            `;
            container.appendChild(card);
            return 0;
        }
    }

    /** Transitions the UI to the results dashboard and populates all data. */
    function renderDashboard(payload, priceData, validResults) {
        // Show Gemini quota warning if any image analysis had Gemini skipped
        const geminiWasSkipped = validResults.some(r => r.gemini_skipped === true);
        if (geminiWasSkipped) showGeminiWarning();

        // Show anomaly warning if the price model flagged the input specs as unusual
        if (priceData.is_anomalous && priceData.anomaly_reason) {
            showAnomalyWarning(priceData.anomaly_reason);
        }

        // Show results layout
        document.getElementById('setup-state').style.display = 'none';
        document.getElementById('results-state').style.display = 'block';
        document.getElementById('floating-menu').style.display = 'flex';
        window.scrollTo(0, 0);

        // Hero card specs
        document.getElementById('res-make').innerText     = payload.Make;
        document.getElementById('res-model').innerText    = payload.Model;
        document.getElementById('res-subtitle').innerText = `${payload.YOM} - Certified Pre-Purchase Report`;
        document.getElementById('res-mileage').innerText  = payload.Mileage_km.toLocaleString();
        document.getElementById('res-condition').innerText = payload.Condition;
        document.getElementById('res-engine').innerText   = `${payload.Engine_cc}cc`;

        const finalPrice = priceData.predicted_price_lkr || 14500000;
        document.getElementById('res-price').innerText = `LKR ${finalPrice.toLocaleString('en-US')}`;

        // Hero image — first valid result
        const first = validResults[0];
        const heroSrc = (first.image && first.image.startsWith('data:image'))
            ? first.image
            : (first.localImage ? URL.createObjectURL(first.localImage) : 'dummy.jpg');
        document.getElementById('res-main-img').src = heroSrc;

        // Damage cards
        let totalEst = 0;
        let hits = 0;
        const container = document.getElementById('damage-list-container');

        validResults.forEach(res => {
            const cost = renderDamageCard(res, container);
            if (res.has_damage) { hits++; totalEst += cost; }
        });

        document.getElementById('res-damage-count').innerText = hits;
        document.getElementById('res-repair-cost').innerText  = `LKR ${totalEst.toLocaleString()}`;

        // Clean up blob URLs after images have loaded
        setTimeout(() => {
            validResults.forEach(res => {
                if (res.localImage) {
                    try { URL.revokeObjectURL(URL.createObjectURL(res.localImage)); } catch (_) {}
                }
            });
        }, 8000);
    }

    // ---------------------------------------------------------------------------
    // Evaluate Button — Main Orchestrator
    // ---------------------------------------------------------------------------

    evalBtn.addEventListener('click', async () => {
        if (!makeInput.value || !modelInput.value || queuedFiles.length === 0) {
            alert('Please complete the vehicle Make & Model, AND upload at least one photo.');
            return;
        }

        evalBtn.disabled = true;
        evalBtn.innerText = 'Processing Pipeline...';
        spinner.style.display = 'inline-block';

        const payload = {
            Make:             makeInput.value,
            Model:            modelInput.value,
            YOM:              parseInt(document.getElementById('yom').value),
            Mileage_km:       parseInt(document.getElementById('mileage').value),
            Engine_cc:        parseInt(document.getElementById('engine').value),
            Fuel_Type:        document.getElementById('fuel').value,
            Gear:             document.getElementById('gear').value,
            Condition:        document.getElementById('condition').value,
            Has_AC:           document.getElementById('ac').checked,
            Has_PowerSteering: document.getElementById('power-steering').checked,
            Has_PowerMirror:  document.getElementById('power-mirror').checked,
            Has_PowerWindow:  document.getElementById('power-window').checked,
        };

        try {
            // Fire price prediction and all image analyses concurrently
            const priceReq = fetch(`${URL_BASE}/predict_price`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            }).then(r => r.json()).catch(() => ({ predicted_price_lkr: 14500000 }));

            const damageReqs = queuedFiles.map(file => {
                const fd = new FormData();
                fd.append('file', file);
                return fetch(`${URL_BASE}/analyze_damage`, { method: 'POST', body: fd })
                    .then(r => r.json())
                    .then(data => ({ ...data, localImage: file }))
                    .catch(() => ({
                        has_damage: true,
                        message: 'Server connection failed.',
                        estimated_cost_lkr: 0,
                        repair_action: 'Identified Imperfection',
                        vlm_reasoning: '(Backend connection fallback) This area requires professional evaluation.',
                        detections: [],
                    }));
            });

            const [priceData, ...damageResults] = await Promise.all([priceReq, ...damageReqs]);

            // Gatekeeper: split valid vs. invalid
            const validResults   = damageResults.filter(r => r.status !== 'invalid_image');
            const invalidResults = damageResults.filter(r => r.status === 'invalid_image');

            if (invalidResults.length > 0) {
                const decision = await showInvalidImagesModal(invalidResults);
                if (decision === 'retry') { resetEvalButton(); return; }
            }

            if (validResults.length === 0) {
                resetEvalButton();
                alert('Every uploaded image was invalid! Please upload at least one valid vehicle photo.');
                return;
            }

            renderDashboard(payload, priceData, validResults);

        } catch (e) {
            alert('Evaluation pipeline encountered a critical block: ' + e.message);
            resetEvalButton();
            spinner.style.display = 'none';
        }
    });
});
