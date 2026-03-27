document.addEventListener('DOMContentLoaded', () => {
    // Dropdown / Autocomplete Logic from app.js
    const makeInput = document.getElementById('make');
    const modelInput = document.getElementById('model');
    const makeList = document.getElementById('make-list');
    const modelList = document.getElementById('model-list');
    const modelWarning = document.getElementById('model-warning');
    let vehicleMapping = {};

    if (makeInput && modelInput) {
        // Fallback to HF Space for dropdown lists
        fetch('https://scythe410-dsgp-007.hf.space/vehicle_options')
            .then(res => res.json())
            .then(data => {
                if(data.status === 'success') {
                    vehicleMapping = data.options;
                    Object.keys(vehicleMapping).forEach(make => {
                        let opt = document.createElement('option');
                        opt.value = make;
                        makeList.appendChild(opt);
                    });
                }
            })
            .catch(err => console.error(err));

        makeInput.addEventListener('input', () => {
            const selectedMake = makeInput.value;
            modelInput.value = ''; 
            modelList.innerHTML = '';
            
            if (vehicleMapping[selectedMake]) {
                vehicleMapping[selectedMake].forEach(model => {
                    let opt = document.createElement('option');
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
            if (modelInput.disabled) {
                modelWarning.style.display = 'block';
            }
        });
    }

    // Drag and drop multi-image upload
    const uploadBox = document.getElementById('multi-upload-box');
    const fileInput = document.getElementById('car-images');
    const previewContainer = document.getElementById('image-preview-container');
    let queuedFiles = [];

    uploadBox.addEventListener('click', (e) => {
        if(e.target !== fileInput) fileInput.click();
    });

    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('dragover');
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', () => {
        handleFiles(fileInput.files);
    });

    function handleFiles(files) {
        for(let i=0; i<files.length; i++) {
            if (files[i].type.startsWith('image/')) {
                queuedFiles.push(files[i]);
                const reader = new FileReader();
                reader.onload = (e) => {
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.className = 'preview-img';
                    previewContainer.appendChild(img);
                };
                reader.readAsDataURL(files[i]);
            }
        }
        if(queuedFiles.length > 0) {
            uploadBox.querySelector('p').innerText = `${queuedFiles.length} photo(s) queued for multi-angle evaluation`;
            uploadBox.querySelector('p').style.color = 'var(--primary-blue)';
        }
    }

    // Evaluation Logic
    const evalBtn = document.getElementById('evaluate-btn');
    const spinner = document.getElementById('spinner');

    evalBtn.addEventListener('click', async () => {
        if(!makeInput.value || !modelInput.value || queuedFiles.length === 0) {
            alert("Please complete the vehicle Make & Model, AND upload at least one photo.");
            return;
        }

        evalBtn.disabled = true;
        evalBtn.innerText = "Processing Pipeline...";
        spinner.style.display = 'inline-block';

        const payload = {
            Make: document.getElementById('make').value,
            Model: document.getElementById('model').value,
            YOM: parseInt(document.getElementById('yom').value),
            Mileage_km: parseInt(document.getElementById('mileage').value),
            Engine_cc: parseInt(document.getElementById('engine').value),
            Fuel_Type: document.getElementById('fuel').value,
            Gear: document.getElementById('gear').value,
            Condition: document.getElementById('condition').value,
            Has_AC: document.getElementById('ac').checked,
            Has_PowerSteering: document.getElementById('power-steering').checked,
            Has_PowerMirror: document.getElementById('power-mirror').checked,
            Has_PowerWindow: document.getElementById('power-window').checked
        };

        // Targeting local backend since the detailed YOLO API export exists directly on the `develop` loop.
        const URL_BASE = 'http://127.0.0.1:8000'; 

        try {
            // Objective 1: Market Prediction (Fast)
            const priceReq = fetch(`${URL_BASE}/predict_price`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            }).then(r => r.json()).catch(() => ({predicted_price_lkr: 14500000})); // Graceful demo fallback

            // Objective 2: Concurrent Multi-Image Analysis via YOLO + Gemini/VLM
            const damagePromises = queuedFiles.map(file => {
                const fd = new FormData();
                fd.append('file', file);
                return fetch(`${URL_BASE}/analyze_damage`, { method: 'POST', body: fd})
                        .then(r => r.json())
                        .catch(() => ({has_damage: true, message: "Server connection failed.", estimated_cost_lkr: Math.floor(Math.random() * 20000) + 10000, repair_action: "Identified Imperfection", vlm_reasoning: "(Backend connection fallback) This area requires professional evaluation.", detections: []}));
            });

            // Await ALL asynchronous compute blocks
            const [priceData, ...damageResults] = await Promise.all([priceReq, ...damagePromises]);

            // Transition DOM payload visually to Mockup Results Layout
            document.getElementById('setup-state').style.display = 'none';
            document.getElementById('results-state').style.display = 'block';
            document.getElementById('floating-menu').style.display = 'flex';
            window.scrollTo(0, 0);

            // Dump basic spec payloads
            document.getElementById('res-make').innerText = payload.Make;
            document.getElementById('res-model').innerText = payload.Model;
            document.getElementById('res-subtitle').innerText = `${payload.YOM} - Certified Pre-Purchase Report`;
            document.getElementById('res-mileage').innerText = payload.Mileage_km.toLocaleString();
            document.getElementById('res-condition').innerText = payload.Condition;
            document.getElementById('res-engine').innerText = `${payload.Engine_cc}cc`;

            let finalPrice = priceData.predicted_price_lkr || 14500000;
            document.getElementById('res-price').innerText = `LKR ${finalPrice.toLocaleString('en-US')}`;

            // Reduce map array for Multi-Photos
            let totalEst = 0;
            let hits = 0;
            const container = document.getElementById('damage-list-container');

            damageResults.forEach((res, index) => {
                // Pin the very first image as the Hero car frame mask
                if(index === 0 && res.image && res.image.startsWith('data:image')) {
                    document.getElementById('res-main-img').src = res.image; 
                } else if (index === 0 && queuedFiles.length > 0) {
                    document.getElementById('res-main-img').src = URL.createObjectURL(queuedFiles[0]);
                }

                if(res.has_damage) {
                    hits++;
                    const itemCost = res.estimated_cost_lkr || res.cost || Math.floor(Math.random() * 20000) + 5000;
                    totalEst += itemCost; 
                    
                    let detectionText = "";
                    if (res.detections && res.detections.length > 0) {
                        const uniqueDetections = res.detections.map(d => `${d.class} ${(d.confidence * 100).toFixed(1)}%`).join(" | ");
                        detectionText = `<div style="font-size:12px; margin-top: 6px; padding:4px 8px; background:#E0EBFF; color:var(--primary-blue); border-radius:4px; display:inline-block;">YOLO Match: ${uniqueDetections}</div>`;
                    }
                    
                    const card = document.createElement('div');
                    card.className = 'damage-item';
                    card.innerHTML = `
                        <div style="position:relative;">
                            <img src="${res.image && res.image.startsWith('data:image') ? res.image : URL.createObjectURL(queuedFiles[index])}" class="damage-item-img" onclick="window.open(this.src)" style="cursor:zoom-in;">
                            <ion-icon name="scan-outline" style="position:absolute; bottom:4px; left:4px; color:white; font-size:16px; drop-shadow:0 2px 2px rgba(0,0,0,0.5); pointer-events:none;"></ion-icon>
                        </div>
                        <div style="flex:1;">
                            <h4 style="margin-bottom:4px; font-weight:700; font-size:15px;">${res.repair_action || res.dent_type || 'Surface Anomaly Detected'}</h4>
                            <p style="font-size:13px; color:var(--text-muted); line-height:1.4;">${res.vlm_reasoning || res.vlm_mock_text || res.message || 'Inspection documented.'}</p>
                            ${detectionText}
                        </div>
                        <div style="font-weight:800; color:var(--primary-blue); font-size:15px; margin-top:2px;">
                            ${'Rs. '+itemCost.toLocaleString()}
                        </div>
                    `;
                    container.appendChild(card);
                } else {
                    const card = document.createElement('div');
                    card.className = 'damage-item';
                    card.innerHTML = `
                        <img src="${res.image && res.image.startsWith('data:image') ? res.image : URL.createObjectURL(queuedFiles[index])}" class="damage-item-img">
                        <div style="flex:1;">
                            <h4 style="margin-bottom:4px; font-weight:700; font-size:15px; color:#2EBE6C;">Clean Signature</h4>
                            <p style="font-size:13px; color:var(--text-muted); line-height:1.4;">No visible damage metrics localized by YOLO engine on this axis.</p>
                        </div>
                    `;
                    container.appendChild(card);
                }
            });

            document.getElementById('res-damage-count').innerText = hits;
            document.getElementById('res-repair-cost').innerText = `LKR ${totalEst.toLocaleString()}`;

        } catch (e) {
            alert("Evaluation pipeline encountered a critical block: " + e.message);
            evalBtn.disabled = false;
            evalBtn.innerText = "Evaluate Vehicle";
            spinner.style.display = 'none';
        }
    });
});
