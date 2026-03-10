document.addEventListener('DOMContentLoaded', () => {
    const uploadBox = document.getElementById('upload-box');
    const imageInput = document.getElementById('image-input');
    const processingState = document.getElementById('processing-state');
    const processingText = document.getElementById('processing-text');
    const damageResults = document.getElementById('damage-results');
    const resultImage = document.getElementById('result-image');
    const damageHeader = document.getElementById('damage-header');

    if (uploadBox && imageInput) {
        imageInput.addEventListener('change', (e) => {
            try {
                const file = e.target.files[0];
                if (!file) {
                    console.log("No file selected.");
                    return;
                }

                console.log("File selected:", file.name);

                const reader = new FileReader();
                reader.onload = (event) => {
                    resultImage.src = event.target.result;
                    console.log("Image preview loaded, starting API chain...");
                    startProcessingChain(file);
                };
                reader.onerror = (err) => {
                    alert('Error reading local file: ' + err.message);
                };
                reader.readAsDataURL(file);

                // Reset to allow the SAME file to be uploaded again if needed
                imageInput.value = '';
            } catch (err) {
                alert('JS Error on select: ' + err.message);
            }
        });
    }

    async function startProcessingChain(file) {
        try {
            uploadBox.style.display = 'none';
            if (damageHeader) damageHeader.style.display = 'none';
            processingState.style.display = 'block';

            processingText.innerHTML = "<strong>Step 1/2</strong><br><br>Assessing your car...";

            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('https://scythe410-dsgp-007.hf.space/analyze_damage', {
                method: 'POST',
                body: formData
            });

            processingText.innerHTML = "<strong>Step 2/2</strong><br><br>Cost reasoning...";

            const data = await response.json();

            if (response.ok) {
                // Remove hardcoded HTML overlay objects so they don't break the actual image display
                document.querySelectorAll('.damage-indicator-circle, .number-circle, .label-badge').forEach(el => {
                    if (el) el.style.display = 'none';
                });

                if (data.has_damage) {
                    const tagsContainer = document.querySelector('.damage-tags');
                    if (tagsContainer) {
                        tagsContainer.innerHTML = data.sides_affected.map(s =>
                            `<span class="damage-tag" style="background-color: #ffeb3b; border-radius: 4px; padding: 6px 12px; font-size: 12px; font-weight: 600; color: #000;">${s}</span>`
                        ).join('');
                    }

                    const costContainer = document.querySelector('.card-section-value');
                    if (costContainer) {
                        const costVal = typeof data.estimated_cost_lkr === 'number' 
                            ? `LKR ${data.estimated_cost_lkr.toLocaleString()}` 
                            : data.estimated_cost_lkr;
                        costContainer.innerHTML = `${costVal}`;
                    }

                    const groupsContainer = document.querySelector('.damage-groups');
                    if (groupsContainer) {
                        groupsContainer.innerHTML = `
                            <div class="card-section-label" style="font-size: 12px; color: #999; margin-bottom: 12px; font-weight: 500;">Damage Groups Detected</div>
                            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                                ${data.detected_groups.map(g => `<span class="damage-group-item" style="background-color: #f5f5f5; border: 1px solid #e0e0e0; border-radius: 4px; padding: 6px 12px; font-size: 13px; font-weight: 600; color: #111; display: inline-block; text-transform: capitalize;">${g}</span>`).join('')}
                            </div>
                        `;
                    }

                    const vlmTextContainers = document.querySelectorAll('strong');
                    vlmTextContainers.forEach(el => {
                        if (el.textContent.includes('VLM')) {
                            const parent = el.parentElement;
                            if (parent) {
                                parent.innerHTML = `
                                    <strong style="color: #000; display: block; margin-bottom: 5px;">Diagnostic Reasoning:</strong>
                                    <em>${data.vlm_reasoning}</em>
                                `;
                            }
                        }
                    });

                } else {
                    const tagsContainer = document.querySelector('.damage-tags');
                    if (tagsContainer) {
                        tagsContainer.innerHTML = `<span class="damage-tag" style="background-color: #e0e0e0; border-radius: 4px; padding: 6px 12px; font-size: 12px; font-weight: 600; color: #000;">None Detected</span>`;
                    }

                    const costContainer = document.querySelector('.card-section-value');
                    if (costContainer) costContainer.innerHTML = `LKR 0`;

                    const groupsContainer = document.querySelector('.damage-groups');
                    if (groupsContainer) {
                        groupsContainer.innerHTML = `
                             <div class="card-section-label" style="font-size: 12px; color: #999; margin-bottom: 12px; font-weight: 500;">Damage Groups Detected</div>
                             <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                                 <span class="damage-group-item" style="background-color: #f5f5f5; border: 1px solid #e0e0e0; border-radius: 4px; padding: 6px 12px; font-size: 13px; font-weight: 600; color: #111; display: inline-block;">System Cleared</span>
                             </div>
                         `;
                    }

                    const vlmTextContainers = document.querySelectorAll('strong');
                    vlmTextContainers.forEach(el => {
                        if (el.textContent.includes('VLM')) {
                            const parent = el.parentElement;
                            if (parent) {
                                parent.innerHTML = `
                                     <strong style="color: #000; display: block; margin-bottom: 5px;">Diagnostic Reasoning:</strong>
                                     <em>${data.message} No bounding boxes were passed from the YOLO model.</em>
                                 `;
                            }
                        }
                    });
                }

                // Show Result
                if (data.image_base64 && resultImage) {
                    resultImage.src = "data:image/jpeg;base64," + data.image_base64;
                }
                if (processingState) processingState.style.display = 'none';
                if (damageResults) damageResults.style.display = 'block';
            } else {
                alert(`API Error: ${response.status} - ${data.detail}`);
                uploadBox.style.display = 'block';
                processingState.style.display = 'none';
            }
        } catch (error) {
            console.error(error);
            alert('CRITICAL NETWORK OR JS ERROR: ' + error.message);
            uploadBox.style.display = 'block';
            if (processingState) processingState.style.display = 'none';
        }
    }
});
