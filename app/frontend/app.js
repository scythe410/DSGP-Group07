document.addEventListener('DOMContentLoaded', () => {
    const priceForm = document.getElementById('price-form');

    if (priceForm) {
        priceForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            // UI elements
            const btn = document.getElementById('predict-btn');
            const spinner = document.getElementById('loading-spinner');
            const resultBox = document.getElementById('result-box');
            const resultPrice = document.getElementById('result-price');
            const resultSettings = document.getElementById('result-settings');

            // Start Loading State
            spinner.style.display = 'inline-block';
            resultBox.classList.remove('visible');
            btn.style.opacity = '0.8';
            btn.disabled = true;

            // Gather Data
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

            try {
                // Determine API URL (using localhost for local development)
                const apiUrl = 'http://127.0.0.1:8000/predict_price';

                const response = await fetch(apiUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();

                if (response.ok) {
                    // Success! Format number with commas
                    const formattedPrice = data.predicted_price_lkr.toLocaleString('en-US');
                    resultPrice.textContent = `LKR ${formattedPrice}`;
                    resultSettings.textContent = `Analysis based on ${payload.YOM} ${payload.Make} ${payload.Model} with ${payload.Mileage_km.toLocaleString('en-US')} km`;

                    // Show result box
                    resultBox.classList.add('visible');
                } else {
                    alert(`Error: ${data.detail || 'Prediction failed.'}`);
                }
            } catch (error) {
                alert('Connection error. Is the backend FastAPI server running on Port 8000?');
                console.error(error);
            } finally {
                // Stop Loading State
                spinner.style.display = 'none';
                btn.style.opacity = '1';
                btn.disabled = false;
            }
        });
    }
});
