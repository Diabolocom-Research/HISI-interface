const input = document.querySelector('.audio-evaluation');
const filenameContainer = document.getElementById('selected-filename-container');
const filenameSpan = document.getElementById('selected-filename');
const removeBtn = document.getElementById('remove-file-btn');
const audioMetrics = document.getElementById('audio');
const textarea = document.getElementById('reference');
const runBtn = document.getElementById('evaluation-btn');

function checkInputs() {
    runBtn.disabled = !(audioMetrics.files.length > 0 && textarea.value.trim().length > 0);
}

audioMetrics.addEventListener('change', checkInputs);
textarea.addEventListener('input', checkInputs);

document.getElementById('custom-audio-upload').addEventListener('click', function() {
    document.getElementById('audio').click();
});

input.addEventListener('change', function() {
    if (input.files.length) {
        filenameSpan.textContent = input.files[0].name;
        filenameContainer.style.display = 'flex';
    } else {
        filenameSpan.textContent = '';
        filenameContainer.style.display = 'none';
    }
});

removeBtn.addEventListener('click', function() {
    input.value = '';
    filenameSpan.textContent = '';
    filenameContainer.style.display = 'none';
});

document.getElementById('eval-form').onsubmit = async function(e) {
    e.preventDefault(); // Prevent default form submission

    const metricsDiv = document.getElementById('metrics_div');
    metricsDiv.innerHTML = '<div class="spinner" style="margin-top: 20px; margin: 0 auto;"></div>'; // Show loading spinner
    metricsDiv.style.marginTop = '15px';
    // metricsDiv.classList.add('loading');

    const formData = new FormData(this); // Get form data including file

    try {
        const response = await fetch('/evaluate_model', {
            method: 'POST',
            body: formData // FormData object is automatically set with correct Content-Type
        });

        if (!response.ok) {
            console.log('Evaluation failed with status:', response);
            const errorData = await response.json();
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        const data = await response.json(); // Parse the JSON response
        
        // Display results
        metricsDiv.classList.remove('loading');
        metricsDiv.innerHTML = `
            <p>WER (Word Error Rate): ${data.wer.toFixed(4)}</p>
            <p>CER (Character Error Rate): ${data.cer.toFixed(4)}</p>
            <div style="display: flex; flex-direction: column; gap: 12px;">
                <p>Hypothesis:</p>
                <textarea class="config-json" rows="4" readonly>${data.hypothesis}</textarea>
            </div>
            <div style="display: flex; flex-direction: column; gap: 12px;">
                <p>Reference:</p>
                <textarea class="config-json" rows="4" readonly>${formData.get('reference')}</textarea>
            </div>
        `;
    } catch (error) {
        console.error('Evaluation failed:', error);
        metricsDiv.classList.remove('loading');
        metricsDiv.innerHTML = `<p style="color: red;">Error during evaluation: ${error.message}</p>`;
    }
};