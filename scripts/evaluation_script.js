document.getElementById('eval-form').onsubmit = async function(e) {
    e.preventDefault(); // Prevent default form submission

    const metricsDiv = document.getElementById('metrics_div');
    metricsDiv.innerHTML = '<div class="spinner" style="margin-top: 20px; margin: 0 auto;"></div>'; // Show loading spinner
    metricsDiv.style.marginTop = '15px';

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

// Enhanced File Upload UI - Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', () => {
    // Only initialize if we're on the metrics tab or elements exist
    const fileUploadArea = document.getElementById('file-upload-area');
    const audioInput = document.getElementById('audio');
    const selectedFileDisplay = document.getElementById('selected-file-display');
    const filenameSpan = document.getElementById('selected-filename');
    const filesizeSpan = document.getElementById('selected-filesize');
    const removeBtn = document.getElementById('remove-file-btn');
    const evaluationBtn = document.getElementById('evaluation-btn');
    const referenceTextarea = document.getElementById('reference');

    // Check if all required elements exist before proceeding
    if (!fileUploadArea || !audioInput || !selectedFileDisplay || !filenameSpan || 
        !filesizeSpan || !removeBtn || !evaluationBtn || !referenceTextarea) {
        console.warn('Some evaluation UI elements not found, skipping initialization');
        return;
    }

    console.log('Initializing evaluation file upload UI...');

    // Click to upload functionality
    fileUploadArea.addEventListener('click', (e) => {
        if (e.target === fileUploadArea || e.target.closest('.upload-content')) {
            audioInput.click();
        }
    });

    // Drag and drop functionality
    fileUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        fileUploadArea.classList.add('dragover');
    });

    fileUploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        fileUploadArea.classList.remove('dragover');
    });

    fileUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        fileUploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('audio/')) {
            // Create a new FileList-like object
            const dt = new DataTransfer();
            dt.items.add(files[0]);
            audioInput.files = dt.files;
            handleFileSelection(files[0]);
        }
    });

    // File input change
    audioInput.addEventListener('change', (e) => {
        console.log('File input changed:', e.target.files);
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    // Handle file selection
    function handleFileSelection(file) {
        console.log('Handling file selection:', file.name);
        filenameSpan.textContent = file.name;
        filesizeSpan.textContent = formatFileSize(file.size);
        
        fileUploadArea.style.display = 'none';
        selectedFileDisplay.style.display = 'flex';
        
        updateEvaluationButton();
    }

    // Remove file
    removeBtn.addEventListener('click', (e) => {
        e.preventDefault();
        console.log('Removing file...');
        audioInput.value = '';
        fileUploadArea.style.display = 'block';
        selectedFileDisplay.style.display = 'none';
        updateEvaluationButton();
    });

    // Character count for textarea
    referenceTextarea.addEventListener('input', () => {
        updateEvaluationButton();
    });

    // Update evaluation button state
    function updateEvaluationButton() {
        const hasFile = audioInput.files.length > 0;
        const hasReference = referenceTextarea.value.trim().length > 0;
        evaluationBtn.disabled = !(hasFile && hasReference);
        console.log('Button state - hasFile:', hasFile, 'hasReference:', hasReference, 'disabled:', evaluationBtn.disabled);
    }

    // Format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Initial button state
    updateEvaluationButton();
});