// --- TIMELINE CODE ---
let timeline, timelineItems;
let recordingTimer = null;
let recordingStartTime = null;
let recordingDuration = 0;

function startRecordingTimer() {
    const timerElement = document.getElementById('recording-timer');
    const recordingStatusContainer = document.querySelector('#recording-status');

    recordingStatusContainer.style.display = 'flex';
    recordingStartTime = Date.now();
    
    recordingTimer = setInterval(() => {
        recordingDuration = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(recordingDuration / 60);
        const seconds = recordingDuration % 60;
        timerElement.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }, 1000);
}

function stopRecordingTimer() {
    const recordingStatusContainer = document.querySelector('#recording-status');


    if (recordingTimer) {
        clearInterval(recordingTimer);
        recordingTimer = null;
    }
    
    recordingStatusContainer.style.display = 'none';
    recordingDuration = 0;
}

function initTimeline() {
    const container = document.getElementById('transcript-timeline');
    timelineItems = new vis.DataSet([]);
    const options = {
        dataAttributes: ['id'],
        onUpdate: function (item, callback) {
            // Get current values in seconds
            const currentStart = item.start instanceof Date ? item.start.getTime() / 1000 : item.start / 1000;
            const currentEnd = item.end instanceof Date ? item.end.getTime() / 1000 : item.end / 1000;

            // Show modal
            const modal = document.getElementById('timeline-edit-modal');
            document.getElementById('timeline-edit-text').value = item.content;
            document.getElementById('timeline-edit-start').value = currentStart;
            document.getElementById('timeline-edit-end').value = currentEnd;
            modal.style.display = 'flex';

            // Save handler
            function saveHandler() {
                const newText = document.getElementById('timeline-edit-text').value.trim();
                const newStart = parseFloat(document.getElementById('timeline-edit-start').value);
                const newEnd = parseFloat(document.getElementById('timeline-edit-end').value);

                if (!newText || isNaN(newStart) || isNaN(newEnd) || newStart >= newEnd) {
                    alert('Invalid input. Please check your values.');
                    return;
                }

                item.content = newText;
                item.start = new Date(newStart * 1000);
                item.end = new Date(newEnd * 1000);

                modal.style.display = 'none';
                document.getElementById('timeline-edit-save').removeEventListener('click', saveHandler);
                document.getElementById('timeline-edit-cancel').removeEventListener('click', cancelHandler);
                callback(item);
            }

            // Cancel handler
            function cancelHandler() {
                modal.style.display = 'none';
                document.getElementById('timeline-edit-save').removeEventListener('click', saveHandler);
                document.getElementById('timeline-edit-cancel').removeEventListener('click', cancelHandler);
                callback(null);
            }

            document.getElementById('timeline-edit-save').addEventListener('click', saveHandler);
            document.getElementById('timeline-edit-cancel').addEventListener('click', cancelHandler);
        },
        showCurrentTime: true,
        editable: {
            add: false,
            remove: true,
            updateGroup: false,
            updateTime: true,
            overrideItems: true
        },
        stack: true,
        itemsAlwaysDraggable: false,
        showMajorLabels: false,
        format: {
            minorLabels: {
                millisecond: "mm:ss.SSS",
                second: "mm:ss",
                minute: "mm:ss",
                hour: "HH:mm:ss"
            }
        },
        rollingMode: {
            follow: true,
            offset: 0.5
        },
        min: 0,
        max: 10000, 
        start: 0,   
        end: 10000, 
        zoomMin: 100,    
        zoomMax: 10000,
        zoomFriction: 5,
        height: '180px'
    };
    timeline = new vis.Timeline(container, timelineItems, options);
    // Add new vertical bar representing a custom time
    timeline.addCustomTime(0, 'cursor');
    window.timeline = timeline;
}

document.addEventListener('DOMContentLoaded', initTimeline);

function updateTimeline(segments) {
    if (!timelineItems) return;
    timelineItems.clear();
    timeline.zoomIn(1);
    timeline.moveTo(1); 

    console.log("Updating timeline with segments:", segments);

    segments.forEach((segment, idx) => {
        timelineItems.add({
            id: idx,
            content: segment.text,
            start: segment.start * 1000, // Convert to milliseconds
            end: segment.end * 1000, // Convert to milliseconds
        });

        timeline.setOptions({
            max: segment.start * 1000 + 2000,
        });
    
        timeline.fit();
    });
    
    window.segments = segments;

    if (segments.length > 0) {
        timeline.fit();
    }
}

// --- DOM REFERENCES ---
const progressElement = document.getElementById('progress');
const viewModelSelection = document.getElementById('view-model-selection');
const viewLoading = document.getElementById('view-loading');
const viewTranscription = document.getElementById('view-transcription');
const loadModelButton = document.getElementById('load-model-button');
const configTextarea = document.getElementById('config-json');
const startButton = document.getElementById('start-button');
const startButtonText = document.querySelector('.start-button-txt');
const recordIcon = document.querySelector('.record-icon'); 
const micImg = document.querySelector('.microphone-icon');

// --- FIX 1: Add the missing variable declarations ---
const transcriptTextElement = document.getElementById('transcript-text');
const segmentsTableBody = document.getElementById('segments-table-body');

// --- Global State ---
let peerConnection; let webrtc_id; let eventSource; let historicalTranscript = "";

// --- Default Configuration for the Text Area ---
const defaultConfig = {
    "model": "tiny",
    "lan": "auto",
    "task": "transcribe",
    "backend": "whisper_timestamped",
    // --- FIX 2: Increase min_chunk_size for more stable processing ---
    "min_chunk_size": 1.0,
    "buffer_trimming": "segment",
    "buffer_trimming_sec": 10.0
};

// --- CORE LOGIC ---
function updateButtonState() {
    if (!peerConnection || peerConnection.connectionState === "closed" || peerConnection.connectionState === "failed") {
        startButtonText.textContent = 'Start Recording';
        micImg.src = 'static/assets/microphone_icon.png'; 
        stopRecordingTimer(); // Stop timer when not recording
    } else if (peerConnection.connectionState === "connecting" || peerConnection.connectionState === "new") {
        startButtonText.textContent = 'Connecting...';
    } else if (peerConnection.connectionState === "connected") {
        startButtonText.textContent = 'Stop Recording';
        micImg.src = 'static/assets/stop_recording.png'; 
        startRecordingTimer(); // Start timer when recording begins
    }
}

function switchView(viewId) {
    viewModelSelection.classList.add('view-hidden');
    viewLoading.classList.add('view-hidden');
    viewTranscription.classList.add('view-hidden');
    document.getElementById(viewId)?.classList.remove('view-hidden');
    document.body.style.alignItems = (viewId === 'view-transcription') ? 'normal' : 'center';
    document.body.style.justifyContent = (viewId === 'view-transcription') ? 'normal' : 'center';
}

async function loadModel() {
    switchView('view-loading');
    const jsonString = configTextarea.value;
    let configPayload;
    try {
        configPayload = JSON.parse(jsonString);
    } catch (e) {
        alert(`Invalid JSON configuration:\n${e.message}`);
        switchView('view-model-selection');
        return;
    }
    try {
        const response = await fetch('/load_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(configPayload)
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }
        document.title = "Whisper Transcription: Live";
        switchView('view-transcription');
        window.createWaveSurfer();
    } catch (err) {
        alert(`Error loading model: ${err.message}`);
        switchView('view-model-selection');
    }
}

async function setupWebRTC() {
    // Immediately update the button state when we start
    startButtonText.textContent = 'Connecting...';

    document.getElementById('mic').style.display = 'block';

    try {
        const config = window.RTC_CONFIGURATION;
        peerConnection = new RTCPeerConnection(config);

        // This event listener now correctly calls our state management function
        peerConnection.addEventListener('connectionstatechange', () => {

            if (peerConnection.connectionState === 'connected') {
                transcriptTextElement.textContent = historicalTranscript;
                segmentsTableBody.innerHTML = ''; // Clear table on new session
            }
            // This call will update the button to "Stop Recording" when connected
            // or reset it if the connection fails.
            updateButtonState();
        });

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach(track => peerConnection.addTrack(track, stream));

        const dataChannel = peerConnection.createDataChannel('text');
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);

        await new Promise((resolve) => {
            if (peerConnection.iceGatheringState === "complete") { resolve(); }
            else {
                const checkState = () => {
                    if (peerConnection.iceGatheringState === "complete") {
                        peerConnection.removeEventListener("icegatheringstatechange", checkState);
                        resolve();
                    }
                };
                peerConnection.addEventListener("icegatheringstatechange", checkState);
            }
        });

        webrtc_id = Math.random().toString(36).substring(7);
        const response = await fetch('/webrtc/offer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sdp: peerConnection.localDescription.sdp, type: peerConnection.localDescription.type, webrtc_id: webrtc_id })
        });
        if (!response.ok) throw new Error(`Server returned an error: ${response.status} ${response.statusText}`);

        const serverResponse = await response.json();
        await peerConnection.setRemoteDescription(serverResponse);

        eventSource = new EventSource(`/transcript?webrtc_id=${webrtc_id}`);
        eventSource.addEventListener("output", (event) => handleServerUpdate(event.data));
        eventSource.onerror = (err) => {
            console.error("Transcript stream disconnected.", err);
            eventSource.close();
        };
    } catch (err) {
        console.error('Failed to establish connection.', err);
        alert('Failed to establish connection. Check console for details.');
        stop(); // Call stop to clean up and reset the UI
    }
}

let currentSegments = [];

function handleServerUpdate(data) {
    try {
        const payload = JSON.parse(data);
        transcriptTextElement.textContent = historicalTranscript + (payload.full_transcript || "");
        document.getElementById('transcript-container').scrollTop = document.getElementById('transcript-container').scrollHeight;
        const segments = payload.segments || [];
        
        // Store segments globally
        currentSegments = segments;

        segmentsTableBody.innerHTML = '';
        segments.forEach(segment => {
            const row = document.createElement('tr');
            row.innerHTML = `<td>${segment.start.toFixed(2)}</td><td>${segment.end.toFixed(2)}</td><td>${segment.text}</td>`;
            segmentsTableBody.appendChild(row);
        });

        updateTimeline(segments);
    } catch (e) {
        console.error("Failed to parse server data:", data, e);
    }
}

function addRegionsToRecordedWaveform() {
    console.log("Adding regions to recorded waveform...", currentSegments);
    
    // Check if we have a valid waveform and segments
    if (!window.lastRecordedWaveSurfer || !currentSegments || currentSegments.length === 0) {
        console.log("No recorded waveform or segments available");
        return;
    }

    // Wait for the waveform to be ready
    if (!window.lastRecordedWaveSurfer.isReady) {
        console.log("Waveform not ready, waiting...");
        window.lastRecordedWaveSurfer.once('ready', () => {
            addRegionsToRecordedWaveform();
        });
        return;
    }

    // Clear existing regions first
    if (window.regions) {
        window.regions.clearRegions();
    }

    console.log("Adding regions:", currentSegments);
    
    // Add regions with error handling
    currentSegments.forEach((segment, index) => {
        try {
            const region = window.regions.addRegion({
                start: segment.start,
                end: segment.end,
                content: segment.text,
                color: getRandomColor(),
                drag: false,
                resize: false,
                id: `segment-${index}` // Add unique ID
            });
            console.log(`Added region ${index}:`, region);
        } catch (error) {
            console.error(`Failed to add region ${index}:`, error);
        }
    });
    
    console.log("Total regions added:", window.regions.getRegions().length);
}

function stop() {
    if (peerConnection) {
        peerConnection.getSenders().forEach(sender => sender.track?.stop());
        peerConnection.close();
        peerConnection = null;
    }
    
    if (eventSource) eventSource.close();
    if (transcriptTextElement && transcriptTextElement.textContent && transcriptTextElement.textContent.trim() !== historicalTranscript.trim()) {
        historicalTranscript = transcriptTextElement.textContent + "\n\n";
    }
    
    // // Delay adding regions to ensure waveform is ready
    // setTimeout(() => {
    //     addRegionsToRecordedWaveform();
    // }, 500); // 500ms delay
    
    updateButtonState();
}

function getRandomColor() {
    const colors = [
        "rgba(186, 233, 255, 0.5)",
        "rgba(201, 247, 210, 0.5)", 
        "rgba(248, 235, 185, 0.5)",
        "rgba(194, 192, 255, 0.5)"
    ];
    return colors[Math.floor(Math.random() * colors.length)];
}

// --- INITIALIZATION ---
configTextarea.value = JSON.stringify(defaultConfig, null, 2);
loadModelButton.addEventListener('click', loadModel);

startButton.addEventListener('click', () => {
    if (!peerConnection || peerConnection.connectionState === "closed") { 
        setupWebRTC(); 
    }
    else { 
        stop(); 
    }
});