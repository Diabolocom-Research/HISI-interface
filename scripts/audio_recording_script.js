// --- TIMELINE CODE ---
let timeline, timelineItems;

function initTimeline() {
    const container = document.getElementById('transcript-timeline');
    timelineItems = new vis.DataSet([]);
    const options = {
        showCurrentTime: true,
        editable: false,
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
        max: 40000,
        start: 0,
        zoomMin: 1000,
        zoomFriction: 6,
        height: '120px'
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

    segments.forEach((segment, idx) => {
        timelineItems.add({
            id: idx,
            content: segment.text,
            start: segment.start * 1000, // Convert to milliseconds
            end: segment.end * 1000, // Convert to milliseconds
        });

        if (segment.end > 40000) {
            timeline.setOptions({
                max: segment.start * 1000 + 20000,
            });
        }

        console.log(timelineItems);
        timeline.fit();
        // timeline.moveTo(secondsToDate(segment.start + 5));
        // timeline.focus(idx)
        // timeline.toggleRollingMode()	
        // timeline.setOptions({
        //     end: segment.end * 1000, // Extend end time by 5 seconds
        //     // max: segment.end * 1000 + 5000 // Extend max time by 5 seconds
        // });
        // timeline.zoomIn(1);
        // timeline.moveTo(segment.start * 1000); // Center the timeline on the segment start
    });
    
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
const recordIcon = document.querySelector('.record-icon'); 

// --- FIX 1: Add the missing variable declarations ---
const transcriptTextElement = document.getElementById('transcript-text');
const segmentsTableBody = document.getElementById('segments-table-body');

// --- Global State ---
let peerConnection; let webrtc_id; let eventSource; let historicalTranscript = "";

startButton.addEventListener('click', () => {
    recordIcon.classList.add('recording');
});

function stopRecording() {
    recordIcon.classList.remove('recording');
}

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
        startButton.textContent = 'Record';
        startButton.disabled = false;
    } else if (peerConnection.connectionState === "connecting" || peerConnection.connectionState === "new") {
        startButton.textContent = 'Connecting...';
        startButton.disabled = true;
    } else if (peerConnection.connectionState === "connected") {
        startButton.textContent = 'Stop';
        startButton.disabled = false;
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
        if (window.createWaveSurfer) window.createWaveSurfer();
    } catch (err) {
        alert(`Error loading model: ${err.message}`);
        switchView('view-model-selection');
    }
}

async function setupWebRTC() {
    // Immediately update the button state when we start
    startButton.textContent = 'Connecting...';
    startButton.disabled = true;

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

function handleServerUpdate(data) {
    try {
        const payload = JSON.parse(data);
        transcriptTextElement.textContent = historicalTranscript + (payload.full_transcript || "");
        document.getElementById('transcript-container').scrollTop = document.getElementById('transcript-container').scrollHeight;
        const segments = payload.segments || [];
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
    updateButtonState();
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
        stopRecording();
    }
});