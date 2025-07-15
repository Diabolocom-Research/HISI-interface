// --- RESET FUNCTION FOR RECORDING ---
function resetRecordingData() {
    // Stop any active recording
    if (peerConnection) {
        stop();
    }
    
    // Clear transcript
    if (transcriptTextElement) {
        transcriptTextElement.textContent = 'Speak to see transcript...';
    }
    
    // Clear segments table
    if (segmentsTableBody) {
        segmentsTableBody.innerHTML = '';
    }
    
    // Clear timeline items and reinitialize
    if (timelineItems) {
        timelineItems.clear();
        // Recreate the DataSet to ensure it's properly reset
        timelineItems = new vis.DataSet([]);
        console.log('Timeline DataSet recreated');
    }
    
    // Reset timeline cursor and state
    if (timeline) {
        timeline.setCustomTime(0, 'cursor');
        timeline.moveTo(0, { animation: false });
        timeline.setOptions({ max: 10000, start: 0, end: 10000 });
        // Ensure timeline is using the fresh DataSet
        timeline.setItems(timelineItems);
        console.log('Timeline reset and reinitialized');
    }
    
    // Clear regions from playback waveform
    if (window.regions) {
        window.regions.clearRegions();
    }
    
    // Reset recorded waveform
    if (window.lastRecordedWaveSurfer) {
        window.lastRecordedWaveSurfer.destroy();
        window.lastRecordedWaveSurfer = null;
    }
    
    if (window.wavesurfer) {
        window.wavesurfer.empty();
    }
    
    // Clear recordings container
    const recordingsContainer = document.getElementById('recordings');
    if (recordingsContainer) {
        recordingsContainer.innerHTML = '';
    }
    
    // Reset playback controls
    const controlsContainer = document.querySelector('.controls-container');
    if (controlsContainer) {
        controlsContainer.style.display = 'none';
    }
    
    // Reset audio duration displays
    const audioDuration = document.querySelector('.audio_duration');
    const totalAudioDuration = document.querySelector('.total_audio_duration_record');
    if (audioDuration) audioDuration.textContent = '00:00';
    if (totalAudioDuration) totalAudioDuration.textContent = '00:00';
    
    // Reset play button icon
    const playBtn = document.querySelector('.play_btn');
    if (playBtn) {
        playBtn.src = 'static/assets/play_black_icon_48.png';
    }
    
    // Reset global state
    historicalTranscript = '';
    currentSegments = [];
    recordingDuration = 0;
    
    // Stop recording timer
    stopRecordingTimer();
    
    // Hide mic visualization and show no-recording message
    const micElement = document.getElementById('mic');
    const noRecordingMessage = document.getElementById('no-recording-message');
    if (micElement) {
        micElement.style.display = 'none';
    }
    if (noRecordingMessage) {
        noRecordingMessage.style.display = 'flex';
    }
    
    // Reset button state
    updateButtonState();
    
    // Re-enable start button after reset
    if (startButton) {
        startButton.disabled = false;
        startButton.style.opacity = '1';
        startButton.style.cursor = 'pointer';
    }
    if (startButtonText) {
        startButtonText.textContent = 'Start Recording';
    }
    
    // Hide reset button since there's nothing to reset now
    const resetBtn = document.getElementById('reset-recording-btn-container');
    if (resetBtn) {
        resetBtn.style.display = 'none';
        resetBtn.disabled = false; // Ensure it's not disabled for next use
    }
    
    console.log('Recording data reset complete');
}

// --- TIMELINE CODE ---
let timeline, timelineItems;
let recordingTimer = null;
let recordingStartTime = null;
let recordingDuration = 0;

function startRecordingTimer() {
    const timerElement = document.getElementById('recording-timer');
    const recordingStatusContainer = document.querySelector('#recording-status');
    const startButton = document.getElementById('start-button');

    recordingStatusContainer.style.display = 'flex';
    startButton.style.innerHTML = 'Stop Recording';
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
    const startButton = document.getElementById('start-button');

    if (recordingTimer) {
        clearInterval(recordingTimer);
        recordingTimer = null;
    }
    
    recordingStatusContainer.style.display = 'none';
    startButton.style.display = 'flex'; // Show start button when not recording
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
    if (!timelineItems || !timeline) {
        console.log("Timeline not initialized properly");
        return;
    }
    
    // Clear existing items
    timelineItems.clear();

    console.log("Updating timeline with segments:", segments);

    if (segments && segments.length > 0) {
        // Add all segments to timeline
        segments.forEach((segment, idx) => {
            timelineItems.add({
                id: idx,
                content: segment.text,
                start: segment.start * 1000, // Convert to milliseconds
                end: segment.end * 1000, // Convert to milliseconds
            });
        });

        // Update timeline range based on last segment
        const lastSegment = segments[segments.length - 1];
        if (lastSegment) {
            timeline.setOptions({
                max: lastSegment.end * 1000 + 1000,
            });
        }
        
        // Move timeline to show latest content
        timeline.fit();
    } else {
        // Reset timeline when no segments
        timeline.setOptions({ max: 10000 });
        timeline.moveTo(0, { animation: false });
    }
    
    window.segments = segments;
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
const segmentsTableBody = document.getElementById('segments-table-body-recording');

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
    const resetBtn = document.getElementById('reset-recording-btn-container');
    
    if (!peerConnection || peerConnection.connectionState === "closed" || peerConnection.connectionState === "failed") {
        // Check if we have any recorded data - if so, disable start button
        const hasTranscript = transcriptTextElement && transcriptTextElement.textContent.trim() !== 'Speak to see transcript...';
        const hasSegments = currentSegments && currentSegments.length > 0;
        const hasRecording = window.lastRecordedWaveSurfer;
        
        if (hasTranscript || hasSegments || hasRecording) {
            // Recording session has ended with data - disable start button
            startButtonText.textContent = 'Recording Complete';
            startButton.style.opacity = '0.5';
            startButton.style.cursor = 'not-allowed';
            startButton.disabled = true;
        } else {
            // No data yet - enable start button
            startButtonText.textContent = 'Start Recording';
            startButton.style.opacity = '1';
            startButton.style.cursor = 'pointer';
            startButton.disabled = false;
        }
        
        micImg.src = 'static/assets/microphone_icon.png'; 
        stopRecordingTimer(); // Stop timer when not recording
        
        // Enable reset button if there's data to reset
        if (resetBtn) {
            resetBtn.disabled = false;
            showResetButtonIfNeeded();
        }
    } else if (peerConnection.connectionState === "connecting" || peerConnection.connectionState === "new") {
        startButtonText.textContent = 'Connecting...';
        startButton.disabled = false; // Keep enabled during connection
        startButton.style.opacity = '1';
        startButton.style.cursor = 'pointer';
        
        // Disable reset button during connection
        if (resetBtn) {
            resetBtn.disabled = true;
        }
    } else if (peerConnection.connectionState === "connected") {
        startButtonText.textContent = 'Stop Recording';
        micImg.src = 'static/assets/stop_recording.png'; 
        startRecordingTimer(); // Start timer when recording begins
        startButton.disabled = false; // Keep enabled so user can stop
        startButton.style.opacity = '1';
        startButton.style.cursor = 'pointer';
        
        // Disable reset button while recording
        if (resetBtn) {
            resetBtn.disabled = true;
        }
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
    
    // Hide the no-recording message when recording starts
    const noRecordingMessage = document.getElementById('no-recording-message');
    if (noRecordingMessage) {
        noRecordingMessage.style.display = 'none';
    }

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
            const duration = segment.end - segment.start;
            row.innerHTML = `
                <td class="time-cell">${formatTime(segment.start)}</td>
                <td class="time-cell">${formatTime(segment.end)}</td>
                <td class="duration-cell">${formatDuration(duration)}</td>
                <td style="max-width: 300px; word-wrap: break-word;">${segment.text}</td>
                <td>
                    <div class="segment-actions">
                        <button class="segment-btn segment-btn-play" data-start="${segment.start}" data-end="${segment.end}" title="Play segment">
                            ▶
                        </button>
                    </div>
                </td>
            `;
            segmentsTableBody.appendChild(row);

            const playButton = row.querySelector('.segment-btn-play');
            playButton.addEventListener('click', (e) => {
                const start = parseFloat(e.target.getAttribute('data-start'));
                const end = parseFloat(e.target.getAttribute('data-end'));
                playSegment(start, end);
            });
        });

        updateTimeline(segments);
        
        console.log(`Timeline update called with ${segments.length} segments`);
        console.log('Timeline state:', timelineItems ? `${timelineItems.length} items` : 'null');
        
        // Show reset button if there's data to reset
        showResetButtonIfNeeded();
    } catch (e) {
        console.error("Failed to parse server data:", data, e);
    }
}


document.querySelectorAll('.segment-btn-play').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const start = parseFloat(e.target.getAttribute('data-start'));
        const end = parseFloat(e.target.getAttribute('data-end'));
        playSegment(start, end);
    });
});

function playSegment(start, end) {
    if (window.lastRecordedWaveSurfer) {
        window.lastRecordedWaveSurfer.setTime(start);
        window.lastRecordedWaveSurfer.play();
        
        // Stop at end time
        const stopHandler = () => {
            if (window.lastRecordedWaveSurfer.getCurrentTime() >= end) {
                window.lastRecordedWaveSurfer.pause();
                window.lastRecordedWaveSurfer.un('audioprocess', stopHandler);
            }
        };
        window.lastRecordedWaveSurfer.on('audioprocess', stopHandler);
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
                // id: `segment-${index}` // Add unique ID
            });
            // console.log(`Added region ${index}:`, region);
        } catch (error) {
            console.error("Failed to add region for segment", error);
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
    
    updateButtonState();
}

function getRandomColor() {
    const colors = [
        "rgba(49, 151, 198, 0.9)",
        "rgba(56, 177, 80, 0.9)", 
        "rgba(184, 156, 45, 0.9)",
        "rgba(95, 92, 185, 0.9)"
    ];
    return colors[Math.floor(Math.random() * colors.length)];
}

// --- TIME FORMATTING HELPER FUNCTIONS ---
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs.toFixed(1)}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs.toFixed(1)}s`;
    } else {
        return `${secs.toFixed(1)}s`;
    }
}

// --- INITIALIZATION ---
configTextarea.value = JSON.stringify(defaultConfig, null, 2);
loadModelButton.addEventListener('click', loadModel);

startButton.addEventListener('click', () => {
    // Don't proceed if button is disabled
    if (startButton.disabled) {
        return;
    }
    
    if (!peerConnection || peerConnection.connectionState === "closed") { 
        setupWebRTC(); 
    }
    else { 
        stop(); 
    }
});

document.querySelector('.speaker-recording').addEventListener('click', () => {
    // Ensure audio is ready before toggling volum        
    const currentVolume = window.lastRecordedWaveSurfer.getVolume();
    const speakerImg = document.querySelector('.speaker-recording img');

    if (currentVolume > 0) {
        // Mute the audio
        window.lastRecordedWaveSurfer.setVolume(0);
        speakerImg.src = 'static/assets/speaker_black_icon_48.png';
    } else {
        // Unmute the audio
        window.lastRecordedWaveSurfer.setVolume(1);
        speakerImg.src = 'static/assets/mute_black_icon_48.png';
    }
});

// Add reset button event listener
document.addEventListener('DOMContentLoaded', () => {
    const resetBtn = document.getElementById('reset-recording-btn-container');
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            resetRecordingData();
        });
    }
});

// Show reset button when there's data to reset
function showResetButtonIfNeeded() {
    const resetBtn = document.getElementById('reset-recording-btn-container');
    if (!resetBtn) return;
    
    const hasTranscript = transcriptTextElement && transcriptTextElement.textContent.trim() !== 'Speak to see transcript...';
    const hasSegments = currentSegments && currentSegments.length > 0;
    const hasRecording = window.lastRecordedWaveSurfer;
    
    if (hasTranscript || hasSegments || hasRecording) {
        resetBtn.style.display = 'flex';
    } else {
        resetBtn.style.display = 'none';
    }
}

// Make function available globally
window.showResetButtonIfNeeded = showResetButtonIfNeeded;