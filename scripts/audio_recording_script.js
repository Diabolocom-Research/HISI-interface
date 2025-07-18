// --- Global Variables and DOM References ---
let timeline;
let timelineItems;
let recordingTimer = null;
let recordingStartTime = null;
let recordingDuration = 0;
let peerConnection;
let webrtc_id;
let eventSource;
let historicalTranscript = "";
let transcriptionStartTime = null;
let currentSegments = [];

// DOM References
const progressElement = document.getElementById('progress');
const viewModelSelection = document.getElementById('view-model-selection');
const viewLoading = document.getElementById('view-loading');
const viewTranscription = document.getElementById('view-transcription');
const loadModelButton = document.getElementById('load-model-button');
const configTextarea = document.getElementById('config-json');
const startButton = document.getElementById('start-button');
const startButtonText = document.querySelector('.start-button-txt');
const micImg = document.querySelector('.microphone-icon');
const transcriptTextElement = document.getElementById('transcript-text');
const segmentsTableBody = document.getElementById('segments-table-body-recording');
const resetBtnContainer = document.getElementById('reset-recording-btn-container');

// Default Configuration for the Text Area
const defaultConfig = {
    "model": "tiny",
    "lan": "auto",
    "task": "transcribe",
    "backend": "whisper_timestamped",
    "min_chunk_size": 1.0,
    "buffer_trimming": "segment",
    "buffer_trimming_sec": 10.0
};

// --- Utility Functions ---

/**
 * Formats time in seconds to MM:SS string.
 * @param {number} seconds - The time in seconds.
 * @returns {string} Formatted time string.
 */
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Formats duration in seconds to Hh Mm S.Sms string.
 * @param {number} seconds - The duration in seconds.
 * @returns {string} Formatted duration string.
 */
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

/**
 * Generates a random color from a predefined list.
 * @returns {string} RGBA color string.
 */
function getRandomColor() {
    const colors = [
        "rgba(49, 151, 198, 0.9)",
        "rgba(56, 177, 80, 0.9)",
        "rgba(184, 156, 45, 0.9)",
        "rgba(95, 92, 185, 0.9)"
    ];
    return colors[Math.floor(Math.random() * colors.length)];
}

// --- Recording Timer Functions ---

/**
 * Starts the recording timer, updating the display every second.
 */
function startRecordingTimer() {
    const timerElement = document.getElementById('recording-timer');
    const recordingStatusContainer = document.querySelector('#recording-status');

    recordingStatusContainer.style.display = 'flex';
    recordingStartTime = Date.now();

    recordingTimer = setInterval(() => {
        recordingDuration = Math.floor((Date.now() - recordingStartTime) / 1000);
        timerElement.textContent = formatTime(recordingDuration);
    }, 1000);
}

/**
 * Stops the recording timer and hides the status display.
 */
function stopRecordingTimer() {
    const recordingStatusContainer = document.querySelector('#recording-status');

    if (recordingTimer) {
        clearInterval(recordingTimer);
        recordingTimer = null;
    }

    recordingStatusContainer.style.display = 'none';
    // startButton.style.display = 'flex'; // This line should be handled by updateButtonState
    recordingDuration = 0;
}

// --- UI State Management ---

/**
 * Updates the state and appearance of UI buttons and elements based on connection status.
 */
function updateButtonState() {
    if (!peerConnection || peerConnection.connectionState === "closed" || peerConnection.connectionState === "failed") {
        const hasTranscript = transcriptTextElement && transcriptTextElement.textContent.trim() !== 'Speak to see transcript...';
        const hasSegments = currentSegments && currentSegments.length > 0;
        const hasRecording = window.lastRecordedWaveSurfer;

        if (hasTranscript || hasSegments || hasRecording) {
            startButtonText.textContent = 'Recording Complete';
            startButton.style.opacity = '0.5';
            startButton.style.cursor = 'not-allowed';
            startButton.disabled = true;
        } else {
            startButtonText.textContent = 'Start Recording';
            startButton.style.opacity = '1';
            startButton.style.cursor = 'pointer';
            startButton.disabled = false;
        }

        micImg.src = 'static/assets/microphone_icon.png';
        stopRecordingTimer();

        if (resetBtnContainer) {
            resetBtnContainer.disabled = false;
            showResetButtonIfNeeded();
        }
    } else if (peerConnection.connectionState === "connecting" || peerConnection.connectionState === "new") {
        startButtonText.textContent = 'Connecting...';
        startButton.disabled = false;
        startButton.style.opacity = '1';
        startButton.style.cursor = 'pointer';

        if (resetBtnContainer) {
            resetBtnContainer.disabled = true;
        }
    } else if (peerConnection.connectionState === "connected") {
        startButtonText.textContent = 'Stop Recording';
        micImg.src = 'static/assets/stop_recording.png';
        startRecordingTimer();
        startButton.disabled = false;
        startButton.style.opacity = '1';
        startButton.style.cursor = 'pointer';

        if (resetBtnContainer) {
            resetBtnContainer.disabled = true;
        }
    }
}

/**
 * Switches the active view in the application.
 * @param {string} viewId - The ID of the view to switch to.
 */
function switchView(viewId) {
    viewModelSelection.classList.add('view-hidden');
    viewLoading.classList.add('view-hidden');
    viewTranscription.classList.add('view-hidden');
    document.getElementById(viewId)?.classList.remove('view-hidden');
    document.body.style.alignItems = (viewId === 'view-transcription') ? 'normal' : 'center';
    document.body.style.justifyContent = (viewId === 'view-transcription') ? 'normal' : 'center';
}

/**
 * Displays or hides the reset button based on whether there's recorded data.
 */
function showResetButtonIfNeeded() {
    if (!resetBtnContainer) return;

    const hasTranscript = transcriptTextElement && transcriptTextElement.textContent.trim() !== 'Speak to see transcript...';
    const hasSegments = currentSegments && currentSegments.length > 0;
    const hasRecording = window.lastRecordedWaveSurfer;

    if (hasTranscript || hasSegments || hasRecording) {
        resetBtnContainer.style.display = 'flex';
    } else {
        resetBtnContainer.style.display = 'none';
    }
}

/**
 * Resets all UI elements and data related to recording and transcription.
 */
function resetRecordingData() {
    if (peerConnection) {
        stop(); // Stop any active recording
    }

    transcriptTextElement.textContent = 'Speak to see transcript...';

    if (segmentsTableBody) {
        document.getElementById('no-file-message-table-record').style.display = 'flex';
        segmentsTableBody.innerHTML = '';
    }

    if (timelineItems) {
        timelineItems.clear();
        timelineItems = new vis.DataSet([]); // Recreate the DataSet
    }

    if (timeline) {
        timeline.setCustomTime(0, 'cursor');
        timeline.moveTo(0, { animation: false });
        timeline.setOptions({ max: 10000, start: 0, end: 10000 });
        timeline.setItems(timelineItems); // Ensure timeline uses fresh DataSet
    }

    if (window.regions) {
        window.regions.clearRegions();
    }

    if (window.currentSegmentHandler && window.lastRecordedWaveSurfer) {
        window.lastRecordedWaveSurfer.un('audioprocess', window.currentSegmentHandler);
        window.currentSegmentHandler = null;
    }

    if (window.lastRecordedWaveSurfer) {
        window.lastRecordedWaveSurfer.destroy();
        window.lastRecordedWaveSurfer = null;
        document.getElementById('recordings').style.display = 'none';
        document.getElementById('no-waveform-message-playback').style.display = 'flex';
    }

    if (window.wavesurfer) { // This might be a reference to a different WaveSurfer instance
        window.wavesurfer.empty();
    }

    const recordingsContainer = document.getElementById('recordings');
    if (recordingsContainer) {
        recordingsContainer.innerHTML = '';
    }

    const controlsContainer = document.querySelector('.controls-container');
    if (controlsContainer) {
        controlsContainer.style.display = 'none';
    }

    document.querySelector('.audio_duration').textContent = '00:00';
    document.querySelector('.total_audio_duration_record').textContent = '00:00';

    const playBtn = document.querySelector('.play_btn');
    if (playBtn) {
        playBtn.src = 'static/assets/play_black_icon_48.png';
    }

    historicalTranscript = '';
    currentSegments = [];
    recordingDuration = 0;
    transcriptionStartTime = null;

    const rtfDisplay = document.getElementById('rtf-display-recording');
    if (rtfDisplay) {
        rtfDisplay.remove();
    }

    stopRecordingTimer();

    document.getElementById('mic').style.display = 'none';
    document.getElementById('no-recording-message').style.display = 'flex';

    updateButtonState(); // Update state after reset
}

// --- Timeline Functions ---

/**
 * Initializes the Vis.js Timeline for transcript visualization.
 */
function initTimeline() {
    const container = document.getElementById('transcript-timeline');
    timelineItems = new vis.DataSet([]);
    const options = {
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
    timeline.addCustomTime(0, 'cursor');
    window.timeline = timeline; // Expose timeline globally
}

/**
 * Updates the Vis.js Timeline with new transcription segments.
 * @param {Array<Object>} segments - An array of segment objects.
 */
function updateTimeline(segments) {
    if (!timelineItems || !timeline) {
        console.warn("Timeline not initialized properly. Cannot update.");
        return;
    }

    timelineItems.clear(); // Clear existing items

    if (segments && segments.length > 0) {
        segments.forEach((segment, idx) => {
            timelineItems.add({
                id: idx,
                content: segment.text,
                start: segment.start * 1000,
                end: segment.end * 1000,
            });
        });

        const lastSegment = segments[segments.length - 1];
        if (lastSegment) {
            timeline.setOptions({
                max: lastSegment.end * 1000 + 1000,
            });
        }
        timeline.fit(); // Adjust timeline to show all content
    } else {
        timeline.setOptions({ max: 10000 });
        timeline.moveTo(0, { animation: false });
    }
    window.segments = segments; // Store segments globally
}

// --- WebRTC and Transcription Logic ---

/**
 * Loads the transcription model by sending a configuration to the server.
 */
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
        window.createWaveSurfer(); // Assuming this is defined elsewhere and needed for playback
    } catch (err) {
        alert(`Error loading model: ${err.message}`);
        switchView('view-model-selection');
    }
}

/**
 * Sets up the WebRTC connection for real-time audio streaming and transcription.
 */
async function setupWebRTC() {
    startButtonText.textContent = 'Connecting...';
    document.getElementById('mic').style.display = 'block';
    document.getElementById('no-recording-message').style.display = 'none';

    try {
        const config = window.RTC_CONFIGURATION; // Assuming RTC_CONFIGURATION is defined globally
        peerConnection = new RTCPeerConnection(config);

        peerConnection.addEventListener('connectionstatechange', () => {
            if (peerConnection.connectionState === 'connected') {
                transcriptTextElement.textContent = historicalTranscript;
                segmentsTableBody.innerHTML = '';
                transcriptionStartTime = performance.now();
            }
            updateButtonState();
        });

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach(track => peerConnection.addTrack(track, stream));

        const dataChannel = peerConnection.createDataChannel('text');
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);

        await new Promise((resolve) => {
            if (peerConnection.iceGatheringState === "complete") {
                resolve();
            } else {
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
            stop(); // Clean up on error
        };
    } catch (err) {
        console.error('Failed to establish connection.', err);
        alert('Failed to establish connection. Check console for details.');
        stop();
    }
}

/**
 * Handles incoming transcription updates from the server (SSE).
 * @param {string} data - JSON string containing transcription payload.
 */
function handleServerUpdate(data) {
    try {
        const payload = JSON.parse(data);
        transcriptTextElement.textContent = historicalTranscript + (payload.full_transcript || "");
        document.getElementById('transcript-container').scrollTop = document.getElementById('transcript-container').scrollHeight;

        const segments = payload.segments || [];
        if (segments.length > 0) {
            document.getElementById('no-file-message-table-record').style.display = 'none';
        }
        currentSegments = segments; // Store segments globally

        // Update segments table
        segmentsTableBody.innerHTML = ''; // Clear existing
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
                            <img src="static/assets/play_black_icon_48.png" alt="Play">
                        </button>
                    </div>
                </td>
            `;
            segmentsTableBody.appendChild(row);

            const playButton = row.querySelector('.segment-btn-play');
            playButton.addEventListener('click', (e) => {
                const btn = e.currentTarget;
                const start = parseFloat(btn.getAttribute('data-start'));
                const end = parseFloat(btn.getAttribute('data-end'));
                playSegment(start, end);
            });
        });

        updateTimeline(segments);
        showResetButtonIfNeeded();

    } catch (e) {
        console.error("Failed to parse server data:", data, e);
    }
}

/**
 * Plays a specific audio segment from the recorded waveform.
 * @param {number} start - The start time of the segment in seconds.
 * @param {number} end - The end time of the segment in seconds.
 */
function playSegment(start, end) {
    if (!window.lastRecordedWaveSurfer) {
        console.error('No recorded waveform available for playback');
        return;
    }

    if (!isFinite(start) || !isFinite(end) || start < 0 || end <= start) {
        console.error('Invalid segment times:', { start, end });
        return;
    }

    const duration = window.lastRecordedWaveSurfer.getDuration();
    if (start >= duration) {
        console.error('Start time exceeds audio duration:', { start, duration });
        return;
    }

    try {
        if (window.currentSegmentHandler) {
            window.lastRecordedWaveSurfer.un('audioprocess', window.currentSegmentHandler);
            window.currentSegmentHandler = null;
        }

        if (window.lastRecordedWaveSurfer.isPlaying()) {
            window.lastRecordedWaveSurfer.pause();
        }

        window.lastRecordedWaveSurfer.setTime(start);
        window.lastRecordedWaveSurfer.play();

        const stopHandler = () => {
            const currentTime = window.lastRecordedWaveSurfer.getCurrentTime();
            if (currentTime >= end || currentTime >= duration) {
                window.lastRecordedWaveSurfer.pause();
                window.lastRecordedWaveSurfer.un('audioprocess', stopHandler);
                window.currentSegmentHandler = null;
            }
        };

        window.currentSegmentHandler = stopHandler;
        window.lastRecordedWaveSurfer.on('audioprocess', stopHandler);

    } catch (error) {
        console.error('Error playing segment:', error, { start, end });
    }
}

/**
 * Adds regions (segments) to the recorded waveform visualization.
 * This relies on `window.lastRecordedWaveSurfer` and `window.regions` being available.
 */
function addRegionsToRecordedWaveform() {
    if (!window.lastRecordedWaveSurfer || !currentSegments || currentSegments.length === 0) {
        console.warn("No recorded waveform or segments available to add regions.");
        return;
    }

    if (!window.lastRecordedWaveSurfer.isReady) {
        window.lastRecordedWaveSurfer.once('ready', () => {
            addRegionsToRecordedWaveform();
        });
        return;
    }

    if (window.regions) {
        window.regions.clearRegions();
    }

    currentSegments.forEach((segment) => {
        try {
            window.regions.addRegion({
                start: segment.start,
                end: segment.end,
                content: segment.text,
                color: getRandomColor(),
                drag: false,
                resize: false,
            });
        } catch (error) {
            console.error("Failed to add region for segment", error);
        }
    });
}

/**
 * Stops the WebRTC connection and finalizes the recording process.
 */
function stop() {
    if (peerConnection) {
        peerConnection.getSenders().forEach(sender => sender.track?.stop());
        peerConnection.close();
        peerConnection = null;
    }

    if (eventSource) {
        eventSource.close();
        eventSource = null; // Clear reference
    }

    if (transcriptTextElement && transcriptTextElement.textContent.trim() !== historicalTranscript.trim()) {
        historicalTranscript = transcriptTextElement.textContent + "\n\n";
    }

    if (transcriptionStartTime && recordingDuration > 0) {
        const transcriptionEndTime = performance.now();
        const processingTimeMs = transcriptionEndTime - transcriptionStartTime;
        const processingTimeSeconds = processingTimeMs / 1000;
        const rtf = processingTimeSeconds / recordingDuration;
        displayRecordingRTF(rtf, processingTimeSeconds, recordingDuration);
    }
    updateButtonState();
}

/**
 * Displays Real-Time Factor (RTF) performance metrics for recording.
 * @param {number} rtf - The Real-Time Factor.
 * @param {number} processingTime - The transcription processing time in seconds.
 * @param {number} audioDuration - The audio duration in seconds.
 */
function displayRecordingRTF(rtf, processingTime, audioDuration) {
    let rtfDisplay = document.getElementById('rtf-display-recording');
    if (!rtfDisplay) {
        rtfDisplay = document.createElement('div');
        rtfDisplay.id = 'rtf-display-recording';
        rtfDisplay.style.cssText = `
            margin-top: 15px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 6px;
            border: 1px solid #e9ecef;
            font-family: monospace;
            font-size: 14px;
        `;
        const transcriptContainer = document.getElementById('transcript-container-recording');
        if (transcriptContainer && transcriptContainer.parentNode) {
            transcriptContainer.parentNode.insertBefore(rtfDisplay, transcriptContainer.nextSibling);
        }
    }

    const rtfColor = rtf < 1.0 ? '#28a745' : rtf < 2.0 ? '#ffc107' : '#dc3545';
    const rtfStatus = rtf < 1.0 ? 'Faster than real-time' : rtf < 2.0 ? 'Near real-time' : 'Slower than real-time';

    rtfDisplay.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <strong>Real-Time Transcription Performance</strong>
            <span style="color: ${rtfColor}; font-weight: bold;">RTF: ${rtf.toFixed(3)}</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; font-size: 12px;">
            <div>Audio Duration: ${audioDuration.toFixed(2)}s</div>
            <div>Processing Time: ${processingTime.toFixed(2)}s</div>
            <div style="color: ${rtfColor}; text-align: end">${rtfStatus}</div>
        </div>
    `;
}

// --- Event Listeners and Initialization ---

// Initialize default config in textarea
configTextarea.value = JSON.stringify(defaultConfig, null, 2);

// Load Model Button
loadModelButton.addEventListener('click', loadModel);

// Start/Stop Recording Button
startButton.addEventListener('click', () => {
    if (startButton.disabled) {
        return;
    }
    if (!peerConnection || peerConnection.connectionState === "closed") {
        setupWebRTC();
    } else {
        stop();
    }
});

// Speaker/Mute Button for Recorded Playback
document.querySelector('.speaker-recording').addEventListener('click', () => {
    if (!window.lastRecordedWaveSurfer) {
        console.warn("No recorded waveform to control volume.");
        return;
    }
    const currentVolume = window.lastRecordedWaveSurfer.getVolume();
    const speakerImg = document.querySelector('.speaker-recording img');

    if (currentVolume > 0) {
        window.lastRecordedWaveSurfer.setVolume(0);
        speakerImg.src = 'static/assets/speaker_black_icon_48.png';
    } else {
        window.lastRecordedWaveSurfer.setVolume(1);
        speakerImg.src = 'static/assets/mute_black_icon_48.png';
    }
});

// Reset Button Event Listener
document.addEventListener('DOMContentLoaded', () => {
    if (resetBtnContainer) {
        resetBtnContainer.addEventListener('click', () => {
            resetRecordingData();
        });
    }
    initTimeline(); // Initialize timeline on DOMContentLoaded
    updateButtonState(); // Set initial button state
    showResetButtonIfNeeded(); // Set initial reset button visibility
});

window.showResetButtonIfNeeded = showResetButtonIfNeeded;
window.addRegionsToRecordedWaveform = addRegionsToRecordedWaveform;
