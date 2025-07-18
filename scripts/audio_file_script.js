import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js';
import ZoomPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/zoom.esm.js';
import RegionsPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.esm.js';
import TimelinePlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/timeline.esm.js';
import Minimap from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/minimap.esm.js';

// --- Global Variables ---
const regions = RegionsPlugin.create();
let timeline;
let timelineItems;
let lastWaveformZoom = 100;
let mousePosition = null;
let zoomTimeout = null;
let wavesurfer;
let activeRegion = null;
let loop = false; // This variable seems unused in the provided code
const BASE_ZOOM = 100;
let segments = [];
let isTranscribing = false;
let addedSegmentKeysTable = new Set();
let addedSegmentKeys = new Set(); // For timeline items
let colorIndex = 0; // Track current color index for regions

const colors = [
    "rgba(186, 233, 255, 0.5)",
    "rgba(201, 247, 210, 0.5)",
    "rgba(248, 235, 185, 0.5)",
    "rgba(194, 192, 255, 0.5)",
];

// --- Utility Functions ---

/**
 * Formats time in seconds to HH:mm:ss string.
 * @param {number} seconds - The time in seconds.
 * @returns {string} Formatted time string.
 */
function formatTime(seconds) {
    const date = new Date(seconds * 1000);
    return date.toISOString().substr(11, 8);
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
 * Resets all UI elements and data related to audio and transcription.
 */
function resetAllData() {
    document.getElementById('output').textContent = '';

    const segmentsTableBody = document.getElementById('segments-table-body-upload');
    if (segmentsTableBody) {
        document.getElementById('no-file-message-table-upload').style.display = 'flex';
        segmentsTableBody.innerHTML = '';
    }

    if (timelineItems) {
        timelineItems.clear();
    }

    segments = [];
    regions.clearRegions();

    const rtfDisplay = document.getElementById('rtf-display');
    if (rtfDisplay) {
        rtfDisplay.remove();
    }

    addedSegmentKeys = new Set();
    addedSegmentKeysTable = new Set();
    colorIndex = 0;

    document.getElementById('progress_upload').textContent = '00:00:00';

    const transcriptBtn = document.getElementById('start-trasncript-btn');
    if (transcriptBtn) {
        transcriptBtn.textContent = 'Start Transcription';
        transcriptBtn.disabled = false;
    }

    const playBtnImg = document.getElementById('play-upload-recording');
    if (playBtnImg) {
        playBtnImg.src = 'static/assets/play_black_icon_48.png';
    }

    if (timeline) {
        timeline.setCustomTime(0, 'cursor');
        timeline.moveTo(0, { animation: false });
    }

    if (wavesurfer) {
        wavesurfer.stop();
        wavesurfer.empty();
        wavesurfer.destroy();
        document.getElementById('waveform').style.display = 'none';
        document.getElementById('no-file-message').style.display = 'flex';
    }

    const fileInput = document.getElementById('audio_file');
    if (fileInput) {
        fileInput.value = '';
    }
}

// --- WaveSurfer Initialization and Event Handlers ---

/**
 * Creates and initializes the WaveSurfer instance with plugins.
 */
const createWaveSurfer = () => {
    wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#E96C64',
        progressColor: '#F8A05D',
        minPxPerSec: 1,
        plugins: [
            regions,
            TimelinePlugin.create(),
            Minimap.create({
                height: 30,
                waveColor: '#E96C64',
                progressColor: '#F8A05D',
            })
        ],
        barGap: 0,
        barHeight: 3,
        barWidth: 2,
        barRadius: 10,
    });

    wavesurfer.registerPlugin(
        ZoomPlugin.create({
            scale: 0.2,
            maxZoom: 1000,
        })
    );

    // Region event listeners
    regions.on('region-in', (region) => {
        activeRegion = region;
    });

    regions.on('region-out', (region) => {
        if (activeRegion === region) {
            if (loop) { // 'loop' variable is currently not used
                region.play();
            } else {
                activeRegion = null;
            }
        }
    });

    regions.on('region-clicked', (region, e) => {
        e.stopPropagation();
        activeRegion = region;
        region.play(true);
        const img = document.getElementById('play-upload-recording'); // Changed from '.play_btn'
        img.src = 'static/assets/pause_black_icon_48.png';
    });

    wavesurfer.on('interaction', () => {
        activeRegion = null;
    });

    // Waveform and Timeline zoom synchronization
    wavesurfer.on('zoom', (minPxPerSec) => {
        const zoomFactor = minPxPerSec / BASE_ZOOM;
        const zoomLabel = `${zoomFactor.toFixed(1)}x`;
        document.querySelector('.zoom_level').textContent = zoomLabel;

        if (timeline && Math.abs(minPxPerSec - lastWaveformZoom) > 5) {
            if (zoomTimeout) {
                clearTimeout(zoomTimeout);
            }

            zoomTimeout = setTimeout(() => {
                const zoomRatio = minPxPerSec / lastWaveformZoom;
                const currentWindow = timeline.getWindow();
                const currentCenter = mousePosition?.time ? mousePosition.time * 1000 :
                                    (currentWindow.start.getTime() + currentWindow.end.getTime()) / 2;
                const currentDuration = currentWindow.end.getTime() - currentWindow.start.getTime();
                const newDuration = currentDuration / zoomRatio;
                const newStart = currentCenter - (newDuration / 2);
                const newEnd = currentCenter + (newDuration / 2);

                timeline.setWindow(new Date(newStart), new Date(newEnd), { animation: false });
                lastWaveformZoom = minPxPerSec;
            }, 100);
        }
    });

    // Play/Pause event handlers
    wavesurfer.on('play', () => {
        const img = document.getElementById('play-upload-recording');
        img.src = 'static/assets/pause_black_icon_48.png';
    });

    wavesurfer.on('pause', () => {
        const img = document.getElementById('play-upload-recording');
        img.src = 'static/assets/play_black_icon_48.png';
    });

    wavesurfer.on('audioprocess', onAudioProcess);

    wavesurfer.on('finish', () => {
        const img = document.getElementById('play-upload-recording'); // Changed from '.play_btn'
        img.src = 'static/assets/play_black_icon_48.png';
    });

    // Sync timeline cursor with WaveSurfer playback on click
    wavesurfer.on('click', (e) => {
        const currentTime = wavesurfer.getCurrentTime();
        timeline.setCustomTime(currentTime * 1000, 'cursor');
        timeline.moveTo(currentTime * 1000, { animation: false });
        const formattedTime = new Date(currentTime * 1000).toISOString().substr(11, 8);
        document.getElementById('progress_upload').textContent = formattedTime;
    });
};

/**
 * Updates the audio progress display and timeline cursor.
 * @param {number} currentTime - The current playback time in seconds.
 */
function onAudioProcess(currentTime) {
    const formattedTime = new Date(currentTime * 1000).toISOString().substr(11, 8);
    document.getElementById('progress_upload').textContent = formattedTime;
    if (window.timeline2 && isTranscribing === false) { // Assuming window.timeline2 is the correct reference
        window.timeline2.setCustomTime(currentTime * 1000, 'cursor');
        window.timeline2.moveTo(currentTime * 1000, { animation: false });
    }
}

/**
 * Tracks mouse position on waveform for targeted zooming.
 * @param {MouseEvent} e - The mouse event.
 */
document.getElementById('waveform').addEventListener('mousemove', (e) => {
    if (wavesurfer && wavesurfer.getDuration && wavesurfer.getDuration() > 0) {
        if (!mousePosition || Date.now() - (mousePosition.lastUpdate || 0) > 50) {
            const rect = e.currentTarget.getBoundingClientRect();
            const relativeX = (e.clientX - rect.left) / rect.width;
            mousePosition = {
                time: relativeX * wavesurfer.getDuration(),
                lastUpdate: Date.now()
            };
        }
    }
});

// --- UI Event Listeners ---

document.getElementById('play-upload-recording-container').addEventListener('click', () => {
    const img = document.getElementById('play-upload-recording');
    if (wavesurfer.isPlaying()) {
        wavesurfer.pause();
        img.src = 'static/assets/play_black_icon_48.png';
    } else {
        wavesurfer.play();
        img.src = 'static/assets/pause_black_icon_48.png';
    }
});

document.getElementById('resetDataBtn').addEventListener('click', () => {
    resetAllData();
});

document.querySelector('.speaker').addEventListener('click', () => {
    const currentVolume = wavesurfer.getVolume();
    const speakerImg = document.querySelector('.speaker img');

    if (currentVolume > 0) {
        wavesurfer.setVolume(0);
        speakerImg.src = 'static/assets/speaker_black_icon_48.png';
    } else {
        wavesurfer.setVolume(1);
        speakerImg.src = 'static/assets/mute_black_icon_48.png';
    }
});

document.getElementById('audio_file').addEventListener('change', function(e) {
    const file = e.target.files[0];
    const noFileMessage = document.getElementById('no-file-message');
    const waveformContainer = document.getElementById('waveform');

    if (file) {
        const transcriptBtn = document.getElementById('start-trasncript-btn');
        transcriptBtn.disabled = false;

        if (noFileMessage) {
            noFileMessage.style.display = 'none';
            waveformContainer.style.display = 'block';
        }

        const url = URL.createObjectURL(file);
        createWaveSurfer();
        wavesurfer.load(url);
    } else {
        if (noFileMessage) {
            noFileMessage.style.display = 'flex';
        }
    }

    wavesurfer.once('ready', () => {
        const duration = wavesurfer.getDuration();
        const formattedDuration = new Date(duration * 1000).toISOString().substr(11, 8);
        document.querySelector('.total_audio_duration').textContent = formattedDuration;

        if (timeline) {
            timeline.setOptions({
                max: duration * 1000 + 1000,
            });
            timeline.moveTo(1);
            lastWaveformZoom = 100;
        }
    });
});

document.getElementById('start-trasncript-btn').addEventListener('click', function(e) {
    document.getElementById('start-trasncript-btn').textContent = 'Transcribing...';
    e.preventDefault();
    startTranscription();
});

// --- Transcription Related Functions ---

/**
 * Updates the segments table with new transcription segments.
 * @param {Array<Object>} newSegments - An array of new segment objects.
 */
function updateSegmentsTable(newSegments) {
    document.getElementById('no-file-message-table-upload').style.display = 'none';
    const segmentsTableBody = document.getElementById('segments-table-body-upload');

    newSegments.forEach(segment => {
        const key = `${segment.start}-${segment.end}-${segment.text}`;

        if (!addedSegmentKeysTable.has(key)) {
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
            addedSegmentKeysTable.add(key);

            const playButton = row.querySelector('.segment-btn-play');
            playButton.addEventListener('click', (e) => {
                const btn = e.currentTarget;
                const start = parseFloat(btn.getAttribute('data-start'));
                const end = parseFloat(btn.getAttribute('data-end'));
                playSegment(start, end);
            });
        }

        regions.addRegion({
            start: segment.start,
            end: segment.end,
            color: colors[colorIndex % colors.length],
            content: segment.text,
            drag: false,
            resize: false,
        });
        colorIndex++;
    });
}

/**
 * Plays a specific audio segment.
 * @param {number} start - The start time of the segment in seconds.
 * @param {number} end - The end time of the segment in seconds.
 */
function playSegment(start, end) {
    if (!wavesurfer) {
        console.error('No waveform available for playback');
        return;
    }

    if (!isFinite(start) || !isFinite(end) || start < 0 || end <= start) {
        console.error('Invalid segment times:', { start, end });
        return;
    }

    const duration = wavesurfer.getDuration();
    if (start >= duration) {
        console.error('Start time exceeds audio duration:', { start, duration });
        return;
    }

    try {
        if (window.currentUploadSegmentHandler) {
            wavesurfer.un('audioprocess', window.currentUploadSegmentHandler);
            window.currentUploadSegmentHandler = null;
        }

        if (wavesurfer.isPlaying()) {
            wavesurfer.pause();
        }

        wavesurfer.setTime(start);
        wavesurfer.play();

        const stopHandler = () => {
            const currentTime = wavesurfer.getCurrentTime();
            if (currentTime >= end || currentTime >= duration) {
                wavesurfer.pause();
                wavesurfer.un('audioprocess', stopHandler);
                window.currentUploadSegmentHandler = null;
            }
        };

        window.currentUploadSegmentHandler = stopHandler;
        wavesurfer.on('audioprocess', stopHandler);

    } catch (error) {
        console.error('Error playing upload segment:', error, { start, end });
    }
}

/**
 * Handles incoming transcription event data.
 * @param {Object} data - The transcription data containing full_transcript and segments.
 */
function handleTranscriptionEvent(data) {
    document.getElementById('output').textContent = data.full_transcript;
    if (data.segments && Array.isArray(data.segments)) {
        segments.push(...data.segments);
        updateSegmentsTable(data.segments);
    }
}

/**
 * Displays Real-Time Factor (RTF) performance metrics.
 * @param {number} rtf - The Real-Time Factor.
 * @param {number} processingTime - The transcription processing time in seconds.
 * @param {number} audioDuration - The audio duration in seconds.
 */
function displayRTF(rtf, processingTime, audioDuration) {
    let rtfDisplay = document.getElementById('rtf-display');
    if (!rtfDisplay) {
        rtfDisplay = document.createElement('div');
        rtfDisplay.id = 'rtf-display';
        rtfDisplay.style.cssText = `
            margin-top: 15px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 6px;
            border: 1px solid #e9ecef;
            font-family: monospace;
            font-size: 14px;
        `;
        const transcriptContainer = document.querySelector('#upload .transcript-container');
        if (transcriptContainer && transcriptContainer.parentNode) {
            transcriptContainer.parentNode.insertBefore(rtfDisplay, transcriptContainer.nextSibling);
        }
    }

    const rtfColor = rtf < 1.0 ? '#28a745' : rtf < 2.0 ? '#ffc107' : '#dc3545';
    const rtfStatus = rtf < 1.0 ? 'Faster than real-time' : rtf < 2.0 ? 'Near real-time' : 'Slower than real-time';

    rtfDisplay.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <strong>Transcription Performance</strong>
            <span style="color: ${rtfColor}; font-weight: bold;">RTF: ${rtf.toFixed(3)}</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; font-size: 12px;">
            <div>Audio Duration: ${audioDuration.toFixed(2)}s</div>
            <div>Processing Time: ${processingTime.toFixed(2)}s</div>
            <div style="color: ${rtfColor}; text-align: end">${rtfStatus}</div>
        </div>
    `;
}

/**
 * Initiates the audio transcription process by sending the file to the server.
 */
async function startTranscription() {
    const fileInput = document.getElementById('audio_file');
    const file = fileInput.files[0];
    if (!file) return;

    const transcriptionStartTime = performance.now();
    const audioDuration = wavesurfer.getDuration();

    const formData = new FormData();
    formData.append('audio_file', file);

    addedSegmentKeys = new Set();
    addedSegmentKeysTable = new Set();
    document.getElementById('segments-table-body-upload').innerHTML = '';
    if (timelineItems) timelineItems.clear();

    const response = await fetch('/upload_and_transcribe', {
        method: 'POST',
        body: formData,
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    wavesurfer.play(); // Start playback

    while (true) {
        isTranscribing = true;
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const match = chunk.match(/data:\s*(\{.*\})/);
        if (match) {
            try {
                const data = JSON.parse(match[1]);
                console.log('Received data:', data);
                handleTranscriptionEvent(data);
            } catch (e) {
                console.error('JSON parse error:', e, match[1]);
            }
        }
    }

    const transcriptionEndTime = performance.now();
    const processingTimeMs = transcriptionEndTime - transcriptionStartTime;
    const processingTimeSeconds = processingTimeMs / 1000;
    const rtf = processingTimeSeconds / audioDuration;

    displayRTF(rtf, processingTimeSeconds, audioDuration);
    updateTimeline(segments);
    isTranscribing = false;
    document.getElementById('start-trasncript-btn').textContent = 'Start Transcription';
}

// --- Timeline Initialization and Updates ---

/**
 * Initializes the Vis.js Timeline for transcript visualization.
 */
function initTimeline() {
    const container = document.getElementById('transcript-timeline-upload');
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
        selectable: true,
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
        zoomMin: 100,
        zoomMax: 10000,
        zoomFriction: 5,
        height: '180px'
    };
    timeline = new vis.Timeline(container, timelineItems, options);
    timeline.addCustomTime(0, 'cursor');
    window.timeline2 = timeline; // Expose timeline globally for onAudioProcess
}

/**
 * Updates the Vis.js Timeline with new transcription segments.
 * @param {Array<Object>} newSegments - An array of new segment objects.
 */
function updateTimeline(newSegments) {
    console.log('Updating timeline with new segments:', typeof(newSegments), newSegments);
    if (!timelineItems) return;

    newSegments.forEach((segment) => {
        const key = `${segment.start}-${segment.end}`;
        if (!addedSegmentKeys.has(key)) {
            timelineItems.add({
                id: key,
                content: segment.text,
                start: segment.start * 1000,
                end: segment.end * 1000,
            });
            addedSegmentKeys.add(key);
        }
    });
}

// --- DOM Content Loaded Event ---
document.addEventListener('DOMContentLoaded', initTimeline);
