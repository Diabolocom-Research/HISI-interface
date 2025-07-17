import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js'
import ZoomPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/zoom.esm.js'
import RegionsPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.esm.js'
import TimelinePlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/timeline.esm.js'
import Minimap from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/minimap.esm.js'

const regions = RegionsPlugin.create()
let timeline, timelineItems;
let lastWaveformZoom = 100; 
let mousePosition = null; 
let zoomTimeout = null;
let wavesurfer;
let activeRegion = null
let loop = false
const BASE_ZOOM = 100;
let segments = [];
let isTranscribing = false;

const colors = [
    "rgba(186, 233, 255, 0.5)", 
    "rgba(201, 247, 210, 0.5)", 
    "rgba(248, 235, 185, 0.5)", 
    "rgba(194, 192, 255, 0.5)", 
]

// Reset all data and UI elements
function resetAllData() {
    const outputElement = document.getElementById('output');
    if (outputElement) {
        outputElement.textContent = '';
    }
    
    const segmentsTableBody = document.getElementById('segments-table-body-upload');
    if (segmentsTableBody) {
        document.getElementById('no-file-message-table-upload').style.display = 'flex';
        segmentsTableBody.innerHTML = '';
    }
    
    if (timelineItems) {
        timelineItems.clear();
    }
    
    if (segments) {
        segments = [];
    }
    
    if (regions) {
        regions.clearRegions();
    }
    
    // Clear RTF display
    const rtfDisplay = document.getElementById('rtf-display');
    if (rtfDisplay) {
        rtfDisplay.remove();
    }
    
    addedSegmentKeys = new Set();
    addedSegmentKeysTable = new Set(); 
    
    colorIndex = 0;
    
    const progressElement = document.getElementById('progress_upload');
    if (progressElement) {
        progressElement.textContent = '00:00:00';
    }
    
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
    
    // Clear the file input to allow re-uploading the same file
    const fileInput = document.getElementById('audio_file');
    if (fileInput) {
        fileInput.value = '';
    }
}

const createWaveSurfer = () => {
    wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#E96C64',
        progressColor: '#F8A05D',
        minPxPerSec: 1,
        plugins: [regions, TimelinePlugin.create(),
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
            scale: 0.2, // the amount of zoom per wheel step, e.g. 0.2 means a 20% magnification per scroll
            maxZoom: 1000, 
        })
    )

    regions.on('region-in', (region) => {
        activeRegion = region
    })

    regions.on('region-out', (region) => {
        if (activeRegion === region) {
            if (loop) {
                region.play()
            } else {
                activeRegion = null
            }
        }
    })

    regions.on('region-clicked', (region, e) => {
        e.stopPropagation() // prevent triggering a click on the waveform
        activeRegion = region
        region.play(true)
        const img = document.querySelector('.play_btn');
        img.src = 'static/assets/pause_black_icon_48.png';
    })

    wavesurfer.on('interaction', () => {
        activeRegion = null
    })

    // Waveform and Timeline zoom synchronization (debounced)
    wavesurfer.on('zoom', (minPxPerSec) => {
        const zoomFactor = minPxPerSec / BASE_ZOOM;
        const zoomLabel = `${zoomFactor.toFixed(1)}x`; 

        document.querySelector('.zoom_level').textContent = zoomLabel;
        if (timeline && Math.abs(minPxPerSec - lastWaveformZoom) > 5) { 

            if (zoomTimeout) {
                clearTimeout(zoomTimeout);
            }
            
            // Debounce zoom synchronization
            zoomTimeout = setTimeout(() => {            
                const zoomRatio = minPxPerSec / lastWaveformZoom;
                
                // Get current timeline window
                const currentWindow = timeline.getWindow();
                const currentCenter = mousePosition?.time ? mousePosition.time * 1000 : 
                                    (currentWindow.start.getTime() + currentWindow.end.getTime()) / 2;
                const currentDuration = currentWindow.end.getTime() - currentWindow.start.getTime();
                
                // Calculate new window duration based on zoom ratio
                const newDuration = currentDuration / zoomRatio;
                
                // Calculate new window centered on mouse position or current center
                const newStart = currentCenter - (newDuration / 2);
                const newEnd = currentCenter + (newDuration / 2);
                
                // Apply the new window with smooth animation disabled for responsiveness
                timeline.setWindow(new Date(newStart), new Date(newEnd), { animation: false });
                
                // Update the last zoom level
                lastWaveformZoom = minPxPerSec;
            }, 100); // 100ms debounce
        }
    });

    wavesurfer.on('play', () => {
        const img = document.getElementById('play-upload-recording');
        img.src = 'static/assets/pause_black_icon_48.png';
        // onAudioProcess(wavesurfer.getCurrentTime());
    });

    wavesurfer.on('pause', () => {
        const img = document.getElementById('play-upload-recording');
        img.src = 'static/assets/play_black_icon_48.png';
    });

    wavesurfer.on('audioprocess', onAudioProcess);

    wavesurfer.on('finish', () => {
        const img = document.querySelector('.play_btn');
        img.src = 'static/assets/play_black_icon_48.png';
    });

    // Sync timeline cursor with WaveSurfer playback
    wavesurfer.on('click', (e) => {
        const currentTime = wavesurfer.getCurrentTime();
                
        // Set the cursor to the clicked position
        timeline.setCustomTime(currentTime * 1000, 'cursor');
        timeline.moveTo(currentTime * 1000, { animation: false });
        const formattedTime = new Date(currentTime * 1000).toISOString().substr(11, 8);
        document.getElementById('progress_upload').textContent = formattedTime;
    });
}

// Track mouse position on waveform for targeted zooming (debounced)
document.getElementById('waveform').addEventListener('mousemove', (e) => {
    if (wavesurfer && wavesurfer.getDuration && wavesurfer.getDuration() > 0) {
        // Throttle mouse position updates
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

document.getElementById('play-upload-recording-container').addEventListener('click', () => {
    const img = document.getElementById('play-upload-recording');
    if (wavesurfer.isPlaying()) {
        wavesurfer.pause();
        img.src = 'static/assets/play_black_icon_48.png'; 
    } else {
        wavesurfer.play();
        img.src = 'static/assets/pause_black_icon_48.png';
        // onAudioProcess(wavesurfer.getCurrentTime());
    }
});

function onAudioProcess(currentTime) {
    const formattedTime = new Date(currentTime * 1000).toISOString().substr(11, 8);
    document.getElementById('progress_upload').textContent = formattedTime;
    if (window.timeline2 && isTranscribing == false) {
        window.timeline2.setCustomTime(currentTime * 1000, 'cursor');
        window.timeline2.moveTo(currentTime * 1000, { animation: false });
    }
}

document.getElementById('resetDataBtn').addEventListener('click', function() {
    resetAllData();
})

document.querySelector('.speaker').addEventListener('click', () => {
    // Ensure audio is ready before toggling volum        
    const currentVolume = wavesurfer.getVolume();
    const speakerImg = document.querySelector('.speaker img');
    
    if (currentVolume > 0) {
        // Mute the audio
        wavesurfer.setVolume(0);
        speakerImg.src = 'static/assets/speaker_black_icon_48.png';
    } else {
        // Unmute the audio
        wavesurfer.setVolume(1);
        speakerImg.src = 'static/assets/mute_black_icon_48.png';
    }
});

document.getElementById('audio_file').addEventListener('change', function(e) {
    const file = e.target.files[0];
    const noFileMessage = document.getElementById('no-file-message');
    const waveformContainer = document.getElementById('waveform');
    
    if (file) {
        // Reset everything when a new file is uploaded
        const transcriptBtn = document.getElementById('start-trasncript-btn');   
        transcriptBtn.disabled = false; 

        // Hide the no-file message
        if (noFileMessage) {
            noFileMessage.style.display = 'none';
            waveformContainer.style.display = 'block';
        }
        
        const url = URL.createObjectURL(file);
        createWaveSurfer();
        wavesurfer.load(url);
    } else {
        // Show the no-file message if no file is selected
        if (noFileMessage) {
            noFileMessage.style.display = 'flex';
        }
    }
    
    // Get the audio duration after loading the file
    wavesurfer.once('ready', () => {
        const duration = wavesurfer.getDuration(); // duration in seconds

        // Set total audio duration in mm:ss format
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

function formatTime(seconds) {
    const date = new Date(seconds * 1000);
    return date.toISOString().substr(11, 8);
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

// --- UPLOAD FILE FOR REAL-TIME TRANSCRIPTION LOGIC ---
let addedSegmentKeysTable = new Set(); 

function updateSegmentsTable(segments) {
    document.getElementById('no-file-message-table-upload').style.display = 'none';
    const segmentsTableBody = document.getElementById('segments-table-body-upload');
    
    segments.forEach(segment => {
        // Create a unique key for each segment
        const key = `${segment.start}-${segment.end}-${segment.text}`;
        
        // Only add if this segment hasn't been added yet
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
            
            // Add event listener for this specific button
            const playButton = row.querySelector('.segment-btn-play');
            playButton.addEventListener('click', (e) => {
                // Find the button element in case the img was clicked
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

function playSegment(start, end) {
    // Validate wavesurfer and segment times
    if (!wavesurfer) {
        console.error('No waveform available for playback');
        return;
    }
    
    // Check if start and end are valid finite numbers
    if (!isFinite(start) || !isFinite(end) || start < 0 || end <= start) {
        console.error('Invalid segment times:', { start, end });
        return;
    }
    
    // Get the duration of the audio to validate against
    const duration = wavesurfer.getDuration();
    if (start >= duration) {
        console.error('Start time exceeds audio duration:', { start, duration });
        return;
    }
    
    try {        
        // Clean up any existing segment playback handlers
        if (window.currentUploadSegmentHandler) {
            wavesurfer.un('audioprocess', window.currentUploadSegmentHandler);
            window.currentUploadSegmentHandler = null;
        }
        
        // Stop current playback if any
        if (wavesurfer.isPlaying()) {
            wavesurfer.pause();
        }
        
        // Set position and start playing
        wavesurfer.setTime(start);
        wavesurfer.play();
        
        // Create new stop handler for this segment
        const stopHandler = () => {
            const currentTime = wavesurfer.getCurrentTime();
            if (currentTime >= end || currentTime >= duration) {
                wavesurfer.pause();
                wavesurfer.un('audioprocess', stopHandler);
                window.currentUploadSegmentHandler = null;
            }
        };
        
        // Store reference to current handler for cleanup
        window.currentUploadSegmentHandler = stopHandler;
        wavesurfer.on('audioprocess', stopHandler);
        
    } catch (error) {
        console.error('Error playing upload segment:', error, { start, end });
    }
}

function handleTranscriptionEvent(data) {
    // 1. Update the output element with the full transcript
    document.getElementById('output').textContent = data.full_transcript;

    // 2. Update the timeline with the segments
    if (data.segments && Array.isArray(data.segments)) {
        segments.push(...data.segments);
        updateSegmentsTable(data.segments);
    }    
}

function displayRTF(rtf, processingTime, audioDuration) {
    // Create or update RTF display in the transcript container
    let rtfDisplay = document.getElementById('rtf-display');
    if (!rtfDisplay) {
        // Create RTF display element
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
        
        // Insert after the transcript container
        const transcriptContainer = document.querySelector('#upload .transcript-container');
        if (transcriptContainer && transcriptContainer.parentNode) {
            transcriptContainer.parentNode.insertBefore(rtfDisplay, transcriptContainer.nextSibling);
        }
    }
    
    // Format the RTF display
    const rtfColor = rtf < 1.0 ? '#28a745' : rtf < 2.0 ? '#ffc107' : '#dc3545'; // Green, Yellow, Red
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

async function startTranscription() {    
    const fileInput = document.getElementById('audio_file');
    const file = fileInput.files[0];
    if (!file) return;

    // Record start time for RTF calculation
    const transcriptionStartTime = performance.now();
    const audioDuration = wavesurfer.getDuration(); // Get audio duration in seconds

    const formData = new FormData();
    formData.append('audio_file', file); // field name must match FastAPI

    addedSegmentKeys = new Set();
    addedSegmentKeysTable = new Set(); // Reset table tracking
    document.getElementById('segments-table-body-upload').innerHTML = '';
    if (timelineItems) timelineItems.clear();

    const response = await fetch('/upload_and_transcribe', {
        method: 'POST',
        body: formData,
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    wavesurfer.play();

    while (true) {
        isTranscribing = true;
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        // Extract JSON from SSE chunk
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

    // Calculate and display RTF
    const transcriptionEndTime = performance.now();
    const processingTimeMs = transcriptionEndTime - transcriptionStartTime;
    const processingTimeSeconds = processingTimeMs / 1000;
    const rtf = processingTimeSeconds / audioDuration;

    // Display RTF in the UI
    displayRTF(rtf, processingTimeSeconds, audioDuration);

    updateTimeline(segments);
    isTranscribing = false;
    document.getElementById('start-trasncript-btn').textContent = 'Start Transcription';
};

document.getElementById('start-trasncript-btn').addEventListener('click', function(e) {
    document.getElementById('start-trasncript-btn').textContent = 'Transcribing...';
    e.preventDefault();
    startTranscription();
});

// --- TIMELINE FOR UPLOAD FILE FOR REAL-TIME TRANSCRIPTION LOGIC ---
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
        start: 0,    // Start at the beginning
        // end: 10000,  // Show first 10 seconds initially
        zoomMin: 100,    // Minimum zoom matches waveform's minPxPerSec default
        zoomMax: 10000, // Maximum zoom proportional to waveform's maxZoom
        zoomFriction: 5, // Adjusted for smoother sync
        height: '180px'
    };
    timeline = new vis.Timeline(container, timelineItems, options);
    // Add new vertical bar representing a custom time
    timeline.addCustomTime(0, 'cursor');
    window.timeline2 = timeline;
}

document.addEventListener('DOMContentLoaded', initTimeline);

let addedSegmentKeys = new Set();
let colorIndex = 0; // Track current color index

function updateTimeline(segments) {
    if (!timelineItems) return;

    segments.forEach((segment, idx) => {
        // Use a unique key for each segment (e.g., start-end)
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