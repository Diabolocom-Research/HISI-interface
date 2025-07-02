// -- WaveSurfer.js to handle audio file upload and playback --
let timeline, timelineItems;

import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js'
import ZoomPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/zoom.esm.js'
import RegionsPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.esm.js'

const regions = RegionsPlugin.create()

const wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: '#E96C64',
    progressColor: '#F8A05D',
    minPxPerSec: 1,
    plugins: [regions],
    barGap: 3,
    barHeight: 4,
    barWidth: 3,
    barRadius: 10,
});

const colors = [
    "rgba(186, 233, 255, 0.5)", // Light blue, semi-transparent
    "rgba(201, 247, 210, 0.5)", // Light green, semi-transparent
    "rgba(248, 235, 185, 0.5)", // Light yellow, semi-transparent
    "rgba(194, 192, 255, 0.5)", // Light purple, semi-transparent
]

// Loop a region on click
let loop = false

{
  let activeRegion = null
  regions.on('region-in', (region) => {
    console.log('region-in', region)
    activeRegion = region
  })
  regions.on('region-out', (region) => {
    console.log('region-out', region)
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
    img.src = 'static/assets/pause_icon.png';
  })
  // Reset the active region when the user clicks anywhere in the waveform
  wavesurfer.on('interaction', () => {
    activeRegion = null
  })
}

wavesurfer.registerPlugin(
    ZoomPlugin.create({
        // the amount of zoom per wheel step, e.g. 0.2 means a 20% magnification per scroll
        scale: 0.2,
        maxZoom: 1000, 
    })
)

let lastWaveformZoom = 100; 
let mousePosition = null; 
let zoomTimeout = null; // Add debouncing for zoom

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

// Waveform and Timeline zoom synchronization (debounced)
wavesurfer.on('zoom', (minPxPerSec) => {
    if (timeline && Math.abs(minPxPerSec - lastWaveformZoom) > 5) { // Only sync if significant change
        // Clear existing timeout
        if (zoomTimeout) {
            clearTimeout(zoomTimeout);
        }
        
        // Debounce zoom synchronization
        zoomTimeout = setTimeout(() => {
            console.log('Waveform zoom changed:', minPxPerSec, 'Previous:', lastWaveformZoom);
            
            // Determine zoom direction
            const isZoomingIn = minPxPerSec > lastWaveformZoom;
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

document.getElementById('play-upload-recording').addEventListener('click', () => {
    const img = document.getElementById('play-btn-img');
    if (wavesurfer.isPlaying()) {
        wavesurfer.pause();
        img.src = 'static/assets/play_icon.png'; 
    } else {
        wavesurfer.play();
        img.src = 'static/assets/pause_icon.png';
        onAudioProcess(wavesurfer.getCurrentTime());
    }
});

wavesurfer.on('play', () => {
    const img = document.getElementById('play-btn-img');
    img.src = 'static/assets/pause_icon.png';
    onAudioProcess(wavesurfer.getCurrentTime());
});

wavesurfer.on('pause', () => {
    const img = document.getElementById('play-btn-img');
    img.src = 'static/assets/play_icon.png';
});

function onAudioProcess(currentTime) {
    const minutes = Math.floor(currentTime / 60);
    const seconds = Math.floor(currentTime % 60).toString().padStart(2, '0');
    document.querySelector('.audio_duration').textContent = `${minutes}:${seconds}`;
    if (window.timeline2) {
        console.log('Updating timeline cursor to:', currentTime * 1000);
        window.timeline2.setCustomTime(currentTime * 1000, 'cursor');
        window.timeline2.moveTo(currentTime * 1000, { animation: false });
    }
}

wavesurfer.on('audioprocess', onAudioProcess);

wavesurfer.on('finish', () => {
    const img = document.querySelector('.play_btn');
    img.src = 'static/assets/play_icon.png';
});

// Sync timeline cursor with WaveSurfer playback
wavesurfer.on('click', (e) => {
    const currentTime = wavesurfer.getCurrentTime();
            
    // Set the cursor to the clicked position
    timeline.setCustomTime(currentTime * 1000, 'cursor');
    timeline.moveTo(currentTime * 1000, { animation: false });
    const minutes = Math.floor(currentTime / 60);
    const seconds = Math.floor(currentTime % 60).toString().padStart(2, '0');
    document.querySelector('.audio_duration').textContent = `${minutes}:${seconds}`;
});

document.getElementById('audio_file').addEventListener('change', function(e) {
    const file = e.target.files[0];
    const noFileMessage = document.getElementById('no-file-message');
    const waveformContainer = document.getElementById('waveform');
    
    if (file) {
        // Hide the no-file message
        if (noFileMessage) {
            noFileMessage.style.display = 'none';
            waveformContainer.style.display = 'block';
        }
        
        const url = URL.createObjectURL(file);
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
        document.querySelector('.controls-container-upload').style.display = 'flex';

        // Set total audio duration in mm:ss format
        const minutes = Math.floor(duration / 60);
        const seconds = Math.floor(duration % 60).toString().padStart(2, '0');
        document.querySelector('.total_audio_duration').textContent = `${minutes}:${seconds}`;

        if (timeline) {
            timeline.setOptions({
                // end: Math.min(10000, duration * 1000), // Show first 10 seconds or full duration if shorter
                max: duration * 1000 + 2000,
            });
            // Move to the beginning and reset zoom
            timeline.moveTo(1, { animation: false });
            lastWaveformZoom = 100; // Reset zoom tracking
        }
    });
});

// --- UPLOAD FILE FOR REAL-TIME TRANSCRIPTION LOGIC ---
function updateSegmentsTable(segments) {
    const segmentsTableBody = document.getElementById('segments-table-body-upload');
    segments.forEach(segment => {
        const row = document.createElement('tr');
        row.innerHTML = `<td>${segment.start.toFixed(2)}</td><td>${segment.end.toFixed(2)}</td><td>${segment.text}</td>`;
        segmentsTableBody.appendChild(row);
    });
}

function handleTranscriptionEvent(data) {
    // 1. Update the output element with the full transcript
    document.getElementById('output').textContent = data.full_transcript;

    // 2. Update the timeline with the segments
    if (data.segments && Array.isArray(data.segments)) {
        updateTimeline(data.segments);
        updateSegmentsTable(data.segments);
    }    
}

async function startTranscription() {    
    const fileInput = document.getElementById('audio_file');
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('audio_file', file); // field name must match FastAPI

    addedSegmentKeys = new Set();
    document.getElementById('segments-table-body-upload').innerHTML = '';
    if (timelineItems) timelineItems.clear();

    const response = await fetch('/upload_and_transcribe', {
        method: 'POST',
        body: formData,
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let output = '';

    // if (timeline) {
    //     console.log('Timeline initialized, setting up for transcription');
    //     timeline.zoomIn(0.9);
    //     timeline.moveTo(1, { animation: false });
    // }

    // Only start playback if not already playing
    if (!wavesurfer.isPlaying()) {
        wavesurfer.play(); 
        const img = document.querySelector('.play_btn');
        img.src = 'static/assets/pause_icon.png'; 
    }

    while (true) {
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
        end: 10000,  // Show first 10 seconds initially
        zoomMin: 100,    // Minimum zoom matches waveform's minPxPerSec default
        zoomMax: 10000, // Maximum zoom proportional to waveform's maxZoom
        zoomFriction: 5, // Adjusted for smoother sync
        height: '150px'
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
                start: segment.start * 1000, // ms
                end: segment.end * 1000,
            });
            addedSegmentKeys.add(key);
        }
        regions.addRegion({
            start: segment.start,
            end: segment.end,
            color: colors[colorIndex % colors.length], // Use sequential colors
            content: segment.text,
            drag: false,
            resize: false,
        });
        colorIndex++; // Move to next color
    });

    if (segments.length > 0) {
        timeline.fit();
    }
}