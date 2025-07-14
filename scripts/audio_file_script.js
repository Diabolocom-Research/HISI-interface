// --- RESET FUNCTION FOR NEW FILE UPLOADS ---
function resetAllData() {
    // Reset transcription output
    const outputElement = document.getElementById('output');
    if (outputElement) {
        outputElement.textContent = '';
    }
    
    // Reset segments table
    const segmentsTableBody = document.getElementById('segments-table-body-upload');
    if (segmentsTableBody) {
        segmentsTableBody.innerHTML = '';
    }
    
    // Clear timeline items
    if (timelineItems) {
        timelineItems.clear();
    }
    
    // Clear regions from wavesurfer
    if (regions) {
        regions.clearRegions();
    }
    
    // Reset added segment keys
    addedSegmentKeys = new Set();
    addedSegmentKeysTable = new Set(); // Also reset table tracking
    
    // Reset color index
    colorIndex = 0;
    
    // Reset progress display
    const progressElement = document.getElementById('progress');
    if (progressElement) {
        progressElement.textContent = '00:00:00';
    }
    
    // Reset transcription button text
    const transcriptBtn = document.getElementById('start-trasncript-btn');
    if (transcriptBtn) {
        transcriptBtn.textContent = 'Start Transcription';
        transcriptBtn.disabled = false; // Re-enable the button
    }
    
    // Reset play button icon
    const playBtnImg = document.getElementById('play-upload-recording');
    if (playBtnImg) {
        playBtnImg.src = 'static/assets/play_black_icon_48.png';
    }
    
    // Reset timeline cursor if timeline exists
    if (timeline) {
        timeline.setCustomTime(0, 'cursor');
        timeline.moveTo(0, { animation: false });
    }
    
    // Stop any current playback
    if (wavesurfer && wavesurfer.isPlaying()) {
        wavesurfer.stop();
    }
    
    console.log('All data reset for new file upload');
}

// -- WaveSurfer.js to handle audio file upload and playback --
let timeline, timelineItems;

import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js'
import ZoomPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/zoom.esm.js'
import RegionsPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.esm.js'
import TimelinePlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/timeline.esm.js'
import Minimap from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/minimap.esm.js'

const regions = RegionsPlugin.create()

const wavesurfer = WaveSurfer.create({
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
    img.src = 'static/assets/pause_black_icon_48.png';
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

const BASE_ZOOM = 100;

// Waveform and Timeline zoom synchronization (debounced)
wavesurfer.on('zoom', (minPxPerSec) => {
    const zoomFactor = minPxPerSec / BASE_ZOOM;
    const zoomLabel = `${zoomFactor.toFixed(1)}x`; 

    document.querySelector('.zoom_level').textContent = zoomLabel;
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

document.getElementById('play-upload-recording-container').addEventListener('click', () => {
    const img = document.getElementById('play-upload-recording');
    if (wavesurfer.isPlaying()) {
        wavesurfer.pause();
        img.src = 'static/assets/play_black_icon_48.png'; 
    } else {
        wavesurfer.play();
        img.src = 'static/assets/pause_black_icon_48.png';
        onAudioProcess(wavesurfer.getCurrentTime());
    }
});

wavesurfer.on('play', () => {
    const img = document.getElementById('play-upload-recording');
    img.src = 'static/assets/pause_black_icon_48.png';
    onAudioProcess(wavesurfer.getCurrentTime());
});

wavesurfer.on('pause', () => {
    const img = document.getElementById('play-upload-recording');
    img.src = 'static/assets/play_black_icon_48.png';
});

function onAudioProcess(currentTime) {
    const formattedTime = new Date(currentTime * 1000).toISOString().substr(11, 8);
    document.getElementById('progress').textContent = formattedTime;
    if (window.timeline2) {
        console.log('Updating timeline cursor to:', currentTime * 1000);
        window.timeline2.setCustomTime(currentTime * 1000, 'cursor');
        window.timeline2.moveTo(currentTime * 1000, { animation: false });
    }
}

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
    document.getElementById('progress').textContent = formattedTime;
});

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
let addedSegmentKeysTable = new Set(); // Track segments already added to table

function updateSegmentsTable(segments) {
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
                                ▶
                            </button>
                        </div>
                    </td>
                `;
            segmentsTableBody.appendChild(row);
            addedSegmentKeysTable.add(key);
            
            // Add event listener for this specific button
            const playButton = row.querySelector('.segment-btn-play');
            playButton.addEventListener('click', (e) => {
                const start = parseFloat(e.target.getAttribute('data-start'));
                const end = parseFloat(e.target.getAttribute('data-end'));
                playSegment(start, end);
            });
        }
    });
}

function playSegment(start, end) {
    if (wavesurfer) {
        wavesurfer.setTime(start);
        wavesurfer.play();
        
        // Stop at end time
        const stopHandler = () => {
            if (wavesurfer.getCurrentTime() >= end) {
                wavesurfer.pause();
                wavesurfer.un('audioprocess', stopHandler);
            }
        };
        wavesurfer.on('audioprocess', stopHandler);
    }
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
    addedSegmentKeysTable = new Set(); // Reset table tracking
    document.getElementById('segments-table-body-upload').innerHTML = '';
    if (timelineItems) timelineItems.clear();

    const response = await fetch('/upload_and_transcribe', {
        method: 'POST',
        body: formData,
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    // Only start playback if not already playing
    if (!wavesurfer.isPlaying()) {
        wavesurfer.play(); 
        const img = document.querySelector('.play_btn');
        img.src = 'static/assets/pause_black_icon_48.png'; 
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

    // if (segments.length > 0) {
    //     timeline.fit();
    // }
}