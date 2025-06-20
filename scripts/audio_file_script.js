// -- WaveSurfer.js to handle audio file upload and playback --
let timeline, timelineItems;

import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js'

const wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: '#E96C64',
    progressColor: '#F8A05D',
});

document.getElementById('play-upload-recording').addEventListener('click', () => {
    const btn = document.getElementById('play-upload-recording');
    if (wavesurfer.isPlaying()) {
        wavesurfer.pause();
        btn.innerHTML = '<i class="fa-solid fa-play"></i>';
    } else {
        wavesurfer.play();
        btn.innerHTML = '<i class="fa-solid fa-pause"></i>';
    }
});

function onAudioProcess(currentTime) {
    const minutes = Math.floor(currentTime / 60);
    const seconds = Math.floor(currentTime % 60).toString().padStart(2, '0');
    document.querySelector('.audio_duration').textContent = `${minutes}:${seconds}`;
    if (timeline) {
        timeline.setCustomTime(currentTime * 1000, 'cursor');
        timeline.moveTo(currentTime * 1000, { animation: false });
    }
}

wavesurfer.on('audioprocess', onAudioProcess);

// Sync timeline cursor with WaveSurfer playback
wavesurfer.on('click', (e) => {
    const currentTime = wavesurfer.getCurrentTime();
            
    // Set the cursor to the clicked position
    timeline.setCustomTime(currentTime * 1000, 'cursor');
    const minutes = Math.floor(currentTime / 60);
    const seconds = Math.floor(currentTime % 60).toString().padStart(2, '0');
    document.querySelector('.audio_duration').textContent = `${minutes}:${seconds}`;
});

document.getElementById('audio_file').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const url = URL.createObjectURL(file);
        wavesurfer.load(url);
    }
    
    // Get the audio duration after loading the file
    wavesurfer.once('ready', () => {
        const duration = wavesurfer.getDuration(); // duration in seconds
       
        if (timeline) {
            timeline.setOptions({
                max: duration * 4000,
            });
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
    // Parse the JSON if needed
    // const dataObj = JSON.parse(data);

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

    if (timeline) {
        console.log('Timeline initialized, setting up for transcription');
        timeline.zoomIn(0.9);
        timeline.moveTo(0, { animation: false });
    }

    // Only start playback if not already playing
    if (!wavesurfer.isPlaying()) {
        wavesurfer.play(); 
        const btn = document.getElementById('play-upload-recording');
        btn.innerHTML = '<i class="fa-solid fa-pause"></i>';
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
        max: 50000,
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

let addedSegmentKeys = new Set();

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
    });

    if (segments.length > 0) {
        timeline.fit();
    }
}