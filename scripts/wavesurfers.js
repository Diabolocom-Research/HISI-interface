// --- Imports ---
import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js';
import RecordPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/record.esm.js';
import ZoomPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/zoom.esm.js';
import RegionsPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.esm.js';
import TimelinePlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/timeline.esm.js';
import Minimap from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/minimap.esm.js';

// --- Global Variables and DOM References ---
const recordedRegions = RegionsPlugin.create();
let wavesurfer, record;
let colorIndex = 0;

// Recording WaveSurfer instance (live visualization)
wavesurfer = WaveSurfer.create({
    container: '#mic',
    waveColor: '#F8A05D',
    progressColor: '#E96C64',
    minPxPerSec: 1,
    barGap: 0,
    barHeight: 3,
    barWidth: 2,
    barRadius: 10,
});

let scrollingWaveform = false;
let continuousWaveform = true;

let lastRecordedWaveSurfer = null;
const playButton = document.getElementById('play-mic-recording-container');

const recButton = document.querySelector('#start-button');
const rectText = document.querySelector('.start-button-txt');
const micImg = document.querySelector('.microphone-icon');

const colors = [
    "rgba(186, 233, 255, 0.5)", // Light blue, semi-transparent
    "rgba(201, 247, 210, 0.5)", // Light green, semi-transparent
    "rgba(248, 235, 185, 0.5)", // Light yellow, semi-transparent
    "rgba(194, 192, 255, 0.5)", // Light purple, semi-transparent
];

let loop = false;
let lastWaveformZoom = 100;
let mousePosition = null;
let zoomTimeout = null;

// --- Utility Functions ---

/**
 * Updates the progress display with formatted time.
 * @param {number} time - Current time in milliseconds.
 */
const updateProgress = (time) => {
    const formattedTime = [
        Math.floor((time % 3600000) / 60000),
        Math.floor((time % 60000) / 1000),
    ]
        .map((v) => (v < 10 ? '0' + v : v))
        .join(':');
    document.querySelector('#progress').textContent = formattedTime;
};

// --- WaveSurfer Event Handlers ---

/**
 * Handles the 'audioprocess' event for WaveSurfer instances, updating time and timeline.
 * @param {number} currentTime - The current playback time in seconds.
 */
function onAudioProcess(currentTime) {
    const formattedTime = new Date(currentTime * 1000).toISOString().substr(11, 8);
    document.querySelector('.audio_duration').textContent = formattedTime;
    if (window.timeline) {
        window.timeline.setCustomTime(currentTime * 1000, 'cursor');
        window.timeline.moveTo(currentTime * 1000, { animation: false });
    }
}

// --- Region Plugin Event Handlers (for recorded audio playback) ---
{
    let activeRegion = null;
    recordedRegions.on('region-in', (region) => {
        activeRegion = region;
    });
    recordedRegions.on('region-out', (region) => {
        if (activeRegion === region) {
            if (loop) {
                region.play();
            } else {
                activeRegion = null;
            }
        }
    });
    recordedRegions.on('region-clicked', (region, e) => {
        e.stopPropagation(); // Prevent triggering a click on the waveform itself
        activeRegion = region;
        region.play(true);
        const img = document.getElementById('play-mic-recording');
        img.src = 'static/assets/pause_black_icon_48.png';
    });
    // Reset the active region when the user clicks anywhere in the waveform
    wavesurfer.on('interaction', () => {
        activeRegion = null;
    });
}

// --- WaveSurfer Creation and Plugin Initialization ---

/**
 * Creates and initializes the WaveSurfer instance for recording and its plugins.
 */
const createWaveSurfer = () => {
    // Initialize the Record plugin for the live waveform
    record = wavesurfer.registerPlugin(
        RecordPlugin.create({
            renderRecordedAudio: false,
            scrollingWaveform,
            continuousWaveform,
            continuousWaveformDuration: 30,
        }),
    );

    window.wavesurfer = wavesurfer;

    // Event listener for when recording ends
    record.on('record-end', (blob) => {
        const container = document.querySelector('#recordings'); // Container for playback waveform
        const recordedUrl = URL.createObjectURL(blob);
        const total_audio_duration_record = document.querySelector('.total_audio_duration_record');

        // Hide no-waveform message and show playback container
        document.getElementById('no-waveform-message-playback').style.display = 'none';
        document.getElementById('recordings').style.display = 'block';

        // Destroy previous playback WaveSurfer instance if it exists
        if (lastRecordedWaveSurfer) {
            lastRecordedWaveSurfer.un('audioprocess', onAudioProcess);
            lastRecordedWaveSurfer.destroy();
        }

        window.regions = recordedRegions;

        // Create new WaveSurfer instance for playing back the recorded audio
        lastRecordedWaveSurfer = WaveSurfer.create({
            container,
            waveColor: '#E96C64',
            progressColor: '#F8A05D',
            url: recordedUrl,
            minPxPerSec: 1,
            plugins: [
                recordedRegions,
                TimelinePlugin.create(),
                Minimap.create({
                    height: 30,
                    waveColor: '#E96C64',
                    progressColor: '#F8A05D',
                }),
            ],
            barGap: 0,
            barHeight: 3,
            barWidth: 2,
            barRadius: 10,
        });

        // Register Zoom Plugin for the playback waveform
        lastRecordedWaveSurfer.registerPlugin(
            ZoomPlugin.create({
                scale: 0.2,
                maxZoom: 1000,
            }),
        );

        window.lastRecordedWaveSurfer = lastRecordedWaveSurfer;

        // Event listener for when the recorded audio waveform is ready
        lastRecordedWaveSurfer.on('ready', () => {
            container.style.height = '128px'; // Adjust container height

            attachZoomSync(); // Attach zoom synchronization handler

            const duration = lastRecordedWaveSurfer.getDuration();
            const minutes = Math.floor(duration / 60);
            const seconds = Math.floor(duration % 60).toString().padStart(2, '0');
            total_audio_duration_record.textContent = `${minutes}:${seconds}`;

            // Add regions to the recorded waveform based on transcribed segments
            if (window.segments && Array.isArray(window.segments)) {
                window.segments.forEach((segment, index) => {
                    recordedRegions.addRegion({
                        start: segment.start,
                        end: segment.end,
                        content: segment.text,
                        drag: false,
                        resize: false,
                        color: colors[colorIndex % colors.length],
                        id: `segment-${index}`,
                    });
                    colorIndex++;
                });
            }

            // Update timeline options based on recorded audio duration
            if (window.timeline) {
                window.timeline.setOptions({
                    max: duration * 1000 + 1000,
                });
                window.timeline.moveTo(1); // Move timeline to the beginning
            }


            // Show reset button since a recording is now available
            if (window.showResetButtonIfNeeded) {
                window.showResetButtonIfNeeded();
            }
        });

        // Click event listener for the playback waveform (to seek and update timeline)
        lastRecordedWaveSurfer.on('click', (e) => {
            const currentTime = lastRecordedWaveSurfer.getCurrentTime();
            if (window.timeline) {
                window.timeline.setCustomTime(currentTime * 1000, 'cursor');
                window.timeline.moveTo(currentTime * 1000, { animation: false });
            }
            const minutes = Math.floor(currentTime / 60);
            const seconds = Math.floor(currentTime % 60).toString().padStart(2, '0');
            document.querySelector('.audio_duration').textContent = `${minutes}:${seconds}`;
        });

        // When playback is active, update the timeline cursor
        lastRecordedWaveSurfer.on('audioprocess', onAudioProcess);

        // Play/Pause button functionality for the recorded audio playback
        const img = document.getElementById('play-mic-recording');
        playButton.onclick = () => {
            if (lastRecordedWaveSurfer.isPlaying()) {
                lastRecordedWaveSurfer.pause();
            } else {
                lastRecordedWaveSurfer.play();
            }
        };

        // Update play button icon and timeline cursor on pause
        lastRecordedWaveSurfer.on('pause', () => {
            img.src = 'static/assets/play_black_icon_48.png';
            const currentTime = lastRecordedWaveSurfer.getCurrentTime();
            if (window.timeline) {
                window.timeline.setCustomTime(currentTime * 1000, 'cursor');
            }
        });

        // Update play button icon and re-attach audioprocess listener on play
        lastRecordedWaveSurfer.on('play', () => {
            img.src = 'static/assets/pause_black_icon_48.png';
            lastRecordedWaveSurfer.on('audioprocess', onAudioProcess);
        });
    });

    // Initial text for the record button
    rectText.textContent = 'Start Recording';

    // Event listener for recording progress
    record.on('record-progress', (time) => {
        // updateProgress(time);
    });
};

// --- Record Button Click Handler ---
recButton.onclick = () => {
    if (window.timeline) {
        window.timeline.zoomIn(1.5); // Zoom in timeline when recording starts
        window.timeline.moveTo(1); // Move timeline to the beginning
    }

    if (record.isRecording() || record.isPaused()) {
        record.stopRecording();
        rectText.textContent = 'Start Recording';
    } else {
        record.startRecording().then(() => {
            rectText.textContent = 'Stop Recording';
            micImg.src = 'static/assets/stop_recording.png';
        });
    }
};

// --- Zoom Synchronization for Recorded Waveform ---

// Track mouse position on waveform for targeted zooming (debounced)
document.getElementById('recordings').addEventListener('mousemove', (e) => {
    if (lastRecordedWaveSurfer && lastRecordedWaveSurfer.getDuration && lastRecordedWaveSurfer.getDuration() > 0) {

        if (!mousePosition || Date.now() - (mousePosition.lastUpdate || 0) > 50) {
            const rect = e.currentTarget.getBoundingClientRect();
            const relativeX = (e.clientX - rect.left) / rect.width;
            mousePosition = {
                time: relativeX * lastRecordedWaveSurfer.getDuration(),
                lastUpdate: Date.now()
            };
        }
    }
});

/**
 * Attaches the zoom synchronization handler between the recorded WaveSurfer and the timeline.
 */
function attachZoomSync() {
    if (!lastRecordedWaveSurfer) return;

    lastRecordedWaveSurfer.on('zoom', (minPxPerSec) => {
        if (window.timeline && Math.abs(minPxPerSec - lastWaveformZoom) > 5) {
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

                window.timeline.setWindow(new Date(newStart), new Date(newEnd), { animation: false });
                lastWaveformZoom = minPxPerSec;
            }, 100); // 100ms debounce
        }
    });
}

// --- Export and Global Exposure ---
export { createWaveSurfer };
window.createWaveSurfer = createWaveSurfer;
