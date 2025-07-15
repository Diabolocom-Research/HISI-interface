// Record plugin

import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js'
import RecordPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/record.esm.js'
import ZoomPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/zoom.esm.js'
import RegionsPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.esm.js'
import TimelinePlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/timeline.esm.js'
import Minimap from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/minimap.esm.js'

const recordedRegions = RegionsPlugin.create();

window.isPlaybackCreated = false;

let wavesurfer, record
let colorIndex = 0; 

// Create a new Wavesurfer instance
wavesurfer = WaveSurfer.create({
  container: '#mic',
  waveColor: '#F8A05D',
  progressColor: '#E96C64',
  minPxPerSec: 1,
  barGap: 0,
  barHeight: 3,
  barWidth: 2,
  barRadius: 10,
})

let scrollingWaveform = false
let continuousWaveform = true

let lastRecordedWaveSurfer = null;
const playButton = document.getElementById('play-mic-recording-container');

const recButton = document.querySelector('#start-button')
const rectText = document.querySelector('.start-button-txt');
const micImg = document.querySelector('.microphone-icon');

const colors = [
    "rgba(186, 233, 255, 0.5)", // Light blue, semi-transparent
    "rgba(201, 247, 210, 0.5)", // Light green, semi-transparent
    "rgba(248, 235, 185, 0.5)", // Light yellow, semi-transparent
    "rgba(194, 192, 255, 0.5)", // Light purple, semi-transparent
]

// Define the audio process handler as a named function
function onAudioProcess(currentTime) {
    const formattedTime = new Date(currentTime * 1000).toISOString().substr(11, 8);
    document.querySelector('.audio_duration').textContent = formattedTime;
    if (window.timeline) {
        window.timeline.setCustomTime(currentTime * 1000, 'cursor');
        window.timeline.moveTo(currentTime * 1000, { animation: false });
    }
}

let loop = false

{
  let activeRegion = null
  recordedRegions.on('region-in', (region) => {
    console.log('region-in', region)
    activeRegion = region
  })
  recordedRegions.on('region-out', (region) => {
    console.log('region-out', region)
    if (activeRegion === region) {
      if (loop) {
        region.play()
      } else {
        activeRegion = null
      }
    }
  })
  recordedRegions.on('region-clicked', (region, e) => {
    e.stopPropagation() // prevent triggering a click on the waveform
    activeRegion = region
    region.play(true)
    const img = document.getElementById('play-mic-recording');
    img.src = 'static/assets/pause_black_icon_48.png';
  })
  // Reset the active region when the user clicks anywhere in the waveform
  wavesurfer.on('interaction', () => {
    activeRegion = null
  })
}

const createWaveSurfer = () => {
  // Initialize the Record plugin
  record = wavesurfer.registerPlugin(
    RecordPlugin.create({
      renderRecordedAudio: false,
      scrollingWaveform,
      continuousWaveform,
      continuousWaveformDuration: 30, // optional
    }),
  )

  window.wavesurfer = wavesurfer;

  record.on('record-end', (blob) => {
    const container = document.querySelector('#recordings')
    const recordedUrl = URL.createObjectURL(blob)
    const total_audio_duration_record = document.querySelector('.total_audio_duration_record')

    // Create wavesurfer from the recorded audio
    if (lastRecordedWaveSurfer) {
      lastRecordedWaveSurfer.un('audioprocess', onAudioProcess);
      lastRecordedWaveSurfer.destroy();
    }

    window.regions = recordedRegions;

    lastRecordedWaveSurfer = WaveSurfer.create({
      container,
      waveColor: '#E96C64',
      progressColor: '#F8A05D',
      url: recordedUrl,
      minPxPerSec: 1,
      plugins: [recordedRegions, TimelinePlugin.create(),
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
    })

    lastRecordedWaveSurfer.registerPlugin(
      ZoomPlugin.create({
        scale: 0.2,
        maxZoom: 1000, 
      }),
    )

    window.lastRecordedWaveSurfer = lastRecordedWaveSurfer;
    
    lastRecordedWaveSurfer.on('ready', () => {
        container.style.height = '128px';
        console.log('segemtns', window.segments, window.timeline);

        attachZoomSync();
        
        const duration = lastRecordedWaveSurfer.getDuration();
        const minutes = Math.floor(duration / 60);
        const seconds = Math.floor(duration % 60).toString().padStart(2, '0');
        total_audio_duration_record.textContent = `${minutes}:${seconds}`;

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
        })
        
        window.timeline.setOptions({
            max: duration * 1000 + 1000,
        });

        window.timeline.moveTo(1, { animation: false });
        
        // Show reset button since we now have a recording
        if (window.showResetButtonIfNeeded) {
            window.showResetButtonIfNeeded();
        }
    });

    lastRecordedWaveSurfer.on('click', (e) => {
        const currentTime = lastRecordedWaveSurfer.getCurrentTime();
                
        // Set the cursor to the clicked position
        window.timeline.setCustomTime(currentTime * 1000, 'cursor');
        window.timeline.moveTo(currentTime * 1000, { animation: false });
        const minutes = Math.floor(currentTime / 60);
        const seconds = Math.floor(currentTime % 60).toString().padStart(2, '0');
        document.querySelector('.audio_duration').textContent = `${minutes}:${seconds}`;
    });

    // when playback is being played, update the timeline
    lastRecordedWaveSurfer.on('audioprocess', onAudioProcess);

    const img = document.getElementById('play-mic-recording');

    playButton.onclick = () => {
      if (lastRecordedWaveSurfer.isPlaying()) {
        lastRecordedWaveSurfer.pause();
      } else {
        lastRecordedWaveSurfer.play();
      }
    };

    lastRecordedWaveSurfer.on('pause', () => {
        img.src = 'static/assets/play_black_icon_48.png';
        const currentTime = lastRecordedWaveSurfer.getCurrentTime();
                
        // Set the cursor to the final paused position
        window.timeline.setCustomTime(currentTime * 1000, 'cursor');
    });
    
    lastRecordedWaveSurfer.on('click', (e) => {
      const currentTime = lastRecordedWaveSurfer.getCurrentTime();
                
      // Set the cursor to the clicked position
      window.timeline.setCustomTime(currentTime * 1000, 'cursor');
    });

    lastRecordedWaveSurfer.on('play', () => {
        img.src = 'static/assets/pause_black_icon_48.png';
        // Re-attach the audioprocess listener when playback starts again
        // This is important if you pause and then play again from the same instance
        lastRecordedWaveSurfer.on('audioprocess', onAudioProcess); 
    });
  })

  rectText.textContent = 'Start Recording'

  record.on('record-progress', (time) => {
    // updateProgress(time)
  })
}

const progress = document.querySelector('#progress')

const updateProgress = (time) => {
  // time will be in milliseconds, convert it to mm:ss format
  const formattedTime = [
    Math.floor((time % 3600000) / 60000), // minutes
    Math.floor((time % 60000) / 1000), // seconds
  ]
    .map((v) => (v < 10 ? '0' + v : v))
    .join(':')
  progress.textContent = formattedTime
}

recButton.onclick = () => {
  window.timeline.zoomIn(1.5);
  window.timeline.moveTo(1)

  if (record.isRecording() || record.isPaused()) {
    record.stopRecording()
    rectText.textContent = 'Start Recording'
    return
  }

  record.startRecording().then(() => {
    rectText.textContent = 'Stop Recording'
    micImg.src = 'static/assets/stop_recording.png';
  })
}

let lastWaveformZoom = 100; 
let mousePosition = null; 
let zoomTimeout = null; // Add debouncing for zoom

// Track mouse position on waveform for targeted zooming (debounced)
document.getElementById('recordings').addEventListener('mousemove', (e) => {
    if (lastRecordedWaveSurfer && lastRecordedWaveSurfer.getDuration && lastRecordedWaveSurfer.getDuration() > 0) {
        // Throttle mouse position updates
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

// Only attach the zoom event handler after window.recordedWaveform is created
function attachZoomSync() {
  if (!lastRecordedWaveSurfer) return;
  lastRecordedWaveSurfer.on('zoom', (minPxPerSec) => {
    if (window.timeline && Math.abs(minPxPerSec - lastWaveformZoom) > 5) { // Only sync if significant change
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
        window.timeline.setWindow(new Date(newStart), new Date(newEnd), { animation: false });
        
        // Update the last zoom level
        lastWaveformZoom = minPxPerSec;
      }, 100); // 100ms debounce
    }
  });
}

export { createWaveSurfer }
window.createWaveSurfer = createWaveSurfer;