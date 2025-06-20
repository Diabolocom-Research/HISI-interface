// Record plugin

import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js'
import RecordPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/record.esm.js'

let wavesurfer, record
let scrollingWaveform = false
let continuousWaveform = true

let lastRecordedWaveSurfer = null;
const playButton = document.getElementById('play-mic-recording');

// Define the audio process handler as a named function
function onAudioProcess(currentTime) {
    if (window.timeline) {
        window.timeline.setCustomTime(currentTime * 1000, 'cursor');
        window.timeline.moveTo(currentTime * 1000, { animation: false });
    }
}

const createWaveSurfer = () => {
  // Destroy the previous wavesurfer instance
  if (wavesurfer) {
    wavesurfer.destroy()
  }

  // Create a new Wavesurfer instance
  wavesurfer = WaveSurfer.create({
    container: '#mic',
    waveColor: '#F8A05D',
    progressColor: '#E96C64',
  })

  // Initialize the Record plugin
  record = wavesurfer.registerPlugin(
    RecordPlugin.create({
      renderRecordedAudio: false,
      scrollingWaveform,
      continuousWaveform,
      continuousWaveformDuration: 30, // optional
    }),
  )

  record.on('record-end', (blob) => {
    const container = document.querySelector('#recordings')
    const recordedUrl = URL.createObjectURL(blob)

    // Create wavesurfer from the recorded audio
    if (lastRecordedWaveSurfer) {
      lastRecordedWaveSurfer.un('audioprocess', onAudioProcess);
      lastRecordedWaveSurfer.destroy();
    }

    lastRecordedWaveSurfer = WaveSurfer.create({
      container,
      waveColor: '#E96C64',
      progressColor: '#F8A05D',
      url: recordedUrl,
    })

    // when playback is being played, update the timeline
    lastRecordedWaveSurfer.on('audioprocess', onAudioProcess);

    playButton.style.display = 'inline';
    playButton.textContent = 'Play';

    playButton.onclick = () => {
      if (lastRecordedWaveSurfer.isPlaying()) {
        lastRecordedWaveSurfer.pause();
      } else {
        lastRecordedWaveSurfer.play();
      }
    };

    lastRecordedWaveSurfer.on('pause', () => {
        playButton.textContent = 'Play';
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
        playButton.textContent = 'Pause';
        // Re-attach the audioprocess listener when playback starts again
        // This is important if you pause and then play again from the same instance
        lastRecordedWaveSurfer.on('audioprocess', onAudioProcess); 
    });
  })

  // pauseButton.style.display = 'none'
  recButton.textContent = 'Record'

  record.on('record-progress', (time) => {
    updateProgress(time)
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

// Record button
const recButton = document.querySelector('#start-button')

recButton.onclick = () => {
  window.timeline.zoomIn(1.5);
  window.timeline.moveTo(1)

  if (record.isRecording() || record.isPaused()) {
    record.stopRecording()
    recButton.textContent = 'Record'
    return
  }

  recButton.disabled = true

  record.startRecording().then(() => {
    recButton.textContent = 'Stop'
    recButton.disabled = false
  })
}

export { createWaveSurfer }
window.createWaveSurfer = createWaveSurfer;

// createWaveSurfer()