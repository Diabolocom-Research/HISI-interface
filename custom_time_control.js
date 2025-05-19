function manageTimeBar(elemId, time) {
    if (!window.visTimelineInstances) {
        console.error(`Timeline instances collection not found`);
        return;
    }

    const timeline = window.visTimelineInstances[elemId];
    // console.log('Timeline instance:', timeline.components);

    if (!timeline) {
        console.error(`Timeline instance ${elemId} not found`);
        return;
    }
    
    if (!window.customTimeBarIds) {
        window.customTimeBarIds = {};
    }
    
    try {
        timeline.setCustomTime(time, elemId);
    } catch (e) {
        timeline.addCustomTime(time, elemId);
    }
}

function setTimeBarDirect(elemId, time) {
    manageTimeBar(elemId, time);
}

function setTimeBarNormalized(elemId, start, end, normalizedPos) {
    const time = start + (end - start) * normalizedPos;
    manageTimeBar(elemId, time);
}

class AudioTimelineSync {
    constructor(timelineId, audioId, trackLength, plotlyDivId) {
        this.timelineId = timelineId;
        this.trackLength = trackLength;
        this.plotlyDivId = plotlyDivId;  

        const container = document.getElementById(audioId);
        
        // Find the progress element through shadow DOM
        const waveform = container.querySelector('#waveform');
        if (!waveform) {
            console.error('Waveform container not found');
            return;
        }

        // Access shadow root and find progress element
        const shadowRoot = waveform.querySelector('div').shadowRoot;
        this.progressElement = shadowRoot.querySelector('div[part="progress"]');
        
        if (!this.progressElement) {
            console.error('Progress element not found');
            return;
        }
        
        this.setupProgressObserver();
    }

    setupProgressObserver() {
        // Create mutation observer to watch for style changes to a specific progress element of the audio component
        this.observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                    this.onProgressUpdate();
                }
            });
        });

        // Observe the progress element for style changes
        this.observer.observe(this.progressElement, {
            attributes: true,
            attributeFilter: ['style']
        });
    }
    
    onProgressUpdate() {
        const style = this.progressElement.style;
        const widthStr = style.width;
        if (!widthStr) return;
        
        // Convert percentage string to number (e.g., "70.7421%" -> 0.707421)
        const percentage = parseFloat(widthStr) / 100;
        this.syncTimeBarToPlayback(percentage);
        this.updatePlotlyCursor(percentage);
    }

    syncTimeBarToPlayback(normalizedPosition) {
        const timeline = window.visTimelineInstances[this.timelineId];
        if (timeline) {
            const currentTime = normalizedPosition * this.trackLength;
            console.log('Current time:', currentTime, 'Normalized position:', normalizedPosition);
            setTimeBarNormalized(this.timelineId, 0, this.trackLength, normalizedPosition);
            timeline.moveTo(currentTime, { animation: false });
        }
    }

    updatePlotlyCursor(normalizedPosition) {
        const shapeGroup = document.getElementsByClassName('shape-group')[0];
        if (!shapeGroup) return;
    
        const path = shapeGroup.querySelector('path');
        if (!path) return;
    
        const plotlyDiv = document.getElementById(this.plotlyDivId);
        if (!plotlyDiv) return;
        
        const containerWidth = plotlyDiv.clientWidth;
        const xyGroup = plotlyDiv.querySelector('.xy');
        const plotWidth = xyGroup?.getBoundingClientRect().width || plotlyDiv.clientWidth;

        const translateX = normalizedPosition * plotWidth;
    
        path.setAttribute('transform', `translate(${translateX}, 0)`);
    }

    cleanup() {
        // Disconnect observer
        if (this.observer) {
            this.observer.disconnect();
            this.observer = null;
        }
    }
}

window.setConfig = function(timelineId, audioDurationMs) {
    console.log('setConfig called with:', audioDurationMs);
    const timeline = window.visTimelineInstances[timelineId];
    if (timeline) {
        timeline.setOptions({
            max: audioDurationMs,
        });
    } else {
        console.warn('Timeline not found:', timelineId);
    }
}

window.initAudioSync = function(timelineId, audioId, trackLength, plotlyDivId) {
    try {
        console.log('Initializing audio sync for:', timelineId, audioId, trackLength, plotlyDivId);

        if (!window.audioTimelineSyncs) {
            window.audioTimelineSyncs = {};
        }

        if (window.audioTimelineSyncs[timelineId]) {
            window.audioTimelineSyncs[timelineId].cleanup();
        }

        window.audioTimelineSyncs[timelineId] = new AudioTimelineSync(timelineId, audioId, trackLength, plotlyDivId);
    } catch (error) {
        console.error('Error initializing audio sync:', error);
    }

    return null;
};

window.initAudioSync = initAudioSync;
window.setConfig = setConfig;