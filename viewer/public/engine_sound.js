// ========== CESSNA 172 ENGINE SOUND SYSTEM ==========
// Lycoming O-360 engine sound simulation using Web Audio API

let audioContext = null;
let engineSound = null;
let soundEnabled = false;
let soundInitialized = false;

const ENGINE_IDLE_RPM = 600;
const ENGINE_MAX_RPM = 2700;

function initEngineSound() {
  if (soundInitialized) return;
  
  try {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    
    engineSound = {
      mainOsc: audioContext.createOscillator(),
      mainGain: audioContext.createGain(),
      pulseOsc: audioContext.createOscillator(),
      pulseGain: audioContext.createGain(),
      exhaustOsc: audioContext.createOscillator(),
      exhaustGain: audioContext.createGain(),
      noiseNode: null,
      noiseGain: audioContext.createGain(),
      masterGain: audioContext.createGain(),
      lpFilter: audioContext.createBiquadFilter()
    };

    // Create noise buffer
    const bufferSize = 2 * audioContext.sampleRate;
    const noiseBuffer = audioContext.createBuffer(1, bufferSize, audioContext.sampleRate);
    const output = noiseBuffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      output[i] = Math.random() * 2 - 1;
    }
    engineSound.noiseNode = audioContext.createBufferSource();
    engineSound.noiseNode.buffer = noiseBuffer;
    engineSound.noiseNode.loop = true;

    // Configure oscillators
    engineSound.mainOsc.type = 'sawtooth';
    engineSound.pulseOsc.type = 'square';
    engineSound.exhaustOsc.type = 'triangle';

    // Configure filter
    engineSound.lpFilter.type = 'lowpass';
    engineSound.lpFilter.frequency.value = 800;
    engineSound.lpFilter.Q.value = 1;

    // Set initial gains
    engineSound.mainGain.gain.value = 0;
    engineSound.pulseGain.gain.value = 0;
    engineSound.exhaustGain.gain.value = 0;
    engineSound.noiseGain.gain.value = 0;
    engineSound.masterGain.gain.value = 0.25;

    // Connect audio graph
    engineSound.mainOsc.connect(engineSound.mainGain);
    engineSound.mainGain.connect(engineSound.lpFilter);
    
    engineSound.pulseOsc.connect(engineSound.pulseGain);
    engineSound.pulseGain.connect(engineSound.lpFilter);
    
    engineSound.exhaustOsc.connect(engineSound.exhaustGain);
    engineSound.exhaustGain.connect(engineSound.lpFilter);
    
    engineSound.noiseNode.connect(engineSound.noiseGain);
    engineSound.noiseGain.connect(engineSound.lpFilter);
    
    engineSound.lpFilter.connect(engineSound.masterGain);
    engineSound.masterGain.connect(audioContext.destination);

    // Start oscillators
    engineSound.mainOsc.start();
    engineSound.pulseOsc.start();
    engineSound.exhaustOsc.start();
    engineSound.noiseNode.start();

    soundInitialized = true;
    console.log('Cessna 172 engine sound initialized');
  } catch (e) {
    console.error('Failed to initialize engine sound:', e);
  }
}

let lastLogTime = 0;
function updateEngineSound(throttle) {
  if (!soundInitialized || !engineSound || !soundEnabled) return;
  // Log every 2 seconds to avoid spam
  const now = Date.now();
  if (now - lastLogTime > 2000) {
    console.log('[Sound] updateEngineSound, throttle:', throttle.toFixed(2));
    lastLogTime = now;
  }
  
  const rpm = ENGINE_IDLE_RPM + throttle * (ENGINE_MAX_RPM - ENGINE_IDLE_RPM);
  const firingFreq = (rpm / 60) * 2;
  const mainFreq = firingFreq * 0.5;
  const exhaustFreq = firingFreq * 1.5;
  
  const now = audioContext.currentTime;
  engineSound.mainOsc.frequency.setTargetAtTime(mainFreq, now, 0.1);
  engineSound.pulseOsc.frequency.setTargetAtTime(firingFreq, now, 0.1);
  engineSound.exhaustOsc.frequency.setTargetAtTime(exhaustFreq, now, 0.1);
  
  const baseGain = 0.12 + throttle * 0.2;
  const pulseGain = 0.04 + throttle * 0.12;
  const exhaustGain = 0.02 + throttle * 0.1;
  const noiseGain = 0.015 + throttle * 0.06;
  
  engineSound.mainGain.gain.setTargetAtTime(baseGain, now, 0.1);
  engineSound.pulseGain.gain.setTargetAtTime(pulseGain, now, 0.1);
  engineSound.exhaustGain.gain.setTargetAtTime(exhaustGain, now, 0.1);
  engineSound.noiseGain.gain.setTargetAtTime(noiseGain, now, 0.1);
  
  const filterFreq = 500 + throttle * 1500;
  engineSound.lpFilter.frequency.setTargetAtTime(filterFreq, now, 0.1);
}

function toggleEngineSound() {
  console.log('[Sound] toggleEngineSound called, soundInitialized:', soundInitialized);
  if (!soundInitialized) {
    initEngineSound();
  }

  soundEnabled = !soundEnabled;
  console.log('[Sound] soundEnabled now:', soundEnabled);

  if (audioContext) {
    console.log('[Sound] audioContext state:', audioContext.state);
    if (soundEnabled) {
      audioContext.resume().then(() => {
        console.log('[Sound] audioContext resumed, state:', audioContext.state);
      });
    } else {
      const now = audioContext.currentTime;
      engineSound.mainGain.gain.setTargetAtTime(0, now, 0.05);
      engineSound.pulseGain.gain.setTargetAtTime(0, now, 0.05);
      engineSound.exhaustGain.gain.setTargetAtTime(0, now, 0.05);
      engineSound.noiseGain.gain.setTargetAtTime(0, now, 0.05);
    }
  }
  
  const soundBtn = document.getElementById('soundBtn');
  if (soundBtn) {
    soundBtn.textContent = soundEnabled ? 'Sound ON' : 'Sound OFF';
    soundBtn.classList.toggle('active', soundEnabled);
  }
  
  return soundEnabled;
}

// Export for use in viewer
window.engineSoundSystem = {
  init: initEngineSound,
  update: updateEngineSound,
  toggle: toggleEngineSound,
  isEnabled: () => soundEnabled
};

// ========== ALTITUDE CALLOUT SYSTEM ==========
// GPWS/TAWS style altitude callouts for landing

let calloutAudioContext = null;
let lastCalloutAltitude = Infinity;
let calloutEnabled = true;

// Altitude callout thresholds (in feet)
const CALLOUT_ALTITUDES = [500, 400, 300, 200, 100, 50, 40, 30, 20, 10];

// Speech synthesis for callouts
function speakCallout(text) {
  if (!calloutEnabled) return;
  
  if ('speechSynthesis' in window) {
    // Cancel any pending speech
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.1;  // Slightly faster
    utterance.pitch = 0.9; // Slightly lower pitch for authority
    utterance.volume = 0.8;
    
    // Try to use a male voice if available
    const voices = window.speechSynthesis.getVoices();
    const maleVoice = voices.find(v => v.name.includes('Male') || v.name.includes('David') || v.name.includes('Daniel'));
    if (maleVoice) {
      utterance.voice = maleVoice;
    }
    
    window.speechSynthesis.speak(utterance);
  }
}

// Generate tone-based callout as fallback
function playToneCallout(altitude) {
  if (!calloutEnabled) return;
  
  try {
    if (!calloutAudioContext) {
      calloutAudioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    const now = calloutAudioContext.currentTime;
    
    // Create oscillator for beep
    const osc = calloutAudioContext.createOscillator();
    const gain = calloutAudioContext.createGain();
    
    // Higher pitch for lower altitudes (more urgent)
    const baseFreq = 440;
    const freqMultiplier = altitude <= 50 ? 2.0 : altitude <= 100 ? 1.5 : 1.0;
    osc.frequency.value = baseFreq * freqMultiplier;
    osc.type = 'sine';
    
    // Quick beep envelope
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(0.3, now + 0.02);
    gain.gain.linearRampToValueAtTime(0.3, now + 0.1);
    gain.gain.linearRampToValueAtTime(0, now + 0.15);
    
    osc.connect(gain);
    gain.connect(calloutAudioContext.destination);
    
    osc.start(now);
    osc.stop(now + 0.2);
    
    // Double beep for very low altitudes
    if (altitude <= 30) {
      const osc2 = calloutAudioContext.createOscillator();
      const gain2 = calloutAudioContext.createGain();
      osc2.frequency.value = baseFreq * freqMultiplier;
      osc2.type = 'sine';
      gain2.gain.setValueAtTime(0, now + 0.2);
      gain2.gain.linearRampToValueAtTime(0.3, now + 0.22);
      gain2.gain.linearRampToValueAtTime(0.3, now + 0.32);
      gain2.gain.linearRampToValueAtTime(0, now + 0.37);
      osc2.connect(gain2);
      gain2.connect(calloutAudioContext.destination);
      osc2.start(now + 0.2);
      osc2.stop(now + 0.4);
    }
  } catch (e) {
    console.error('Tone callout error:', e);
  }
}

function checkAltitudeCallout(altitudeFt, verticalSpeedFpm) {
  // Only call out when descending
  if (verticalSpeedFpm >= -50) {
    // Not descending significantly, reset
    if (altitudeFt > 600) {
      lastCalloutAltitude = Infinity;
    }
    return;
  }
  
  // Find if we crossed a callout threshold
  for (const threshold of CALLOUT_ALTITUDES) {
    if (lastCalloutAltitude > threshold && altitudeFt <= threshold && altitudeFt > threshold - 15) {
      // Crossed this threshold
      lastCalloutAltitude = threshold;
      
      // Speak the callout
      if (threshold >= 100) {
        speakCallout(threshold.toString());
      } else {
        speakCallout(threshold.toString());
      }
      
      // Also play tone
      playToneCallout(threshold);
      
      console.log('Altitude callout:', threshold, 'ft');
      break;
    }
  }
  
  // Special callouts
  if (altitudeFt <= 5 && lastCalloutAltitude > 5 && verticalSpeedFpm < -100) {
    speakCallout('Retard');  // Standard Airbus callout for touchdown
    lastCalloutAltitude = 5;
  }
}

function resetAltitudeCallouts() {
  lastCalloutAltitude = Infinity;
}

function toggleCallouts() {
  calloutEnabled = !calloutEnabled;
  
  const calloutBtn = document.getElementById('calloutBtn');
  if (calloutBtn) {
    calloutBtn.textContent = calloutEnabled ? 'Callouts ON' : 'Callouts OFF';
    calloutBtn.classList.toggle('active', calloutEnabled);
  }
  
  return calloutEnabled;
}

// Preload voices
if ('speechSynthesis' in window) {
  window.speechSynthesis.getVoices();
  window.speechSynthesis.onvoiceschanged = () => {
    window.speechSynthesis.getVoices();
  };
}

// Export callout system
window.altitudeCalloutSystem = {
  check: checkAltitudeCallout,
  reset: resetAltitudeCallouts,
  toggle: toggleCallouts,
  isEnabled: () => calloutEnabled
};
