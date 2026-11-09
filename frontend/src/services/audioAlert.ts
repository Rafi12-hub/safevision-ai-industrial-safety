class AudioAlertManager {
  private audioCtx: AudioContext | null = null;
  private isEnabled: boolean = true;
  private volume: number = 0.4;
  private lastAlertTime: number = 0;
  private cooldownMs: number = 4000; // prevent alert sound fatigue

  constructor() {
    // Loaded lazily on first user interaction
  }

  private initContext() {
    if (!this.audioCtx) {
      const AudioCtxClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.audioCtx = new AudioCtxClass();
    }
    if (this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
  }

  public setEnabled(enabled: boolean) {
    this.isEnabled = enabled;
  }

  public setVolume(vol: number) {
    this.volume = Math.max(0, Math.min(1, vol));
  }

  public playViolationAlarm(details?: string) {
    if (!this.isEnabled) return;

    const now = Date.now();
    if (now - this.lastAlertTime < this.cooldownMs) {
      return;
    }
    this.lastAlertTime = now;

    try {
      this.initContext();
      if (!this.audioCtx) return;

      const ctx = this.audioCtx;
      const osc1 = ctx.createOscillator();
      const osc2 = ctx.createOscillator();
      const gainNode = ctx.createGain();

      osc1.type = 'sawtooth';
      osc2.type = 'square';

      // Warning two-tone industrial pitch
      osc1.frequency.setValueAtTime(880, ctx.currentTime);
      osc1.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.35);

      osc2.frequency.setValueAtTime(740, ctx.currentTime);
      osc2.frequency.exponentialRampToValueAtTime(370, ctx.currentTime + 0.35);

      gainNode.gain.setValueAtTime(this.volume * 0.3, ctx.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.45);

      osc1.connect(gainNode);
      osc2.connect(gainNode);
      gainNode.connect(ctx.destination);

      osc1.start();
      osc2.start();
      osc1.stop(ctx.currentTime + 0.45);
      osc2.stop(ctx.currentTime + 0.45);

      // Optional Web Speech synthesis announcement if available
      if ('speechSynthesis' in window && details) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance("Safety violation detected");
        utterance.rate = 1.1;
        utterance.volume = this.volume;
        window.speechSynthesis.speak(utterance);
      }
    } catch (e) {
      console.warn("Audio warning could not play:", e);
    }
  }
}

export const audioAlert = new AudioAlertManager();
