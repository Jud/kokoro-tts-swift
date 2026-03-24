import AVFoundation
import KokoroCoreML
import Observation

@Observable
@MainActor
final class AudioEngine: NSObject, AVAudioPlayerDelegate {
    var isPlaying = false
    var isSynthesizing = false
    var availableVoices: [String] = []
    var isReady = false
    var error: String?
    var levelMonitor: AudioLevelMonitor?

    private var engine: KokoroEngine?
    private var player: AVAudioPlayer?
    private var animationTask: Task<Void, Never>?
    private var _analyzer: SpectrumAnalyzer?
    private var spectrumAnalyzer: SpectrumAnalyzer {
        if let a = _analyzer { return a }
        let a = SpectrumAnalyzer(frameCount: 1024, sampleRate: Float(KokoroEngine.sampleRate))
        _analyzer = a
        return a
    }

    func load(modelDirectory: URL) {
        do {
            let kokoroEngine = try KokoroEngine(modelDirectory: modelDirectory)
            self.engine = kokoroEngine
            self.availableVoices = kokoroEngine.availableVoices
            self.isReady = true
        } catch {
            self.error = "Failed to load models: \(error.localizedDescription)"
        }
    }

    func speak(text: String, voice: String, speed: Float = 1.0) {
        guard let engine, !text.isEmpty else { return }

        stop()
        isSynthesizing = true
        error = nil

        let thread = Thread { [weak self] in
            do {
                let result = try engine.synthesize(text: text, voice: voice, speed: speed)
                let wavData = Self.wavData(from: result.samples, sampleRate: KokoroEngine.sampleRate)

                Task { @MainActor in
                    guard let self, self.isSynthesizing else { return }
                    self.isSynthesizing = false
                    self.isPlaying = true
                    self.playWAV(wavData, samples: result.samples)
                }
            } catch {
                Task { @MainActor in
                    self?.error = "Synthesis failed: \(error.localizedDescription)"
                    self?.isSynthesizing = false
                    self?.isPlaying = false
                }
            }
        }
        thread.stackSize = 8 * 1024 * 1024
        thread.start()
    }

    func stop() {
        player?.stop()
        teardown()
        isSynthesizing = false
    }

    // MARK: - AVAudioPlayerDelegate

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in self.teardown() }
    }

    // MARK: - Private

    private func teardown() {
        animationTask?.cancel()
        animationTask = nil
        player = nil
        isPlaying = false
        levelMonitor?.reset()
    }

    private func playWAV(_ data: Data, samples: [Float]) {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default)
            try session.setActive(true)

            let audioPlayer = try AVAudioPlayer(data: data)
            audioPlayer.delegate = self
            audioPlayer.prepareToPlay()
            audioPlayer.play()
            self.player = audioPlayer

            animateWaveform(samples: samples, duration: audioPlayer.duration)
        } catch {
            self.error = "Playback failed: \(error.localizedDescription)"
            isPlaying = false
        }
    }

    private func animateWaveform(samples: [Float], duration: Double) {
        let analyzer = spectrumAnalyzer

        animationTask?.cancel()
        animationTask = Task { @MainActor [weak self] in
            let sampleRate = Double(KokoroEngine.sampleRate)
            let startTime = CFAbsoluteTimeGetCurrent()

            while let self, self.isPlaying, !Task.isCancelled {
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                if elapsed >= duration + 0.2 { break }

                let sampleIndex = Int(elapsed * sampleRate)
                if sampleIndex < samples.count {
                    let bands = analyzer.analyze(samples: samples, offset: sampleIndex)
                    self.levelMonitor?.pushBands(bands)
                }

                try? await Task.sleep(for: .milliseconds(33))
            }
        }
    }

    /// Encode Float samples as a 16-bit PCM WAV in memory.
    private static func wavData(from samples: [Float], sampleRate: Int) -> Data {
        let numSamples = samples.count
        let dataSize = numSamples * 2
        let fileSize = 44 + dataSize

        var data = Data(capacity: fileSize)

        func appendLE<T: FixedWidthInteger>(_ value: T) {
            var v = value.littleEndian
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }

        data.append(contentsOf: [0x52, 0x49, 0x46, 0x46])  // "RIFF"
        appendLE(UInt32(fileSize - 8))
        data.append(contentsOf: [0x57, 0x41, 0x56, 0x45])  // "WAVE"
        data.append(contentsOf: [0x66, 0x6D, 0x74, 0x20])  // "fmt "
        appendLE(UInt32(16))
        appendLE(UInt16(1))  // PCM
        appendLE(UInt16(1))  // mono
        appendLE(UInt32(sampleRate))
        appendLE(UInt32(sampleRate * 2))
        appendLE(UInt16(2))  // block align
        appendLE(UInt16(16))  // bits per sample
        data.append(contentsOf: [0x64, 0x61, 0x74, 0x61])  // "data"
        appendLE(UInt32(dataSize))

        // Bulk convert float samples to Int16
        var int16Samples = [Int16](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            let clamped = max(-1.0, min(1.0, samples[i]))
            int16Samples[i] = Int16(clamped * 32767)
        }
        int16Samples.withUnsafeBytes { data.append(contentsOf: $0) }

        return data
    }
}
