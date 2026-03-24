import AVFoundation
import Accelerate
import Observation
import Speech

@Observable
@MainActor
final class SpeechRecognizer {
    var transcript = ""
    var isListening = false
    var micAvailable = true
    var silenceDetected = false
    var levelMonitor: AudioLevelMonitor?

    private var silenceTimer: Task<Void, Never>?
    private static let silenceThreshold: TimeInterval = 0.8

    private var audioEngine: AVAudioEngine?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))

    nonisolated var isAuthorized: Bool {
        SFSpeechRecognizer.authorizationStatus() == .authorized
            && AVAudioApplication.shared.recordPermission == .granted
    }

    func requestAuthorization() async -> Bool {
        let speechOK = await withCheckedContinuation { cont in
            SFSpeechRecognizer.requestAuthorization { status in
                cont.resume(returning: status == .authorized)
            }
        }
        guard speechOK else { return false }
        return await AVAudioApplication.requestRecordPermission()
    }

    func startListening() {
        guard !isListening else { return }
        transcript = ""
        micAvailable = true

        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement, options: .duckOthers)
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            micAvailable = false
            return
        }

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)

        guard recordingFormat.sampleRate > 0, recordingFormat.channelCount > 0 else {
            micAvailable = false
            return
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        recognitionRequest = request

        let analyzer = SpectrumAnalyzer(frameCount: 1024, sampleRate: Float(recordingFormat.sampleRate))
        let monitor = self.levelMonitor

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            // Append original audio to recognizer before any modification
            request.append(buffer)
            if let monitor {
                // Boost for visualization only — recognizer already has the original
                if let channelData = buffer.floatChannelData?[0] {
                    let count = Int(buffer.frameLength)
                    var gain: Float = 4.0
                    vDSP_vsmul(channelData, 1, &gain, channelData, 1, vDSP_Length(count))
                }
                let bands = analyzer.analyze(buffer)
                monitor.pushBands(bands)
            }
        }

        do {
            try engine.start()
            self.audioEngine = engine
            isListening = true
        } catch {
            inputNode.removeTap(onBus: 0)
            micAvailable = false
            return
        }

        recognitionTask = speechRecognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }
            if let result {
                Task { @MainActor in
                    let newText = result.bestTranscription.formattedString
                    if newText != self.transcript {
                        self.transcript = newText
                        self.restartSilenceTimer()
                    }
                }
            }
            if error != nil || (result?.isFinal ?? false) {
                Task { @MainActor in
                    self.stopListening()
                }
            }
        }
    }

    func stopListening() {
        guard isListening else { return }
        silenceTimer?.cancel()
        silenceTimer = nil
        silenceDetected = false
        if let engine = audioEngine {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        audioEngine = nil
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        recognitionTask?.finish()
        recognitionTask = nil
        isListening = false
    }

    private func restartSilenceTimer() {
        silenceTimer?.cancel()
        silenceDetected = false
        silenceTimer = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(Self.silenceThreshold))
            guard let self, self.isListening, !self.transcript.isEmpty, !Task.isCancelled else { return }
            self.silenceDetected = true
        }
    }
}
