import ArgumentParser
import AVFoundation
import Foundation
import KokoroTTS

extension ModelBucket: ExpressibleByArgument {
    public init?(argument: String) {
        for c in Self.allCases where "\(c)" == argument {
            self = c
            return
        }
        return nil
    }
}

@main
struct KokoroSay: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "kokoro-say",
        abstract: "Text-to-speech using KokoroTTS",
        version: "1.0.0"
    )

    @Option(name: [.short, .long], help: "Voice preset")
    var voice: String = "af_heart"

    @Option(name: [.short, .long], help: "Speed multiplier, 0.5–2.0")
    var speed: Float = 1.0

    @Option(name: [.short, .long], help: "Write WAV to file")
    var output: String?

    @Option(name: .long, help: "Model directory path")
    var modelDir: String?

    @Option(name: .long, help: "Force model bucket")
    var bucket: ModelBucket?

    @Flag(name: [.short, .long], help: "Play audio through speakers")
    var play = false

    @Flag(name: .long, help: "Skip audio post-processing")
    var raw = false

    @Flag(name: .long, help: "Print debug information")
    var debug = false

    @Flag(name: .long, help: "List available voices")
    var listVoices = false

    @Argument(help: "Text to synthesize (reads stdin if omitted)")
    var text: [String] = []

    func validate() throws {
        guard (0.5...2.0).contains(speed) else {
            throw ValidationError("Speed must be between 0.5 and 2.0")
        }
    }

    mutating func run() throws {
        let dir: URL
        if let modelDir {
            dir = URL(fileURLWithPath: modelDir)
        } else {
            dir = ModelManager.defaultDirectory()
        }

        guard ModelManager.modelsAvailable(at: dir) else {
            print("Models not found at \(dir.path)")
            print("Use --model-dir to specify the model directory.")
            throw ExitCode.failure
        }

        let engine = try KokoroEngine(modelDirectory: dir)

        if listVoices {
            for v in engine.availableVoices.sorted() { print(v) }
            return
        }

        let inputText = try resolveText()

        guard engine.availableVoices.contains(voice) else {
            print("Unknown voice '\(voice)'. Available:")
            for v in engine.availableVoices.sorted() { print("  \(v)") }
            throw ExitCode.failure
        }

        if debug {
            print("Model dir: \(dir.path)")
            print("Loaded buckets: \(engine.activeBuckets.map(\.modelName).joined(separator: ", "))")
        }

        let result = try engine.synthesize(
            text: inputText, voice: voice, speed: speed,
            bucket: bucket, rawAudio: raw
        )

        if debug { printDebugInfo(result: result) }

        let tag = result.bucket.map { " \($0.modelName)" } ?? ""
        let stats = String(
            format: "%.0fms synth, %.1fs audio, %.1fx RT",
            result.synthesisTime * 1000, result.duration, result.realTimeFactor
        )
        print("[\(voice)\(tag)] \(stats)")

        if let output {
            try writeWAV(samples: result.samples, to: output)
            print("Wrote \(output)")
        }

        if play || output == nil {
            try playAudio(samples: result.samples)
        }
    }

    // MARK: - Input

    private func resolveText() throws -> String {
        if !text.isEmpty {
            return text.joined(separator: " ")
        }
        guard isatty(fileno(stdin)) == 0 else {
            throw ValidationError("No text provided. Pass text as arguments or pipe to stdin.")
        }
        var lines: [String] = []
        while let line = readLine() {
            lines.append(line)
        }
        let result = lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !result.isEmpty else {
            throw ValidationError("Empty input from stdin")
        }
        return result
    }

    // MARK: - Audio Helpers

    private func makePCMBuffer(
        from samples: [Float], format: AVAudioFormat
    ) throws -> AVAudioPCMBuffer {
        guard let buf = AVAudioPCMBuffer(
            pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            throw ValidationError("Failed to create audio buffer")
        }
        buf.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in
            buf.floatChannelData![0].update(from: src.baseAddress!, count: samples.count)
        }
        return buf
    }

    private func writeWAV(samples: [Float], to path: String) throws {
        let url = URL(fileURLWithPath: path)
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: Double(KokoroEngine.sampleRate),
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        let file = try AVAudioFile(forWriting: url, settings: settings)
        let buf = try makePCMBuffer(from: samples, format: file.processingFormat)
        try file.write(from: buf)
    }

    private func playAudio(samples: [Float]) throws {
        let audioEngine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        guard let format = AVAudioFormat(
            standardFormatWithSampleRate: Double(KokoroEngine.sampleRate), channels: 1
        ) else {
            throw ValidationError("Failed to create audio format")
        }
        audioEngine.attach(player)
        audioEngine.connect(player, to: audioEngine.mainMixerNode, format: format)
        try audioEngine.start()
        player.play()

        let buf = try makePCMBuffer(from: samples, format: format)
        let done = DispatchSemaphore(value: 0)
        player.scheduleBuffer(buf) { done.signal() }
        done.wait()
        // Allow audio hardware buffer to flush
        Thread.sleep(forTimeInterval: 0.1)
    }

    // MARK: - Debug

    private func printDebugInfo(result: SynthesisResult) {
        if let bucket = result.bucket {
            print("Selected bucket: \(bucket.modelName) (\(bucket.maxTokens) max tokens)")
        }
        let windowSize = 120
        let windows = min(20, result.samples.count / windowSize)
        print("\nOnset amplitude profile (first \(windows * 5)ms):")
        for w in 0..<windows {
            let start = w * windowSize
            let end = min(start + windowSize, result.samples.count)
            var peak: Float = 0
            for i in start..<end { peak = max(peak, abs(result.samples[i])) }
            let bar = String(repeating: "#", count: Int(peak * 50))
            print(String(format: "  %3d-%3dms: %.3f %@", w * 5, (w + 1) * 5, peak, bar))
        }
        var globalPeak: Float = 0
        for s in result.samples { globalPeak = max(globalPeak, abs(s)) }
        print(String(format: "\n  Global peak: %.3f", globalPeak))
        print(String(format: "  Total samples: %d (%.1fs)", result.samples.count, result.duration))
        print(String(format: "  Token durations (%d): %@", result.tokenDurations.count,
            result.tokenDurations.map(String.init).joined(separator: ", ")))

        let hopSize = KokoroEngine.hopSize
        var cumFrames = 0
        for (i, dur) in result.tokenDurations.enumerated() {
            let startSample = cumFrames * hopSize
            let endSample = min((cumFrames + dur) * hopSize, result.samples.count)
            var peak: Float = 0
            if startSample < endSample {
                for j in startSample..<endSample { peak = max(peak, abs(result.samples[j])) }
            }
            print(String(format: "    t%02d: dur=%2d  %6d-%6d  peak=%.3f",
                i, dur, startSample, endSample, peak))
            cumFrames += dur
        }
    }
}
