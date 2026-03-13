import AVFoundation
import Foundation
import KokoroTTS

var args = Array(CommandLine.arguments.dropFirst())

// Parse --voice flag
var voiceArg: String?
if let idx = args.firstIndex(of: "--voice"), idx + 1 < args.count {
    voiceArg = args[idx + 1]
    args.removeSubrange(idx...idx + 1)
}

// Parse --debug flag
var debug = false
if let idx = args.firstIndex(of: "--debug") {
    debug = true
    args.remove(at: idx)
}

// Parse --wav flag (write to file instead of playing)
var wavPath: String?
if let idx = args.firstIndex(of: "--wav"), idx + 1 < args.count {
    wavPath = args[idx + 1]
    args.removeSubrange(idx...idx + 1)
}

// Parse --raw flag (skip post-processing)
var rawAudio = false
if let idx = args.firstIndex(of: "--raw") {
    rawAudio = true
    args.remove(at: idx)
}

// Parse --bucket flag (force a specific model bucket)
var forcedBucket: ModelBucket?
if let idx = args.firstIndex(of: "--bucket"), idx + 1 < args.count {
    switch args[idx + 1] {
    case "small": forcedBucket = .small
    case "medium": forcedBucket = .medium
    default:
        print("Unknown bucket '\(args[idx + 1])'. Use: small, medium")
        exit(1)
    }
    args.removeSubrange(idx...idx + 1)
}

let text = args.joined(separator: " ")

guard !text.isEmpty else {
    print("Usage: kokoro-say [--voice NAME] [--debug] [--raw] [--wav out.wav] [--bucket small|medium] <text>")
    exit(1)
}

let modelDir = ModelManager.defaultDirectory(for: "com.operator")
guard ModelManager.modelsAvailable(at: modelDir) else {
    print("Models not found at \(modelDir.path)")
    exit(1)
}

let engine = try KokoroEngine(modelDirectory: modelDir)
let voice = voiceArg ?? "af_heart"

guard engine.availableVoices.contains(voice) else {
    print("Unknown voice '\(voice)'. Available: \(engine.availableVoices.joined(separator: ", "))")
    exit(1)
}

if debug {
    let bucketNames = engine.activeBuckets.map(\.modelName).joined(separator: ", ")
    print("Loaded buckets: \(bucketNames)")
}

let result = try engine.synthesize(text: text, voice: voice, bucket: forcedBucket, rawAudio: rawAudio)

if debug {
    if let bucket = result.bucket {
        print("Selected bucket: \(bucket.modelName) (\(bucket.maxTokens) max tokens)")
    }
    let windowSize = 120  // 5ms at 24kHz
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

    // Per-token amplitude profile
    let hopSize = KokoroEngine.hopSize
    var cumFrames = 0
    for (i, dur) in result.tokenDurations.enumerated() {
        let startSample = cumFrames * hopSize
        let endSample = min((cumFrames + dur) * hopSize, result.samples.count)
        var peak: Float = 0
        if startSample < endSample {
            for j in startSample..<endSample { peak = max(peak, abs(result.samples[j])) }
        }
        print(String(format: "    t%02d: dur=%2d  %6d-%6d  peak=%.3f", i, dur, startSample, endSample, peak))
        cumFrames += dur
    }
}

let bucketTag = result.bucket.map { " \($0.modelName)" } ?? ""
let fmt = String(format: "%.0fms synth, %.1fs audio", result.synthesisTime * 1000, result.duration)
print("[\(voice)\(bucketTag)] \(fmt) | \"\(text)\"")

// Write WAV if requested
if let wavPath {
    let url = URL(fileURLWithPath: wavPath)
    let settings: [String: Any] = [
        AVFormatIDKey: Int(kAudioFormatLinearPCM),
        AVSampleRateKey: Double(KokoroEngine.sampleRate),
        AVNumberOfChannelsKey: 1,
        AVLinearPCMBitDepthKey: 16,
        AVLinearPCMIsFloatKey: false,
        AVLinearPCMIsBigEndianKey: false,
    ]
    let file = try AVAudioFile(forWriting: url, settings: settings)
    guard let pcmFormat = file.processingFormat as AVAudioFormat?,
          let buf = AVAudioPCMBuffer(pcmFormat: pcmFormat, frameCapacity: AVAudioFrameCount(result.samples.count))
    else {
        print("Failed to create buffer")
        exit(1)
    }
    buf.frameLength = AVAudioFrameCount(result.samples.count)
    result.samples.withUnsafeBufferPointer { src in
        buf.floatChannelData![0].update(from: src.baseAddress!, count: result.samples.count)
    }
    try file.write(from: buf)
    print("Wrote \(wavPath)")
    exit(0)
}

// Play through speakers
let audioEngine = AVAudioEngine()
let player = AVAudioPlayerNode()
guard let format = AVAudioFormat(
    standardFormatWithSampleRate: Double(KokoroEngine.sampleRate), channels: 1
) else {
    print("Failed to create audio format")
    exit(1)
}

audioEngine.attach(player)
audioEngine.connect(player, to: audioEngine.mainMixerNode, format: format)
try audioEngine.start()
player.play()

let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(result.samples.count))!
buf.frameLength = AVAudioFrameCount(result.samples.count)
result.samples.withUnsafeBufferPointer { src in
    buf.floatChannelData![0].update(from: src.baseAddress!, count: result.samples.count)
}

let done = DispatchSemaphore(value: 0)
player.scheduleBuffer(buf) { done.signal() }
done.wait()

Thread.sleep(forTimeInterval: 0.1)
