import AVFoundation
import Accelerate

/// Computes frequency band magnitudes from audio samples using FFT.
///
/// Bands are logarithmically spaced from 60 Hz to 8 kHz. Sorted by energy
/// and mirrored so the tallest bands are always at center — creating the
/// mountain-shaped waveform.
final class SpectrumAnalyzer: @unchecked Sendable {
    static let bandCount = 40

    private static let dbFloor: Float = -55
    private static let dbRange: Float = 50
    private static let minFreq: Float = 60
    private static let maxFreq: Float = 8_000

    private let fftSetup: FFTSetup
    private let log2n: vDSP_Length
    private let halfN: Int
    private let frameCount: Int
    private let window: [Float]
    private let bandRanges: [(start: Int, end: Int)]

    private let lock = NSLock()
    private var windowed: [Float]
    private var realp: [Float]
    private var imagp: [Float]
    private var magnitudes: [Float]
    private var bands: [Float]
    private var halfSorted: [Float]

    init(frameCount: Int = 1024, sampleRate: Float) {
        self.frameCount = frameCount
        let log2n = vDSP_Length(log2(Float(frameCount)))
        self.log2n = log2n
        self.halfN = frameCount / 2
        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("vDSP_create_fftsetup failed")
        }
        self.fftSetup = setup

        var win = [Float](repeating: 0, count: frameCount)
        vDSP_hann_window(&win, vDSP_Length(frameCount), Int32(vDSP_HANN_NORM))
        self.window = win

        self.windowed = [Float](repeating: 0, count: frameCount)
        self.realp = [Float](repeating: 0, count: frameCount / 2)
        self.imagp = [Float](repeating: 0, count: frameCount / 2)
        self.magnitudes = [Float](repeating: 0, count: frameCount / 2)
        self.bands = [Float](repeating: 0, count: Self.bandCount)
        self.halfSorted = [Float](repeating: 0, count: Self.bandCount / 2)

        let halfBands = Self.bandCount / 2
        let binWidth = sampleRate / Float(frameCount)
        var ranges: [(start: Int, end: Int)] = []
        for idx in 0..<halfBands {
            let t0 = Float(idx) / Float(halfBands)
            let t1 = Float(idx + 1) / Float(halfBands)
            let f0 = Self.minFreq * pow(Self.maxFreq / Self.minFreq, t0)
            let f1 = Self.minFreq * pow(Self.maxFreq / Self.minFreq, t1)
            let bin0 = max(1, Int(f0 / binWidth))
            let bin1 = min(self.halfN - 2, Int(f1 / binWidth))
            ranges.append((start: bin0, end: max(bin0, bin1)))
        }
        self.bandRanges = ranges
    }

    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    /// Analyze an AVAudioPCMBuffer and return band magnitudes.
    func analyze(_ buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channelData = buffer.floatChannelData else {
            return [Float](repeating: 0, count: Self.bandCount)
        }
        lock.lock()
        defer { lock.unlock() }

        let frames = min(Int(buffer.frameLength), windowed.count)
        guard frames > 0 else { return [Float](repeating: 0, count: Self.bandCount) }

        vDSP_vmul(channelData[0], 1, window, 1, &windowed, 1, vDSP_Length(frames))
        if frames < windowed.count {
            for i in frames..<windowed.count { windowed[i] = 0 }
        }

        computeFFT()
        mapBands()
        mirrorBands()

        return bands
    }

    /// Analyze raw Float samples and return band magnitudes.
    func analyze(samples: [Float], offset: Int) -> [Float] {
        lock.lock()
        defer { lock.unlock() }

        let available = min(frameCount, samples.count - offset)
        guard available > 0 else { return [Float](repeating: 0, count: Self.bandCount) }

        samples.withUnsafeBufferPointer { buf in
            vDSP_vmul(buf.baseAddress! + offset, 1, window, 1, &windowed, 1, vDSP_Length(available))
        }
        if available < frameCount {
            for i in available..<frameCount { windowed[i] = 0 }
        }

        computeFFT()
        mapBands()
        mirrorBands()

        return bands
    }

    private func computeFFT() {
        windowed.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            base.withMemoryRebound(to: DSPComplex.self, capacity: halfN) { complexPtr in
                var split = DSPSplitComplex(realp: &realp, imagp: &imagp)
                vDSP_ctoz(complexPtr, 2, &split, 1, vDSP_Length(halfN))
            }
        }
        var split = DSPSplitComplex(realp: &realp, imagp: &imagp)
        vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
        vDSP_zvmags(&split, 1, &magnitudes, 1, vDSP_Length(halfN))
        var scale = 1.0 / Float(frameCount * frameCount)
        vDSP_vsmul(magnitudes, 1, &scale, &magnitudes, 1, vDSP_Length(halfN))
    }

    private func mapBands() {
        let halfBands = Self.bandCount / 2
        magnitudes.withUnsafeMutableBufferPointer { magPtr in
            guard let base = magPtr.baseAddress else { return }
            for idx in 0..<halfBands {
                let range = bandRanges[idx]
                let count = range.end - range.start + 1
                if count > 0 {
                    var sum: Float = 0
                    vDSP_sve(base + range.start, 1, &sum, vDSP_Length(count))
                    let avg = sum / Float(count)
                    let db = 10 * log10(max(avg, 1e-10))
                    bands[idx] = max(0, min(1, (db - Self.dbFloor) / Self.dbRange))
                } else {
                    bands[idx] = 0
                }
            }
        }
    }

    private func mirrorBands() {
        let halfBands = Self.bandCount / 2
        for idx in 0..<halfBands { halfSorted[idx] = bands[idx] }
        halfSorted[0..<halfBands].sort(by: >)
        for idx in 0..<halfBands {
            bands[halfBands - 1 - idx] = halfSorted[idx]
            bands[halfBands + idx] = halfSorted[idx]
        }
    }
}
