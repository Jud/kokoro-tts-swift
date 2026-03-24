import Observation
import SwiftUI
import os

/// Thread-safe band levels shared between audio threads and the UI.
final class AudioLevelMonitor: @unchecked Sendable {
    static let bandCount = SpectrumAnalyzer.bandCount

    private let lock = OSAllocatedUnfairLock<[Float]>(
        initialState: [Float](repeating: 0, count: bandCount)
    )

    func pushBands(_ bands: [Float]) {
        lock.withLock { state in
            for i in 0..<min(bands.count, state.count) {
                state[i] = bands[i]
            }
        }
    }

    func readBands(into destination: inout [Float]) {
        lock.withLockUnchecked { state in
            for i in 0..<min(state.count, destination.count) {
                destination[i] = state[i]
            }
        }
    }

    func reset() {
        lock.withLock { state in
            for i in 0..<state.count { state[i] = 0 }
        }
    }
}

/// Spectrum-driven waveform model. Reads band levels at 30 Hz with exponential smoothing.
@Observable
@MainActor
final class WaveformModel {
    static let sampleCount = AudioLevelMonitor.bandCount
    private static let smoothingFactor: CGFloat = 0.45

    var samples = [CGFloat](repeating: 0, count: sampleCount)

    private var smoothed = [CGFloat](repeating: 0, count: sampleCount)
    private var rawBands = [Float](repeating: 0, count: sampleCount)
    private var timer: Timer?
    private weak var monitor: AudioLevelMonitor?

    private var observers: [Any] = []

    func startAnimating(monitor: AudioLevelMonitor) {
        self.monitor = monitor
        guard timer == nil else { return }
        startTimer()
        guard observers.isEmpty else { return }
        let nc = NotificationCenter.default
        observers.append(
            nc.addObserver(
                forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main
            ) { [weak self] _ in Task { @MainActor in self?.stopTimer() } })
        observers.append(
            nc.addObserver(
                forName: UIApplication.willEnterForegroundNotification, object: nil, queue: .main
            ) { [weak self] _ in Task { @MainActor in self?.startTimer() } })
    }

    func stopAnimating() {
        stopTimer()
        for obs in observers { NotificationCenter.default.removeObserver(obs) }
        observers.removeAll()
        monitor = nil
        for i in 0..<Self.sampleCount { smoothed[i] = 0; samples[i] = 0 }
    }

    private func startTimer() {
        guard timer == nil, monitor != nil else { return }
        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }

    private func tick() {
        if let monitor {
            monitor.readBands(into: &rawBands)
        } else {
            for i in 0..<rawBands.count { rawBands[i] = 0 }
        }
        for i in 0..<Self.sampleCount {
            let raw = CGFloat(rawBands[i])
            smoothed[i] += (raw - smoothed[i]) * Self.smoothingFactor
            samples[i] = smoothed[i]
        }
    }
}

/// Smooth filled waveform mirrored above/below center, driven by frequency bands.
struct WaveformView: View {
    var model: WaveformModel

    var body: some View {
        Canvas { context, size in
            let path = Self.symmetricPath(samples: model.samples, size: size)
            context.fill(path, with: .color(.white.opacity(0.9)))
        }
    }

    private static func symmetricPath(samples: [CGFloat], size: CGSize) -> Path {
        guard samples.count >= 2 else { return Path() }

        let midY = size.height / 2
        let amplitude = size.height / 2
        let step = size.width / CGFloat(samples.count - 1)

        var path = Path()

        var prevX: CGFloat = 0
        var prevY = midY - samples[0] * amplitude
        path.move(to: CGPoint(x: 0, y: prevY))

        for i in 1..<samples.count {
            let x = CGFloat(i) * step
            let y = midY - samples[i] * amplitude
            let cpX = (prevX + x) / 2
            path.addCurve(
                to: CGPoint(x: x, y: y),
                control1: CGPoint(x: cpX, y: prevY),
                control2: CGPoint(x: cpX, y: y)
            )
            prevX = x
            prevY = y
        }

        prevY = midY + samples[samples.count - 1] * amplitude
        path.addLine(to: CGPoint(x: prevX, y: prevY))

        for i in stride(from: samples.count - 2, through: 0, by: -1) {
            let x = CGFloat(i) * step
            let y = midY + samples[i] * amplitude
            let cpX = (prevX + x) / 2
            path.addCurve(
                to: CGPoint(x: x, y: y),
                control1: CGPoint(x: cpX, y: prevY),
                control2: CGPoint(x: cpX, y: y)
            )
            prevX = x
            prevY = y
        }

        path.closeSubpath()
        return path
    }
}
