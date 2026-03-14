import Foundation

enum ModelDownloader {
    static let repo = "Jud/kokoro-tts-swift"
    static let tag = "models-v1"
    static let asset = "kokoro-models.tar.gz"

    static var assetURL: URL {
        URL(string: "https://github.com/\(repo)/releases/download/\(tag)/\(asset)")!
    }

    static func download(to directory: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)

        fputs("Downloading KokoroTTS models...\n", stderr)
        fputs("  \(assetURL)\n", stderr)

        let tmpDir = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try fm.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tmpDir) }

        let tarball = tmpDir.appendingPathComponent(asset)
        let tarballPath = tarball.path

        // Download with progress
        let sem = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var downloadResult: Result<Void, Error> = .success(())

        let task = URLSession.shared.downloadTask(with: assetURL) { url, response, error in
            if let error {
                downloadResult = .failure(error)
            } else if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                downloadResult = .failure(URLError(.badServerResponse))
            } else if let url {
                do {
                    try FileManager.default.moveItem(at: url, to: URL(fileURLWithPath: tarballPath))
                } catch {
                    downloadResult = .failure(error)
                }
            }
            sem.signal()
        }

        nonisolated(unsafe) var lastPct = -1
        let observation = task.progress.observe(\.fractionCompleted) { progress, _ in
            let pct = Int(progress.fractionCompleted * 100)
            if pct != lastPct {
                lastPct = pct
                fputs("\r  \(pct)%", stderr)
            }
        }

        task.resume()
        sem.wait()
        observation.invalidate()
        fputs("\r      \r", stderr)

        try downloadResult.get()

        // Extract
        fputs("  Extracting...\n", stderr)
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        proc.arguments = ["xzf", tarball.path, "-C", directory.path]
        try proc.run()
        proc.waitUntilExit()

        guard proc.terminationStatus == 0 else {
            fputs("  Extraction failed.\n", stderr)
            throw CocoaError(.fileReadCorruptFile)
        }

        fputs("  Models installed to \(directory.path)\n", stderr)
    }
}
