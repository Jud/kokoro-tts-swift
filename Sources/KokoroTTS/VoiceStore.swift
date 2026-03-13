import Foundation

/// Loads and caches voice style embeddings from JSON files.
///
/// Each voice file contains a generic 256-dim embedding plus length-indexed
/// embeddings keyed by token count ("1", "2", ..., "510"). The reference
/// implementation selects `pack[len(ps)-1]` — the embedding calibrated for
/// the specific input token count. Using the wrong embedding (especially
/// the generic one for short inputs) produces garbled audio.
final class VoiceStore: Sendable {
    /// Voice name → length-indexed embeddings. Key 0 is the generic fallback.
    private let voicePacks: [String: VoicePack]

    /// Cached sorted voice names (computed once at init).
    private let sortedVoiceNames: [String]

    /// Style embedding dimension.
    static let styleDim = 256

    /// Load all voice embeddings from a directory of JSON files.
    init(directory: URL) throws {
        let fm = FileManager.default

        guard fm.fileExists(atPath: directory.path) else {
            throw KokoroError.modelsNotAvailable(directory)
        }

        var loaded = [String: VoicePack]()

        let files = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        for file in files where file.pathExtension == "json" {
            let voiceName = file.deletingPathExtension().lastPathComponent
            if let pack = try? Self.loadVoicePack(from: file) {
                loaded[voiceName] = pack
            }
        }

        self.voicePacks = loaded
        self.sortedVoiceNames = loaded.keys.sorted()
    }

    /// Get the embedding for a specific voice, calibrated for the given token count.
    ///
    /// Matches the reference: `ref_s = pack[len(ps)-1]`. Falls back to the
    /// nearest available length, then the generic embedding.
    func embedding(for voice: String, tokenCount: Int) throws -> [Float] {
        guard let pack = voicePacks[voice] else {
            let available = sortedVoiceNames.prefix(5).joined(separator: ", ")
            throw KokoroError.voiceNotFound(
                "\(voice) — available: \(available)...")
        }
        return pack.embedding(forTokenCount: tokenCount)
    }

    /// Get the generic embedding (for warmup or when token count is unknown).
    func embedding(for voice: String) throws -> [Float] {
        guard let pack = voicePacks[voice] else {
            let available = sortedVoiceNames.prefix(5).joined(separator: ", ")
            throw KokoroError.voiceNotFound(
                "\(voice) — available: \(available)...")
        }
        return pack.generic
    }

    /// Available voice preset names.
    var availableVoices: [String] {
        sortedVoiceNames
    }

    // MARK: - Private

    private static func loadVoicePack(from url: URL) throws -> VoicePack {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let generic = json["embedding"] as? [Double],
              !generic.isEmpty else {
            throw KokoroError.modelLoadFailed(
                "Invalid voice embedding: \(url.lastPathComponent)")
        }

        let genericFloat = generic.prefix(styleDim).map { Float($0) }

        // Load length-indexed embeddings ("1", "2", ..., "510").
        var indexed = [Int: [Float]]()
        for (key, value) in json {
            guard let idx = Int(key), let arr = value as? [Double], arr.count >= styleDim else {
                continue
            }
            indexed[idx] = arr.prefix(styleDim).map { Float($0) }
        }

        return VoicePack(generic: genericFloat, indexed: indexed)
    }
}

/// A voice's complete set of style embeddings indexed by token length.
struct VoicePack: Sendable {
    let generic: [Float]
    /// Token length → 256-dim style vector. Key = tokenCount - 1 (matching reference).
    let indexed: [Int: [Float]]

    /// Select the best embedding for a given token count.
    ///
    /// Matches reference: `pack[len(ps)-1]`. Falls back to nearest length,
    /// then the generic embedding.
    func embedding(forTokenCount count: Int) -> [Float] {
        let key = max(0, count - 1)

        // Exact match.
        if let emb = indexed[key] { return emb }

        // Find nearest available length.
        if !indexed.isEmpty {
            let nearest = indexed.keys.min(by: { abs($0 - key) < abs($1 - key) })!
            return indexed[nearest]!
        }

        return generic
    }
}
