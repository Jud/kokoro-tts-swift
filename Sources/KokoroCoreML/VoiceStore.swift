import Foundation

/// Loads and caches voice style embeddings from binary (.bin) or JSON files.
///
/// Each voice contains a generic 256-dim embedding plus length-indexed
/// embeddings keyed by token count (1–510). The reference implementation
/// selects `pack[len(ps)-1]` — the embedding calibrated for the specific
/// input token count.
///
/// Binary format (preferred, ~5x smaller than JSON):
///   Header: num_keys (UInt16 LE), dim (UInt16 LE)
///   Entries: key_id (UInt16 LE) + dim × Float32 LE values, sorted by key_id
///   Key 0 = generic embedding, keys 1–510 = length-indexed.
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

        var loaded: [String: VoicePack] = [:]

        let files = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        for file in files {
            let ext = file.pathExtension
            guard ext == "bin" || ext == "json" else { continue }
            let voiceName = file.deletingPathExtension().lastPathComponent
            guard loaded[voiceName] == nil else { continue }  // .bin takes priority by sort order
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
        try pack(for: voice).embedding(forTokenCount: tokenCount)
    }

    private func pack(for voice: String) throws -> VoicePack {
        guard let pack = voicePacks[voice] else {
            let available = sortedVoiceNames.prefix(5).joined(separator: ", ")
            throw KokoroError.voiceNotFound(
                "\(voice) — available: \(available)...")
        }
        return pack
    }

    /// Available voice preset names.
    var availableVoices: [String] {
        sortedVoiceNames
    }

    // MARK: - Private

    private static func loadVoicePack(from url: URL) throws -> VoicePack {
        if url.pathExtension == "bin" {
            return try loadBinaryVoicePack(from: url)
        }
        return try loadJSONVoicePack(from: url)
    }

    private static func loadBinaryVoicePack(from url: URL) throws -> VoicePack {
        let data = try Data(contentsOf: url)
        guard data.count >= 4 else {
            throw KokoroError.modelLoadFailed("Voice file too small: \(url.lastPathComponent)")
        }

        let numKeys = Int(
            data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt16.self).littleEndian })
        let dim = Int(
            data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 2, as: UInt16.self).littleEndian })
        let entrySize = 2 + dim * 4  // UInt16 key + dim × Float32

        guard data.count >= 4 + numKeys * entrySize else {
            throw KokoroError.modelLoadFailed("Voice file truncated: \(url.lastPathComponent)")
        }

        var generic: [Float]?
        var entries: [(Int, [Float])] = []

        data.withUnsafeBytes { buf in
            for i in 0..<numKeys {
                let offset = 4 + i * entrySize
                let keyId = Int(buf.loadUnaligned(fromByteOffset: offset, as: UInt16.self).littleEndian)
                var vec = [Float](repeating: 0, count: dim)
                for j in 0..<dim {
                    let bits = buf.loadUnaligned(fromByteOffset: offset + 2 + j * 4, as: UInt32.self)
                        .littleEndian
                    vec[j] = Float(bitPattern: bits)
                }
                if keyId == 0 {
                    generic = vec
                } else {
                    entries.append((keyId, vec))
                }
            }
        }

        guard let genericVec = generic else {
            throw KokoroError.modelLoadFailed("Missing generic embedding: \(url.lastPathComponent)")
        }

        var indexed = [[Float]?](repeating: nil, count: (entries.map(\.0).max() ?? -1) + 1)
        for (idx, vec) in entries { indexed[idx] = vec }

        return VoicePack(generic: genericVec, indexed: indexed)
    }

    private static func loadJSONVoicePack(from url: URL) throws -> VoicePack {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let generic = json["embedding"] as? [Double],
            !generic.isEmpty
        else {
            throw KokoroError.modelLoadFailed(
                "Invalid voice embedding: \(url.lastPathComponent)")
        }

        let genericFloat = generic.prefix(styleDim).map { Float($0) }

        var entries: [(Int, [Float])] = []
        for (key, value) in json {
            guard let idx = Int(key), let arr = value as? [Double], arr.count >= styleDim else {
                continue
            }
            entries.append((idx, arr.prefix(styleDim).map { Float($0) }))
        }
        var indexed = [[Float]?](repeating: nil, count: (entries.map(\.0).max() ?? -1) + 1)
        for (idx, vec) in entries { indexed[idx] = vec }

        return VoicePack(generic: genericFloat, indexed: indexed)
    }
}

/// A voice's complete set of style embeddings indexed by token length.
struct VoicePack: Sendable {
    let generic: [Float]
    /// Length-indexed style vectors. Index = token length key, nil = no embedding at that length.
    private let indexed: [[Float]?]
    /// Highest valid index in `indexed`.
    private let maxIndex: Int

    init(generic: [Float], indexed: [[Float]?]) {
        self.generic = generic
        self.indexed = indexed
        self.maxIndex = indexed.count - 1
    }

    /// Select the best embedding for a given token count.
    ///
    /// Matches reference: `pack[len(ps)-1]`. Falls back to nearest length,
    /// then the generic embedding.
    func embedding(forTokenCount count: Int) -> [Float] {
        let key = max(0, count - 1)

        // Exact match (O(1) array lookup).
        if key <= maxIndex, let emb = indexed[key] { return emb }

        // Nearest available: scan outward from key.
        if maxIndex >= 0 {
            let clamped = min(key, maxIndex)
            for dist in 0...maxIndex {
                let lo = clamped - dist
                let hi = clamped + dist
                if lo < 0, hi > maxIndex { break }
                if lo >= 0, let emb = indexed[lo] { return emb }
                if hi <= maxIndex, let emb = indexed[hi] { return emb }
            }
        }

        return generic
    }
}
