import Testing
@testable import KokoroTTS

@Suite("Tokenizer")
struct TokenizerTests {
    private func makeTokenizer() throws -> Tokenizer {
        try Tokenizer.loadFromBundle()
    }

    @Test("Encode adds BOS and EOS")
    func encodeAddsBosEos() throws {
        let tok = try makeTokenizer()
        let ids = tok.encode("a")
        #expect(ids.first == Tokenizer.bosId)
        #expect(ids.last == Tokenizer.eosId)
        #expect(ids.count >= 3) // BOS + at least 1 token + EOS
    }

    @Test("Unknown characters silently dropped")
    func unknownCharsDropped() throws {
        let tok = try makeTokenizer()
        // Emoji is not in vocab — should be dropped
        let withEmoji = tok.encode("a🎉b")
        let without = tok.encode("ab")
        #expect(withEmoji == without)
    }

    @Test("Padding extends to target length")
    func padding() throws {
        let tok = try makeTokenizer()
        let ids = tok.encode("a")
        let padded = tok.pad(ids, to: 20)
        #expect(padded.count == 20)
        #expect(padded.last == Tokenizer.padId)
    }

    @Test("Padding truncates if too long")
    func paddingTruncates() throws {
        let tok = try makeTokenizer()
        let ids = Array(0..<50)
        let padded = tok.pad(ids, to: 10)
        #expect(padded.count == 10)
    }

    @Test("Max length enforced")
    func maxLength() throws {
        let tok = try makeTokenizer()
        let longIPA = String(repeating: "a", count: 1000)
        let ids = tok.encode(longIPA, maxLength: 100)
        #expect(ids.count <= 100)
        #expect(ids.last == Tokenizer.eosId)
    }

    @Test("Empty string produces BOS + EOS only")
    func emptyString() throws {
        let tok = try makeTokenizer()
        let ids = tok.encode("")
        #expect(ids == [Tokenizer.bosId, Tokenizer.eosId])
    }
}
