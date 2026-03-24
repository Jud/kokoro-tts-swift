import SwiftUI

struct VoiceInfo {
    let id: String
    let accent: Accent
    let gender: Gender
    let name: String

    enum Accent: String, CaseIterable, Identifiable {
        case american = "a", british = "b", spanish = "e", french = "f"
        case hindi = "h", italian = "i", japanese = "j", portuguese = "p", chinese = "z"
        var id: String { rawValue }
        var flag: String {
            switch self {
            case .american: "🇺🇸";
            case .british: "🇬🇧";
            case .spanish: "🇪🇸"
            case .french: "🇫🇷";
            case .hindi: "🇮🇳";
            case .italian: "🇮🇹"
            case .japanese: "🇯🇵";
            case .portuguese: "🇧🇷";
            case .chinese: "🇨🇳"
            }
        }
    }

    enum Gender: String, CaseIterable, Identifiable {
        case female = "f", male = "m"
        var id: String { rawValue }
        var emoji: String { self == .female ? "👩" : "👨" }
    }

    static func parse(_ name: String) -> VoiceInfo? {
        guard name.count >= 3,
            let accent = Accent(rawValue: String(name.prefix(1))),
            let gender = Gender(rawValue: String(name[name.index(name.startIndex, offsetBy: 1)])),
            name[name.index(name.startIndex, offsetBy: 2)] == "_"
        else { return nil }
        return VoiceInfo(id: name, accent: accent, gender: gender, name: String(name.dropFirst(3)))
    }
}

// MARK: - Slot Pill

struct SlotPill<Item: Identifiable & Equatable>: View {
    let items: [Item]
    let label: (Item) -> String
    @Binding var selected: Item
    var onCommit: (() -> Void)?

    @Binding var isSpinning: Bool
    @State private var scrollPosition: Item.ID?

    private let itemHeight: CGFloat = 44
    private let reelHeight: CGFloat = 44 * 5

    var body: some View {
        pillBase
            .overlay { reelOverlay }
            .contentShape(Rectangle())
            .onTapGesture { handleTap() }
            .animation(.spring(duration: 0.25), value: isSpinning)
            .onChange(of: isSpinning) {
                if isSpinning {
                    scrollPosition = selected.id
                } else {
                    // Auto-commit when closed (e.g. by another pill opening)
                    if let id = scrollPosition, let item = items.first(where: { $0.id == id }) {
                        if item != selected {
                            selected = item
                            onCommit?()
                        }
                    }
                }
            }
    }

    private var pillBase: some View {
        Text(label(selected))
            .font(.system(size: 15, weight: .medium, design: .monospaced))
            .foregroundStyle(.white.opacity(isSpinning ? 0 : 0.85))
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background { Capsule().fill(Color.white.opacity(0.1)) }
    }

    @ViewBuilder
    private var reelOverlay: some View {
        if isSpinning {
            ZStack {
                Capsule()
                    .fill(Color.white.opacity(0.9))
                    .frame(height: itemHeight)

                reelScroll
            }
            .transition(.opacity)
        }
    }

    private var reelScroll: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(spacing: 0) {
                Color.clear.frame(height: itemHeight * 2)

                ForEach(items) { item in
                    reelItem(item)
                }

                Color.clear.frame(height: itemHeight * 2)
            }
            .scrollTargetLayout()
        }
        .scrollTargetBehavior(.viewAligned)
        .scrollPosition(id: $scrollPosition, anchor: .center)
        .frame(height: reelHeight)
        .mask { reelMask }
        .onAppear { scrollPosition = selected.id }
    }

    private func reelItem(_ item: Item) -> some View {
        let isCentered = item.id == (scrollPosition ?? selected.id)
        return Text(label(item))
            .font(.system(size: 15, weight: .medium, design: .monospaced))
            .foregroundStyle(isCentered ? .black : .white)
            .opacity(isCentered ? 1.0 : 0.3)
            .scaleEffect(isCentered ? 1.0 : 0.8)
            .frame(height: itemHeight)
            .frame(maxWidth: .infinity)
            .contentShape(Rectangle())
            .id(item.id)
            .onTapGesture { commitItem(item) }
            .animation(.easeOut(duration: 0.15), value: isCentered)
    }

    private var reelMask: some View {
        LinearGradient(
            stops: [
                .init(color: .clear, location: 0),
                .init(color: .black.opacity(0.3), location: 0.2),
                .init(color: .black, location: 0.4),
                .init(color: .black, location: 0.6),
                .init(color: .black.opacity(0.3), location: 0.8),
                .init(color: .clear, location: 1.0),
            ],
            startPoint: .top, endPoint: .bottom
        )
    }

    private func handleTap() {
        if isSpinning {
            if let id = scrollPosition, let item = items.first(where: { $0.id == id }) {
                selected = item
            }
            scrollPosition = selected.id
            onCommit?()
        } else {
            scrollPosition = selected.id
        }
        withAnimation(.spring(duration: 0.25)) {
            isSpinning.toggle()
        }
    }

    private func commitItem(_ item: Item) {
        selected = item
        scrollPosition = selected.id
        withAnimation(.spring(duration: 0.25)) {
            isSpinning = false
        }
        onCommit?()
    }
}

// MARK: - Voice Picker

struct VoicePicker: View {
    let voices: [String]
    @Binding var selectedVoice: String
    @Binding var dismissTrigger: Bool

    @State private var accent: VoiceInfo.Accent = .american
    @State private var gender: VoiceInfo.Gender = .female
    @State private var selectedName = NameItem(voiceId: defaultVoice, name: "heart")
    static let defaultVoice = "af_heart"
    @State private var accentOpen = false
    @State private var genderOpen = false
    @State private var nameOpen = false
    @State private var parsed: [VoiceInfo] = []

    private var availableAccents: [VoiceInfo.Accent] {
        let s = Set(parsed.map(\.accent))
        return VoiceInfo.Accent.allCases.filter { s.contains($0) }
    }

    private var availableGenders: [VoiceInfo.Gender] {
        let s = Set(parsed.filter { $0.accent == accent }.map(\.gender))
        return VoiceInfo.Gender.allCases.filter { s.contains($0) }
    }

    private var filteredVoices: [NameItem] {
        parsed
            .filter { $0.accent == accent && $0.gender == gender }
            .map { NameItem(voiceId: $0.id, name: $0.name) }
    }

    private func dismissAll() {
        withAnimation(.spring(duration: 0.25)) {
            if accentOpen { accentOpen = false }
            if genderOpen { genderOpen = false }
            if nameOpen { nameOpen = false }
        }
    }

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            Spacer()
            SlotPill(
                items: availableAccents, label: \.flag, selected: $accent, onCommit: fixup,
                isSpinning: $accentOpen)
            SlotPill(
                items: availableGenders, label: \.emoji, selected: $gender, onCommit: fixup,
                isSpinning: $genderOpen)
            SlotPill(items: filteredVoices, label: \.name, selected: $selectedName, isSpinning: $nameOpen)
            Spacer()
        }
        .padding(.horizontal, 20)
        .onAppear {
            reparse(); syncFromSelected()
        }
        .onChange(of: voices) { reparse() }
        .onChange(of: selectedName) { selectedVoice = selectedName.voiceId }
        .onChange(of: selectedVoice) { syncFromSelected() }
        .onChange(of: accentOpen) { if accentOpen { genderOpen = false; nameOpen = false } }
        .onChange(of: genderOpen) { if genderOpen { accentOpen = false; nameOpen = false } }
        .onChange(of: nameOpen) { if nameOpen { accentOpen = false; genderOpen = false } }
        .onChange(of: dismissTrigger) {
            if dismissTrigger {
                dismissAll()
                dismissTrigger = false
            }
        }
    }

    private func reparse() {
        parsed = voices.compactMap { VoiceInfo.parse($0) }
    }

    private func fixup() {
        let genders = availableGenders
        if !genders.contains(gender), let first = genders.first { gender = first }
        let names = filteredVoices
        if names.isEmpty { return }
        if !names.contains(selectedName), let first = names.first {
            selectedName = first
            selectedVoice = first.voiceId
        }
    }

    private func syncFromSelected() {
        if let v = VoiceInfo.parse(selectedVoice) {
            accent = v.accent
            gender = v.gender
            selectedName = NameItem(voiceId: v.id, name: v.name)
        }
    }
}

struct NameItem: Identifiable, Equatable {
    let voiceId: String
    let name: String
    var id: String { voiceId }
}
