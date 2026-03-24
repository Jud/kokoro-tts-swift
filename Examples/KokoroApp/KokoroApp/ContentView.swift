import KokoroCoreML
import SwiftUI

struct ContentView: View {
    @State private var audio = AudioEngine()
    @State private var recognizer = SpeechRecognizer()
    @State private var waveform = WaveformModel()
    @State private var levelMonitor = AudioLevelMonitor()
    @State private var selectedVoice = VoicePicker.defaultVoice
    @State private var pickerDismiss = false
    @State private var status: Status = .idle
    @State private var typedText = ""
    @State private var showTextInput = false
    @State private var conversationActive = false
    @FocusState private var textFieldFocused: Bool

    enum Status: Equatable {
        case idle, listening, synthesizing, speaking
    }

    private var displayText: String {
        if !typedText.isEmpty && (status == .speaking || status == .synthesizing) {
            return typedText
        }
        if !recognizer.transcript.isEmpty && status != .idle {
            return recognizer.transcript
        }
        switch status {
        case .idle: return showTextInput ? "" : "tap to speak"
        case .listening: return "listening..."
        case .synthesizing: return "thinking..."
        case .speaking: return ""
        }
    }

    private var micIcon: String {
        if !conversationActive { return "mic.fill" }
        switch status {
        case .listening: return "waveform"
        case .synthesizing: return "ellipsis"
        case .speaking: return "speaker.wave.2.fill"
        default: return "stop.fill"
        }
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
                .onTapGesture {
                    textFieldFocused = false
                    pickerDismiss = true
                }

            VStack(spacing: 0) {
                Spacer()

                WaveformView(model: waveform)
                    .frame(height: 120)
                    .padding(.horizontal, 20)

                if showTextInput && status == .idle {
                    HStack(spacing: 12) {
                        TextField("type something...", text: $typedText)
                            .font(.system(size: 16, design: .monospaced))
                            .foregroundStyle(.white)
                            .tint(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 12)
                            .background(Color.white.opacity(0.08), in: Capsule())
                            .focused($textFieldFocused)
                            .onSubmit { speakTypedText() }

                        Button(action: speakTypedText) {
                            Image(systemName: "arrow.up.circle.fill")
                                .font(.system(size: 32))
                                .foregroundStyle(.white)
                        }
                        .disabled(typedText.isEmpty)
                        .opacity(typedText.isEmpty ? 0.3 : 1)
                    }
                    .padding(.horizontal, 24)
                    .padding(.top, 32)
                    .frame(height: 80)
                } else {
                    Text(displayText)
                        .font(.system(size: 16, weight: .regular, design: .monospaced))
                        .foregroundStyle(.white.opacity(status == .speaking ? 0.9 : 0.4))
                        .multilineTextAlignment(.center)
                        .lineLimit(3)
                        .padding(.horizontal, 40)
                        .padding(.top, 32)
                        .frame(height: 80)
                }

                Spacer()

                VoicePicker(
                    voices: audio.availableVoices, selectedVoice: $selectedVoice,
                    dismissTrigger: $pickerDismiss
                )
                .padding(.bottom, 32)

                HStack(spacing: 24) {
                    Button {
                        showTextInput.toggle()
                        if showTextInput { textFieldFocused = true }
                    } label: {
                        Circle()
                            .fill(showTextInput ? Color.white.opacity(0.2) : Color.white.opacity(0.08))
                            .frame(width: 48, height: 48)
                            .overlay {
                                Image(systemName: "keyboard")
                                    .font(.system(size: 18))
                                    .foregroundStyle(.white.opacity(0.6))
                            }
                    }

                    Button(action: handleMicTap) {
                        Circle()
                            .fill(conversationActive ? Color.red : Color.white)
                            .frame(width: 72, height: 72)
                            .overlay {
                                Image(systemName: micIcon)
                                    .font(.system(size: 28, weight: .medium))
                                    .foregroundStyle(conversationActive ? .white : .black)
                            }
                            .scaleEffect(status == .listening ? 1.1 : 1.0)
                            .animation(.easeInOut(duration: 0.3), value: status)
                    }
                    .disabled(!audio.isReady)
                    .opacity(audio.isReady ? 1.0 : 0.3)

                    Circle()
                        .fill(Color.clear)
                        .frame(width: 48, height: 48)
                }
                .padding(.bottom, 50)
            }

            // Loading / error overlay
            if !audio.isReady {
                VStack(spacing: 16) {
                    if audio.error == nil {
                        ProgressView().tint(.white)
                    }
                    Text(audio.error ?? "loading models...")
                        .font(.system(size: 14, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.5))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 40)
                }
            }
        }
        .task { loadModels() }
        .onChange(of: recognizer.silenceDetected) {
            if recognizer.silenceDetected && status == .listening {
                commitSpeech()
            }
        }
        .onChange(of: recognizer.micAvailable) {
            if !recognizer.micAvailable {
                showTextInput = true
                status = .idle
            }
        }
        .onChange(of: audio.isSynthesizing) {
            if audio.isSynthesizing {
                status = .synthesizing
            } else if audio.isPlaying {
                status = .speaking
            }
        }
        .onChange(of: audio.isPlaying) {
            if !audio.isPlaying && !audio.isSynthesizing {
                if conversationActive {
                    startListening()
                } else {
                    status = .idle
                }
            }
        }
    }

    // MARK: - Actions

    private func commitSpeech() {
        let text = recognizer.transcript
        recognizer.stopListening()
        if !text.isEmpty {
            audio.speak(text: text, voice: selectedVoice)
        } else {
            status = .idle
            conversationActive = false
        }
    }

    private func handleMicTap() {
        if status == .listening {
            commitSpeech()
        } else if conversationActive {
            conversationActive = false
            recognizer.stopListening()
            audio.stop()
            status = .idle
        } else {
            showTextInput = false
            conversationActive = true
            startListening()
        }
    }

    private func startListening() {
        Task {
            if !recognizer.isAuthorized {
                guard await recognizer.requestAuthorization() else {
                    showTextInput = true
                    conversationActive = false
                    return
                }
            }
            recognizer.startListening()
            if !recognizer.micAvailable {
                showTextInput = true
                conversationActive = false
                return
            }
            status = .listening
        }
    }

    private func speakTypedText() {
        guard !typedText.isEmpty else { return }
        textFieldFocused = false
        let text = typedText
        typedText = ""
        audio.speak(text: text, voice: selectedVoice)
    }

    private func loadModels() {
        let bundledPath = Bundle.main.resourceURL?.appendingPathComponent("Models")
        let appSupportPath = KokoroEngine.defaultModelDirectory

        let modelDir: URL
        if let bundled = bundledPath, KokoroEngine.isDownloaded(at: bundled) {
            modelDir = bundled
        } else {
            modelDir = appSupportPath
        }

        audio.levelMonitor = levelMonitor
        recognizer.levelMonitor = levelMonitor
        waveform.startAnimating(monitor: levelMonitor)
        audio.load(modelDirectory: modelDir)

        if let first = audio.availableVoices.first, !audio.availableVoices.contains(selectedVoice) {
            selectedVoice = first
        }
    }
}

#Preview {
    ContentView()
}
