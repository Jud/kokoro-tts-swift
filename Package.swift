// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "KokoroCoreML",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "KokoroCoreML", targets: ["KokoroCoreML"]),
        .executable(name: "kokoro", targets: ["CLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/Jud/swift-bart-g2p.git", from: "0.4.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
        .package(url: "https://github.com/edgeengineer/cbor.git", .upToNextMinor(from: "0.0.6")),
    ],
    targets: [
        .target(
            name: "KokoroCoreML",
            dependencies: [
                .product(name: "BARTG2P", package: "swift-bart-g2p")
            ],
            path: "Sources/KokoroCoreML",
            resources: [
                .process("Resources")
            ]
        ),
        .executableTarget(
            name: "CLI",
            dependencies: [
                "KokoroCoreML",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "CBOR", package: "cbor"),
            ],
            path: "Sources/CLI"
        ),
        .testTarget(
            name: "KokoroCoreMLTests",
            dependencies: ["KokoroCoreML"],
            path: "Tests/KokoroCoreMLTests",
            resources: [.process("kokoro_g2p_reference.json")]
        ),
    ]
)
