PREFIX ?= /usr/local

.PHONY: build install uninstall clean

build:
	swift build -c release

install: build
	install -d "$(PREFIX)/bin"
	install ".build/release/kokoro-say" "$(PREFIX)/bin/kokoro-say"

uninstall:
	rm -f "$(PREFIX)/bin/kokoro-say"

clean:
	swift package clean
