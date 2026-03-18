PREFIX ?= /usr/local

.PHONY: build install uninstall clean

build:
	swift build -c release

install: build
	install -d "$(PREFIX)/bin"
	install ".build/release/kokoro" "$(PREFIX)/bin/kokoro"

uninstall:
	rm -f "$(PREFIX)/bin/kokoro"

clean:
	swift package clean
