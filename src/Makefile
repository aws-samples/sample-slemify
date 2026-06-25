BINARY := slemify
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
LDFLAGS := -ldflags "-s -w -X main.version=$(VERSION)"

.PHONY: build test clean all

build:
	go build $(LDFLAGS) -o $(BINARY) .

test:
	go test ./...

test-verbose:
	go test -v ./...

# Cross-platform builds
build-linux-amd64:
	GOOS=linux GOARCH=amd64 go build $(LDFLAGS) -o $(BINARY)-linux-amd64 .

build-linux-arm64:
	GOOS=linux GOARCH=arm64 go build $(LDFLAGS) -o $(BINARY)-linux-arm64 .

build-darwin-arm64:
	GOOS=darwin GOARCH=arm64 go build $(LDFLAGS) -o $(BINARY)-darwin-arm64 .

all: build-linux-amd64 build-linux-arm64 build-darwin-arm64

# Container images
build-data-pipeline:
	docker build -t ghcr.io/slemify/data-pipeline:latest containers/data-pipeline/

build-gguf-convert:
	docker build -t ghcr.io/slemify/gguf-convert:latest containers/gguf-convert/

build-containers: build-data-pipeline build-gguf-convert

clean:
	rm -f $(BINARY) $(BINARY)-linux-* $(BINARY)-darwin-*
