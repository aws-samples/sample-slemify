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

build-training:
	@echo "NOTE: Training now uses the official unsloth/unsloth Docker image directly."
	@echo "This target is retained for building the deprecated custom training image."
	docker build -t ghcr.io/slemify/training:latest containers/training/

build-containers: build-data-pipeline build-training

clean:
	rm -f $(BINARY) $(BINARY)-linux-* $(BINARY)-darwin-*
