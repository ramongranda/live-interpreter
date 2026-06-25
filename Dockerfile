FROM rust:1.96-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY static ./static
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates ffmpeg pipewire-bin jq curl \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/olares-voice-translator /usr/local/bin/olares-voice-translator
COPY scripts ./scripts
COPY docs ./docs
COPY static ./static
ENV OVT_BIND=0.0.0.0:8787
EXPOSE 8787
CMD ["olares-voice-translator"]
