//! Time-to-first-token benchmark for the active translation backend.
//!
//! Select the backend with env before running:
//!   LI_TRANSLATE_BACKEND=http  cargo bench --bench translate_latency
//!   LI_TRANSLATE_BACKEND=candle cargo bench --features translate-candle --bench translate_latency
//!
//! `iter_custom` times only up to the first streamed item (first token), then drains the rest so
//! the model finishes cleanly. LLM calls are slow and non-deterministic, hence small sample size.

use criterion::{Criterion, criterion_group, criterion_main};
use futures_util::StreamExt;
use live_interpreter::translate::Translator;
use live_interpreter::types::Direction;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;

fn first_token_latency(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let translator = Translator::from_env(
        std::env::var("LI_OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".into()),
        std::env::var("LI_OLLAMA_MODEL").unwrap_or_else(|_| "translator:latest".into()),
    )
    .expect("failed to build translator");

    let mut group = c.benchmark_group("translate");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("first_token", |b| {
        b.iter_custom(|iters| {
            rt.block_on(async {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = Instant::now();
                    let mut stream = translator
                        .translate_stream("Hola, ¿cómo estás hoy?", &Direction::EsToEn)
                        .await
                        .expect("stream start");
                    // First decoded token == time-to-first-token boundary.
                    let _first = stream.next().await;
                    total += start.elapsed();
                    // Drain the remainder so the backend completes the generation.
                    while stream.next().await.is_some() {}
                }
                total
            })
        });
    });

    group.finish();
}

criterion_group!(benches, first_token_latency);
criterion_main!(benches);
