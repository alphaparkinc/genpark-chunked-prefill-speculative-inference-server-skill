class ChunkedPrefillSpeculativeInferenceServerClient:
    def serve_speculative_batch(self, batch_requests_count=128, draft_model_spec='qwen2.5-0.5b', target_model_spec='qwen2.5-72b'):
        return {
            'inference_batch_id': 'vlm_spc_8812',
            'batch_size': batch_requests_count,
            'paged_kv_cache_fragmentation_pct': 0.8,
            'chunked_prefill_chunk_size_tokens': 512,
            'speculative_acceptance_rate_pct': 86.4,
            'time_to_first_token_p99_ms': 18,
            'aggregate_throughput_tok_per_sec': 3820
        }
