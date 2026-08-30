from client import ChunkedPrefillSpeculativeInferenceServerClient

def main():
    client = ChunkedPrefillSpeculativeInferenceServerClient()
    res = client.serve_speculative_batch(64, 'llama-3.2-1b', 'llama-3.3-70b')
    print('vLLM Speculative Server: ' + res['inference_batch_id'] + ' (Batch: ' + str(res['batch_size']) + ')')
    print('Throughput: ' + str(res['aggregate_throughput_tok_per_sec']) + ' tok/s | P99 TTFT: ' + str(res['time_to_first_token_p99_ms']) + 'ms')
    print('Acceptance Rate: ' + str(res['speculative_acceptance_rate_pct']) + '% | KV Fragmentation: ' + str(res['paged_kv_cache_fragmentation_pct']) + '%')

if __name__ == '__main__':
    main()
