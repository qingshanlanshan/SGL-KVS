from typing import List, Set, Tuple
from sglang.utils import wait_for_server, print_highlight, terminate_process, launch_server_cmd
import argparse
import requests
import random
import time
import numpy as np
import torch
import string
import sglang as sgl
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
import json


# ----------------------------- Server & Runtime -----------------------------

def start_server(model_path="zai-org/GLM-4-9B-0414", hicache_ratio=1.5, hicache_storage_backend=None, port=31000):
    """Launch an sglang server with configurable parameters."""
    command = f"""
    python3 -m sglang.launch_server \
    --model-path {model_path} \
    --host 0.0.0.0 \
    --disable-overlap-schedule \
    --dtype float16 \
    --enable-metrics \
    --enable-hierarchical-cache \
    --hicache-ratio {hicache_ratio} \
    {"--hicache-storage-backend " + hicache_storage_backend if hicache_storage_backend else ""} \
    """
    print(f"Launching server with command: {command}")
    server_process, actual_port = launch_server_cmd(command, port=port)
    wait_for_server(f"http://localhost:{actual_port}")
    return server_process, actual_port


def get_cache_hit_rate_from_metrics(port: int) -> tuple:
    """Read cached/prompt token totals and hit rate from the server's /metrics endpoint."""
    try:
        metrics_response = requests.get(f"http://localhost:{port}/metrics", timeout=10)
        if metrics_response.status_code == 200:
            metrics_text = metrics_response.text

            cached_tokens_total = 0.0
            prompt_tokens_total = 0.0

            for line in metrics_text.split('\n'):
                if line.startswith('sglang:cached_tokens_total{'):
                    try:
                        cached_tokens_total = float(line.split('} ')[1])
                    except (IndexError, ValueError):
                        continue
                if line.startswith('sglang:prompt_tokens_total{'):
                    try:
                        prompt_tokens_total = float(line.split('} ')[1])
                    except (IndexError, ValueError):
                        continue

            if prompt_tokens_total > 0:
                return cached_tokens_total, prompt_tokens_total, cached_tokens_total / prompt_tokens_total
    except Exception:
        pass
    return 0.0, 0.0, 0.0


def generate_batch(qas, port=31000) -> List[str]:
    """Run a batch of prompts and return only the generated continuations (without the prompts)."""
    @sgl.function
    def multi_turns(s, qas):
        for qa in qas:
            s += qa["prompt"]
            s += sgl.gen(max_tokens=qa["new_tokens"], ignore_eos=True)

    responses = multi_turns.run_batch(
        qas,
        temperature=0,
        backend=RuntimeEndpoint(f"http://localhost:{port}"),
        num_threads=4,
        progress_bar=True,
    )
    # Strip the prompts from outputs to keep only model continuations.
    return [resp.text()[len(item['qas'][0]['prompt']):] for resp, item in zip(responses, qas)]


def load_dataset(json_path, max_new_tokens=32):
    """Load a list of instructions from a JSON file and wrap them into the expected 'qas' format."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = data["instructions"] if isinstance(data, dict) and "instructions" in data else data

    multi_qas = []
    for item in dataset:
        multi_qas.append({
            "qas": [{
                "prompt": item,
                "new_tokens": max_new_tokens
            }]
        })
    return multi_qas


def build_chunk_pool_char(chunk_token_num: int, pool_size: int,
                          charset: str = string.ascii_letters + string.digits) -> List[str]:
    """
    Build a character-based pseudo-token chunk pool. Each chunk has exactly `chunk_token_num` characters.
    This approximates token counts using characters.
    """
    pool = []
    for _ in range(pool_size):
        pool.append(''.join(random.choices(charset, k=chunk_token_num)))
    return pool


# ------------------- Prefix-only cache-ratio control -------------------
class PrefixChunkController:
    """
    Generator that uses a finite chunk pool and enforces *prefix-only* reuse accounting:

    - Only prefix matches (continuous match from position 0) are considered hits.
    - A prefix at position l (1-based) counts toward the credited cache ratio for the current request
      only if this exact prefix appeared at least `min_hits_to_count` times *before* the current request.
      First appearances (and those below the threshold) do not contribute to credited hits.
    - All chunks are drawn from the given finite `chunk_pool`.
    """

    def __init__(self,
                 chunk_size: int,
                 chunks_per_prompt: int,
                 chunk_pool: List[str],
                 min_hits_to_count: int = 3):
        self.k = chunk_size
        self.B = chunks_per_prompt
        self.chunk_pool = chunk_pool
        self.min_hits_to_count = min_hits_to_count

        # prefix_counts[l]: dict mapping a prefix (tuple of length l+1) to prior occurrence count
        self.prefix_counts: List[dict] = [dict() for _ in range(self.B)]
        self.total_tokens = 0
        self.cached_tokens = 0

    def _rand_block(self) -> str:
        return random.choice(self.chunk_pool)

    def _choose_known_prefix(self, H: int) -> Tuple[str, ...]:
        """Pick an existing prefix of length H uniformly from historical prefixes."""
        candidates = list(self.prefix_counts[H - 1].keys())
        return random.choice(candidates)

    def _first_chunk_breaking(self, prefix: Tuple[str, ...], H: int) -> str:
        """
        Break the prefix at position H:
        Try to find a chunk c so that (prefix + (c,)) is a *new* (H+1)-length prefix.
        If all candidates are already known, fall back to random selection.
        """
        if H >= len(self.prefix_counts) or not self.prefix_counts[H]:
            return self._rand_block()
        for c in random.sample(self.chunk_pool, k=len(self.chunk_pool)):
            if (prefix + (c,)) not in self.prefix_counts[H]:
                return c
        return self._rand_block()

    def make_prompt(self, target_ratio: float) -> tuple[str, float]:
        """
        Create a prompt that aims for a desired number of credited prefix hits.

        Returns:
            prompt (str): the concatenated chunk string.
            credited_request_hit_ratio (float): credited hits / total chunks in this prompt.
        """
        desired_hits = int(round(target_ratio * self.B))

        # If we have no recorded 1-chunk prefixes yet, we can't hit anything.
        H = 0 if not self.prefix_counts[0] else min(desired_hits, self.B)

        chunks: List[str] = []
        prefix: Tuple[str, ...] = tuple()

        if H > 0:
            prefix = self._choose_known_prefix(H)
            chunks.extend(prefix)

        if H < self.B:
            c_break = self._first_chunk_breaking(prefix, H)
            chunks.append(c_break)
            while len(chunks) < self.B:
                chunks.append(self._rand_block())

        # --- Account (count first, then record) ---
        self.total_tokens += self.B * self.k

        creditable_hits = 0
        for l in range(1, H + 1):
            pt = tuple(chunks[:l])
            prev_occ = self.prefix_counts[l - 1].get(pt, 0)
            if prev_occ >= self.min_hits_to_count:
                creditable_hits += 1

        self.cached_tokens += creditable_hits * self.k

        # Record occurrences for all prefixes of this request
        for l in range(1, self.B + 1):
            pt = tuple(chunks[:l])
            self.prefix_counts[l - 1][pt] = self.prefix_counts[l - 1].get(pt, 0) + 1

        prompt = ''.join(chunks)
        return prompt, (creditable_hits / self.B if self.B else 0.0)

    def cumulative_ratio(self) -> float:
        """Return cumulative credited cache ratio = cached_tokens / total_tokens."""
        return (self.cached_tokens / self.total_tokens) if self.total_tokens else 0.0


def generate_synthetic_sequences_with_cache_control(
    intervals,                # list of counts per stage, e.g. [1000, 1000, 1000, 1000]
    target_hit_rates,         # list of targets per stage, in [0,1]
    seq_length=8192,          # total tokens per prompt (must be divisible by chunk_size)
    chunk_size=256,           # tokens per chunk
    max_new_tokens=1,
    chunk_pool: List[str] = None,   # finite chunk pool (required)
    min_hits_to_count: int = 3,     # hit threshold for a prefix to be credited
):
    """
    Generate synthetic prompts composed of chunks pulled from a finite pool, targeting
    specific credited prefix hit rates per interval.
    """
    assert seq_length % chunk_size == 0, "seq_length must be divisible by chunk_size"
    B = seq_length // chunk_size
    assert chunk_pool is not None and len(chunk_pool) > 0, "chunk_pool must not be empty"

    ctrl = PrefixChunkController(chunk_size, B, chunk_pool=chunk_pool, min_hits_to_count=min_hits_to_count)
    all_requests = []
    interval_stats = []
    prev_end = 0

    print("=== Generating (Pool-based, Prefix-only, hit-threshold controlled) ===", flush=True)
    for stage_idx, (count, target_ratio) in enumerate(zip(intervals, target_hit_rates)):
        stage_start = prev_end + 1
        stage_end = prev_end + count
        print(f"\nStage {stage_idx+1}: Generating {count} sequences (requests {stage_start} to {stage_end})", flush=True)
        print(f"Target prefix-hit ratio: {target_ratio:.2f}", flush=True)

        start_total = ctrl.total_tokens
        start_cached = ctrl.cached_tokens

        for t in range(count):
            prompt, req_hit = ctrl.make_prompt(target_ratio)
            all_requests.append({"qas": [{"prompt": prompt, "new_tokens": max_new_tokens}]})
            if ((t + 1) % 100 == 0) or (t == count - 1):
                print(f"  Progress: {t+1}/{count}, Cumulative cache ratio: {ctrl.cumulative_ratio():.4f}")

        stage_total = ctrl.total_tokens - start_total
        stage_cached = ctrl.cached_tokens - start_cached
        stage_ratio = (stage_cached / stage_total) if stage_total else 0.0

        interval_stats.append({
            "stage": stage_idx + 1,
            "interval": f"{stage_start}-{stage_end}",
            "target_ratio": target_ratio,
            "stage_ratio": stage_ratio,
            "cumulative_ratio": ctrl.cumulative_ratio(),
            "total_tokens": ctrl.total_tokens,
            "cached_tokens": ctrl.cached_tokens,
        })

        print(f"Stage {stage_idx+1} Complete:")
        print(f"  Stage cache ratio: {stage_ratio:.4f} (target {target_ratio:.2f})")
        print(f"  Cumulative cache ratio: {ctrl.cumulative_ratio():.4f}")
        print(f"  Total tokens processed: {ctrl.total_tokens}")
        print(f"  Cached tokens: {ctrl.cached_tokens}")

        prev_end = stage_end

    print_highlight("\n=== Cache Ratio Statistics Summary ===")
    for stat in interval_stats:
        print(f"Stage {stat['stage']} ({stat['interval']}): "
              f"Target={stat['target_ratio']:.2f}, StageActual={stat['stage_ratio']:.4f}, "
              f"Cumulative={stat['cumulative_ratio']:.4f}")

    return all_requests, interval_stats


# ----------------------------- CLI & Entrypoint -----------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    # Data source selection
    parser.add_argument("--data-source", type=str, default="synthetic",
                        choices=["synthetic", "real"],
                        help="Use synthetic generation or a real dataset from file (default: synthetic)")
    parser.add_argument("--dataset-path", type=str, default="/home/mengke/code/data_preprocess/final_data.json",
                        help="Path to dataset JSON file when using real data")

    # Model / server configuration
    parser.add_argument("--model-path", type=str, default="zai-org/GLM-4-9B-0414",
                        help="Model path (default: meta-llama/Llama-3.1-8B)")
    parser.add_argument("--port", type=int, default=31000, help="Server port (default: 31000)")

    # Synthetic generation configuration
    parser.add_argument("--seq-length", type=int, default=8192,
                        help="Tokens per synthetic prompt (must be divisible by chunk-size; default: 8192)")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="Tokens per aligned chunk (default: 256)")
    parser.add_argument("--max-new-tokens", type=int, default=1,
                        help="Maximum new tokens to generate per request (default: 1)")

    # HiCache config (passed through to server)
    parser.add_argument("--hicache-ratio", type=float, default=1.01,
                        help="Hierarchical cache size ratio (default: 1.01)")
    parser.add_argument("--hicache-storage-backend", type=str, default=None,
                        help="Storage backend for hierarchical cache (e.g., 'file', 'lsm')")

    # Interval targets
    parser.add_argument("--intervals", type=int, nargs="+",
                        default=[1000, 1000, 1000, 1000, 1000, 1000, 1000],
                        help="Number of requests in each stage")
    parser.add_argument("--target-hit-rates", type=float, nargs="+",
                        default=[0.1, 0.3, 0.5, 0.7, 0.5, 0.3, 0.1],
                        help="Target credited prefix hit rates per stage (values in [0, 1])")

    # Output
    parser.add_argument("--output-file", type=str, default="output.txt",
                        help="Output file path (default: output.txt)")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=47, help="Random seed (default: 47)")

    # Finite chunk pool (character-based)
    parser.add_argument("--pool-size", type=int, default=4,
                        help="Finite chunk pool size (default: 4)")
    parser.add_argument("--min-hits-to-count", type=int, default=1,
                        help="Prefix hit threshold: a prefix counts only after it has appeared at least this many times previously (default: 1)")
    parser.add_argument("--chunk-charset", type=str, default=string.ascii_letters + string.digits,
                        help="Character set used to build the character-based chunk pool")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Validate intervals and hit rates
    if len(args.intervals) != len(args.target_hit_rates):
        raise ValueError("Number of intervals must match number of target hit rates")

    # Prepare requests
    if args.data_source == "real":
        print_highlight(f"Loading real dataset from {args.dataset_path}")
        multi_qas = load_dataset(args.dataset_path, args.max_new_tokens)
        interval_stats = None
    else:
        # Build a finite character-based chunk pool
        chunk_pool = build_chunk_pool_char(
            chunk_token_num=args.chunk_size,
            pool_size=args.pool_size,
            charset=args.chunk_charset,
        )
        print_highlight(f"Chunk pool built: size={len(chunk_pool)} (character-based)")

        print("Generating synthetic sequences with cache control", flush=True)
        multi_qas, interval_stats = generate_synthetic_sequences_with_cache_control(
            args.intervals,
            args.target_hit_rates,
            seq_length=args.seq_length,
            chunk_size=args.chunk_size,
            max_new_tokens=args.max_new_tokens,
            chunk_pool=chunk_pool,
            min_hits_to_count=args.min_hits_to_count,
        )

    # Start server
    server_process, port = start_server(
        model_path=args.model_path,
        hicache_ratio=args.hicache_ratio,
        hicache_storage_backend=args.hicache_storage_backend,
        port=args.port
    )

    # Execute requests in intervals
    print_highlight("\n=== Starting Request Generation ===")
    start_time = time.time()

    results = []
    interval_results = []
    prev = 0
    prev_cached, prev_total = 0, 0   # for interval diff

    interval_results = []
    for stage_idx, count in enumerate(args.intervals):
        stage_qas = multi_qas[prev: prev + count]

        # interval 开始计时
        t0 = time.time()
        stage_results = generate_batch(stage_qas, port)
        results.extend(stage_results)
        t1 = time.time()

        interval_time = t1 - t0
        avg_interval_time = interval_time / count if count > 0 else 0.0

        # 查询服务端 metrics（累计值）
        cached_tokens, total_tokens, hit_rate_cum = get_cache_hit_rate_from_metrics(port)

        # 区间差分
        delta_cached = cached_tokens - prev_cached
        delta_total = total_tokens - prev_total
        hit_rate_interval = delta_cached / delta_total if delta_total > 0 else 0.0

        # 生成一行字符串
        log_line = (
            f"[Stage {stage_idx+1}] Requests {prev+1}-{prev+count} | "
            f"Interval hit rate: {hit_rate_interval:.4f} ({delta_cached}/{delta_total}) | "
            f"Cumulative hit rate: {hit_rate_cum:.4f} ({cached_tokens}/{total_tokens}) | "
            f"Interval time: {interval_time:.2f}s | "
            f"Avg per request: {avg_interval_time:.2f}s"
        )

        # 打印到屏幕
        print_highlight(log_line)

        # 保存到 interval_results
        interval_results.append(log_line)

        prev_cached, prev_total = cached_tokens, total_tokens
        prev += count

    elapsed_time = time.time() - start_time
    total_requests = len(multi_qas)

    # Print summary
    print_highlight(f"\n== Completed {total_requests} requests ==")
    print_highlight(f"== Data source: {args.data_source} ==")
    print_highlight(f"== Total time: {elapsed_time:.2f} seconds ==")
    print_highlight(f"== Average time per request: {elapsed_time / total_requests:.2f} seconds ==")

    # Get final cache metrics from server
    cached_tokens, total_tokens, hit_rate = get_cache_hit_rate_from_metrics(port)
    if total_tokens > 0:
        print_highlight(f"== Final server-reported cache hit rate: {hit_rate:.4f} ==")

    # Save results
    with open(args.output_file, "w") as f:
        f.write("=== Configuration ===\n")
        f.write(f"Data source: {args.data_source}\n")
        if args.data_source == "synthetic":
            f.write(f"Sequence length: {args.seq_length}\n")
            f.write(f"Chunk size: {args.chunk_size}\n")
            f.write(f"Intervals: {args.intervals}\n")
            f.write(f"Target hit rates: {args.target_hit_rates}\n\n")

            if interval_stats:
                f.write("=== Synthetic Generation Statistics ===\n")
                for stat in interval_stats:
                    f.write(f"Stage {stat['stage']} ({stat['interval']}): "
                            f"Target={stat['target_ratio']:.2f}, "
                            f"StageActual={stat['stage_ratio']:.4f}, "
                            f"Cumulative={stat['cumulative_ratio']:.4f}\n")
                f.write("\n")

        f.write("=== Interval Server Cache Stats ===\n")
        for line in interval_results:
            f.write(line + "\n")
        f.write("\n")

        f.write("=== Sample Requests and Responses ===\n")
        for i in range(min(10, len(results))):
            f.write(f"Request {i + 1}:\n")
            f.write(f"  Prompt: {multi_qas[i]['qas'][0]['prompt'][:50]}...\n")
            f.write(f"  Response: {results[i]}\n\n")

        f.write(f"\n=== Performance Summary ===\n")
        f.write(f"Total requests: {total_requests}\n")
        f.write(f"Total time: {elapsed_time:.2f} seconds\n")
        f.write(f"Average latency: {elapsed_time / total_requests:.2f} seconds\n")
        if total_tokens > 0:
            f.write(f"Final server cache hit rate: {hit_rate:.4f}\n")
            f.write(f"Total tokens processed: {total_tokens:.0f}\n")
            f.write(f"Cached tokens: {cached_tokens:.0f}\n")

    print_highlight(f"Output saved to {args.output_file}")

    # Cleanup
    time.sleep(2)
    terminate_process(server_process)


if __name__ == "__main__":
    main()
