#!/bin/bash
BIN=./build_bench/tensor_run
OUT=benchmarks/gemm_roofline.csv

mkdir -p benchmarks
echo "impl,N,time_ms,gflops,LLC_misses" > "$OUT"

for impl in naive tiled; do
  for N in 256 512 1024 2048; do
    perf_out=$(perf stat -e LLC-load-misses "$BIN" "$impl" "$N" 2>&1)

    result_line=$(echo "$perf_out" | grep "^RESULT")
    misses=$(echo "$perf_out" | grep "LLC-load-misses" | awk '{print $1}' | tr -d ',')

    IFS=',' read -r _ i n ms gflops <<< "$result_line"
    echo "$i,$n,$ms,$gflops,$misses" >> "$OUT"
    echo "done: $impl N=$N -> misses=$misses"
  done
done

echo "Wrote $OUT"
