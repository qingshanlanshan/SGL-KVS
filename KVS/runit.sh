# number of tokens in each random prompts, set to 0 to use template prompts
prompt_token_num=8192
# maximum number of new tokens to generate
max_new_tokens=1

export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DISABLE_HASH=1
export CUDA_VISIBLE_DEVICES=1
export LD_PRELOAD=/lib/x86_64-linux-gnu/libpthread.so.0:/lib/x86_64-linux-gnu/libjemalloc.so.2

hicache_storage_dir=/mnt/gds_nvme/weiping
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=$hicache_storage_dir/file
python test.py \
    --seq-length $prompt_token_num \
    --max-new-tokens $max_new_tokens \
    --output-file output_backend_file.txt \
    --hicache-storage-backend file \
    > test_backend_file.log 2>&1

hicache_storage_dir=/mnt/gds_nvme/mkhe
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=$hicache_storage_dir/lsm
python test.py \
    --seq-length $prompt_token_num \
    --max-new-tokens $max_new_tokens \
    --output-file output_backend_lsm.txt \
    --hicache-storage-backend lsm \
    > test_backend_lsm.log 2>&1

export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=$hicache_storage_dir/blob
python test.py \
    --seq-length $prompt_token_num \
    --max-new-tokens $max_new_tokens \
    --output-file output_backend_blob.txt \
    --hicache-storage-backend lsm \
    > test_backend_lsm.log 2>&1

echo "=================== Settings ==================="
echo prompt_token_num=$prompt_token_num
echo max_new_tokens=$max_new_tokens
echo "================== Results ==================="
echo "SGLang"
tail -n 1 output.txt
echo "backend file"
tail -n 1 output_backend_file.txt
echo "backend lsm"
tail -n 1 output_backend_lsm.txt
echo "backend blob"
tail -n 1 output_backend_blob.txt