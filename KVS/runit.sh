# number of requests to run
num_requests=1024
# number of tokens in each random prompts, set to 0 to use template prompts
prompt_token_num=1024
# maximum number of new tokens to generate
max_new_tokens=1
# hicache storage location
hicache_storage_dir=db

export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=$hicache_storage_dir
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DISABLE_HASH=1

# origin sglang
python test.py --num-requests $num_requests \
    --prompt-token-num $prompt_token_num \
    --max-new-tokens $max_new_tokens \
    --output-file output.txt \
    |& tee test.log

rm -rf "$hicache_storage_dir"
python test.py --num-requests $num_requests \
    --prompt-token-num $prompt_token_num \
    --max-new-tokens $max_new_tokens \
    --output-file output_backend_file.txt \
    --hicache-storage-backend file \
    |& tee test_backend_file.log

rm -rf "$hicache_storage_dir"
python test.py --num-requests $num_requests \
    --prompt-token-num $prompt_token_num \
    --max-new-tokens $max_new_tokens \
    --output-file output_backend_lsm.txt \
    --hicache-storage-backend lsm \
    |& tee test_backend_lsm.log

rm -rf "$hicache_storage_dir"
python test.py --num-requests $num_requests \
    --prompt-token-num $prompt_token_num \
    --max-new-tokens $max_new_tokens \
    --output-file output_backend_blob.txt \
    --hicache-storage-backend blob \
    |& tee test_backend_blob.log

echo "=================== Settings ==================="
echo num_requests=$num_requests
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