# number of requests to run
num_requests=1024
# number of tokens in each random prompts, set to 0 to use template prompts
prompt_token_num=1024
# maximum number of new tokens to generate
max_new_tokens=1

rm -rf db
rm -rf kv_cache_storage

# origin sglang
python test.py --num-requests $num_requests \
    --prompt-token-num $prompt_token_num \
    --max-new-tokens $max_new_tokens \
    --output-file output.txt \
    |& tee test.log

python test.py --num-requests $num_requests \
    --prompt-token-num $prompt_token_num \
    --max-new-tokens $max_new_tokens \
    --output-file test_backend_file.txt \
    --hicache-storage-backend file \
    |& tee test_backend_file.log

python test.py --num-requests $num_requests \
    --prompt-token-num $prompt_token_num \
    --max-new-tokens $max_new_tokens \
    --output-file test_backend_lsm.txt \
    --hicache-storage-backend lsm \
    |& tee test_backend_lsm.log

echo "=================== Settings ==================="
echo num_requests=$num_requests
echo prompt_token_num=$prompt_token_num
echo max_new_tokens=$max_new_tokens
echo "================== Results ==================="
echo "SGLang"
tail -n 1 output.txt
echo "KVS (warmup)"
tail -n 1 test_backend_file.txt
echo "KVS"
tail -n 1 test_backend_lsm.txt