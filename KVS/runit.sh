# number of requests to run
num_requests=256
# number of tokens in each random prompts, set to 0 to use template prompts
prompt_token_num=1024
# maximum number of new tokens to generate
max_new_tokens=4
# compress kvcache in kvstore
kvstore_compress=true

rm -rf db

# origin sglang
python test.py --num-requests $num_requests \
    --prompt-token-num $prompt_token_num \
    --max-new-tokens $max_new_tokens \
    --output-file output.txt \
    --enable-hierarchical-cache \
    --hicache-ratio 1.5 \
    |& tee test.log

python test.py --num-requests $num_requests \
    --prompt-token-num $prompt_token_num \
    --max-new-tokens $max_new_tokens \
    --output-file output_kv.txt \
    --enable-hierarchical-cache \
    --hicache-ratio 1.5 \
    --enable-kvstore \
    $( [ "$kvstore_compress" = true ] && echo "--kvstore-compress" ) \
    |& tee test_kv.log

diff output.txt output_kv.txt |& tee diff.txt

echo "=================== Settings ==================="
echo num_requests=$num_requests
echo prompt_token_num=$prompt_token_num
echo max_new_tokens=$max_new_tokens
echo kvstore_compress=$kvstore_compress
echo "================== Results ==================="
echo "SGLang"
tail -n 1 output.txt
echo "KVS"
tail -n 1 output_kv.txt