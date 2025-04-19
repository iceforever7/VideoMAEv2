echo "开始测试，记录资源消耗..." | tee -a /home/vllm/CN/test_log/performance_log.txt
start_time=$(date +%s)

python run_class_finetuning.py \
        --model vit_giant_patch14_224 \
        --data_set HMDB51 \
        --nb_classes 51 \
        --data_path /root/autodl-tmp/hmdb51_org\
        --finetune /root/autodl-tmp/key+att.pt\
        --log_dir /root/autodl-tmp/test_log \
        --output_dir /root/autodl-tmp/test_results \
        --input_size 224 \
        --short_side_size 224 \
        --num_frames 16 \
        --sampling_rate 2\
        --test_num_segment 3 \
        --test_num_crop 3 \
        --batch_size 3 \
        --num_workers 10 \
        --eval \

        2>&1 | tee -a /root/autodl-tmp/test_log/performance_log.txt

end_time=$(date +%s)
total_time=$((end_time - start_time))
echo "总运行时间: $(date -d@$total_time -u +%H:%M:%S)" | tee -a /home/vllm/CN/test_log/performance_log.txt
echo "测试完成，详细结果请查看 /home/vllm/CN/test_log/performance_log.txt" | tee -a /home/vllm/CN/test_log/performance_log.txt