for c in 50; do
  for s in 0.5; do
    for g in 10.0; do
      echo "Running with lambda_c=$c, lambda_s=$s, lambda_g=$g"
      CUDA_VISIBLE_DEVICES=$1 python3 test.py \
        --prompt "A painting" \
        --content_img ../vangogh2photo/trainB/ \
        --output_dir exps/exp_vangogh \
        --train_data_dir ./datasets/syn_vangogh/ \
        --label_resize_train_data_dir ./datasets/syn_vangogh/ \
        --model_type MLP \
        --num_test_samples 1 \
        --lambda_c $c \
        --lambda_s $s \
        --lambda_g $g \
        --mlp_weight 1.0
    done
  done
done
