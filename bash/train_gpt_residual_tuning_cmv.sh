train_path="../../data/cmv"
dev_path="../../data/cmv"
test_path="../../data/cmv"

model_name_or_path="/home2/zhanghanqing/pretrained_model/gpt2/large"
check_point_load="../check_point/5/prompt_model/template_(8,4)_epoch_5000320_metric_52.48.ckpt"
pretrain_plm="gpt"
train_stage="fine_tuning"
tuning_mode='fp'
model_type="Residual_Tuning"
dataset="cmv"

top_p=0.85
training_sample_num=32000
temperature=0.1
num_layer=4
lr=1e-5
max_length=100

out_dir="../check_point/control"
template="(2,2)"
batch_size=32
max_epoch=6


for seed in 2
    do
        CUDA_VISIBLE_DEVICES=1  python ../train.py  --train --model_name_or_path $model_name_or_path  --train_path $train_path  --dev_path $dev_path --test_path $test_path --batch_size $batch_size --template $template  --max_epoch $max_epoch --out_dir $out_dir --seed $seed  --pretrain_plm $pretrain_plm --top_p $top_p  --tuning_mode $tuning_mode --training_sample_num $training_sample_num --temperature $temperature --lr $lr --num_layer $num_layer --train_stage $train_stage  --model_type $model_type --dataset $dataset  --max_length $max_length --check_point_load $check_point_load 
            
    done



# echo  $seed

