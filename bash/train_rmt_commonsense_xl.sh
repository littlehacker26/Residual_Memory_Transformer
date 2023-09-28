train_path="../data/word_include/wiki_train_common.json"

dev_path="../data/word_include/commongen.dev.jsonl"
test_path="../data/word_include/commongen.test.jsonl"

pretrain_path="xxxx"
pretrain_path_val="xxxx"
long_test_path="../data/word_include/c2gen.json"

model_name_or_path="/home2/zhanghanqing/pretrained_model/gpt2/median"
# check_point_load="../check_point/pretrain/prompt_model/4_layer_epoch_4000128_metric_1.68.ckpt"
check_point_load="../check_point/layer_size_median/prompt_model/layer_1_epoch_4000032_metric_2.25.ckpt"


pretrain_plm="gpt"
train_stage="fine_tuning"
tuning_mode='fp'
model_type="Prompt_Residual_Tuning"
dataset="CommonGen"

top_p=0.6
training_sample_num=11
step_size=7000000
temperature=0.1
lr=5e-5
memory_p=0.5
residual_layer=1

out_dir="../check_point/xl"
batch_size=48
max_epoch=6


for seed in 40
    do
        CUDA_VISIBLE_DEVICES=0  python ../train.py   --saving_model --train  --saving_model --model_name_or_path $model_name_or_path --train_path $train_path  --dev_path $dev_path --test_path $test_path --batch_size $batch_size  --max_epoch $max_epoch --out_dir $out_dir --seed $seed  --pretrain_plm $pretrain_plm --top_p $top_p  --tuning_mode $tuning_mode --training_sample_num $training_sample_num --temperature $temperature --lr $lr  --train_stage $train_stage --pretrain_path $pretrain_path --step_size $step_size --pretrain_path_val $pretrain_path_val  --long_test_path $long_test_path --model_type $model_type --memory_p $memory_p  --residual_layer $residual_layer  --dataset $dataset --check_point_load $check_point_load
            
    done

# echo  $seed
# --saving_model
