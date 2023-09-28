train_path="../data/sentiment/training/wiki_sen.src"

dev_path="xxxx"
test_path="xxx"

pretrain_path="xxx"
pretrain_path_val="xxx"
long_test_path="xxx"

model_name_or_path="/home2/zhanghanqing/pretrained_model/gpt2/large"
check_point_load="../check_point_load/pretrained_check.ckpt"

ranking_scope=90
disc_embedding_checkpoint="../data/sentiment/disc_check/disc_sentiment.ckpt"
template_disc="(2,3)"


pretrain_plm="gpt"
train_stage="fine_tuning"
tuning_mode='fp'
model_type="Prompt_Residual_Tuning"
dataset="CommonGen"

top_p=0.98
training_sample_num=11
step_size=7000000
temperature=0.005
lr=2e-4
memory_p=0.5
residual_layer=3

out_dir="../check_point/sen"
batch_size=64
max_epoch=3


for seed in 1
    do
        CUDA_VISIBLE_DEVICES=3  python ../train_att.py   --train  --saving_model --model_name_or_path $model_name_or_path --train_path $train_path  --dev_path $dev_path --test_path $test_path --batch_size $batch_size  --max_epoch $max_epoch --out_dir $out_dir --seed $seed  --pretrain_plm $pretrain_plm --top_p $top_p  --tuning_mode $tuning_mode --training_sample_num $training_sample_num --temperature $temperature --lr $lr  --train_stage $train_stage --pretrain_path $pretrain_path --step_size $step_size --pretrain_path_val $pretrain_path_val  --long_test_path $long_test_path --model_type $model_type --memory_p $memory_p  --residual_layer $residual_layer  --dataset $dataset --ranking_scope $ranking_scope --disc_embedding_checkpoint $disc_embedding_checkpoint --template_disc $template_disc --check_point_load $check_point_load
            
    done


