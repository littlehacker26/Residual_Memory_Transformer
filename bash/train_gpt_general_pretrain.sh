train_path="../../data/commongen.train.jsonl"
dev_path="../../data/commongen.dev.jsonl"
test_path="../../data/commongen.test.jsonl"
pretrain_path="/home2/xxx/formatted_wikipedia"
pretrain_path_val="/home2/xxx/formatted_wiki_val"

model_name_or_path="/home2/xxx/pretrained_model/gpt2/large"

pretrain_plm="gpt"
tuning_mode='pt'
train_stage="general_pretrain"
model_type="Residual_Tuning"

top_p=0.5
training_sample_num=1000000
step_size=2000000
step_log=200000
temperature=0.1
lr=5e-5

out_dir="../check_point/add"
batch_size=48
max_epoch=10
seed=42
residual_layer=4


for residual_layer in 3
do
    echo  $encoder_layer
    CUDA_VISIBLE_DEVICES=0  python ../train.py  --train --model_name_or_path $model_name_or_path --train_path $train_path  --dev_path $dev_path --test_path $test_path --batch_size $batch_size  --max_epoch $max_epoch --out_dir $out_dir --seed $seed  --pretrain_plm $pretrain_plm --top_p $top_p  --tuning_mode $tuning_mode --training_sample_num $training_sample_num --temperature $temperature --lr $lr  --train_stage $train_stage --pretrain_path $pretrain_path --step_size $step_size --pretrain_path_val $pretrain_path_val --model_type $model_type --step_log $step_log --residual_layer $residual_layer
            
done
