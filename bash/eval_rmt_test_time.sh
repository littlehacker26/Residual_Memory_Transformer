test_path="../data/sentiment/test/positive_prompts.jsonl"

pretrain_path="xxx"
pretrain_path_val="xxx"
long_test_path="../data/c2gen.json"

model_name_or_path="/home2/xxx/pretrained_model/gpt2/large"
check_point_load="../check_point_load/attribute/top_095_k110.ckpt"

ranking_scope=30

pretrain_plm="gpt"
train_stage="fine_tuning"
tuning_mode='fp'
model_type="Prompt_Residual_Tuning"
dataset="CommonGen"

top_p=0.95
training_sample_num=11
step_size=7000000
temperature=0.01
lr=1e-4
memory_p=0.5
residual_layer=3

out_dir="../check_point/trained"
batch_size=1
max_epoch=6
number_beam=1


for seed in 41
    do
        CUDA_VISIBLE_DEVICES=0  python ../train_att.py   --test   --model_name_or_path $model_name_or_path  --test_path $test_path --batch_size $batch_size  --max_epoch $max_epoch --out_dir $out_dir --seed $seed  --pretrain_plm $pretrain_plm --top_p $top_p  --tuning_mode $tuning_mode --training_sample_num $training_sample_num --temperature $temperature --lr $lr  --train_stage $train_stage --pretrain_path $pretrain_path --step_size $step_size --pretrain_path_val $pretrain_path_val  --long_test_path $long_test_path --model_type $model_type --memory_p $memory_p  --residual_layer $residual_layer  --dataset $dataset --ranking_scope $ranking_scope --number_beam $number_beam --check_point_load $check_point_load
            
    done


#negative_prompts.jsonl  neutral_prompts.jsonl  positive_prompts.jsonl


