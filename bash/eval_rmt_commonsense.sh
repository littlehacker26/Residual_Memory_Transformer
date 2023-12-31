test_path="../data/word_include/commongen.test.jsonl"
long_test_path="../data/word_include/c2gen.json"

model_name_or_path="/home2/xxx/pretrained_model/gpt2/large"
check_point_load="../check_point_load/commongen/common_gen_without_length_control.ckpt"


pretrain_plm="gpt"
train_stage="fine_tuning"
tuning_mode='fp'
model_type="Prompt_Residual_Tuning"
dataset="CommonGen"

top_p=0.5
step_size=7000000
temperature=0.1
lr=1e-4
memory_p=0.5
residual_layer=3

out_dir="../check_point/trained"
batch_size=49
max_epoch=6
generated_len=17
max_length=32
seed=42


for generated_len in 15
    do
        CUDA_VISIBLE_DEVICES=0  python ../train.py   --test --model_name_or_path $model_name_or_path   --test_path $test_path --batch_size $batch_size  --max_epoch $max_epoch --out_dir $out_dir --seed $seed  --pretrain_plm $pretrain_plm --top_p $top_p  --tuning_mode $tuning_mode --temperature $temperature --lr $lr  --train_stage $train_stage --step_size $step_size  --long_test_path $long_test_path --model_type $model_type --memory_p $memory_p  --residual_layer $residual_layer  --dataset $dataset --max_length  $max_length --generated_len $generated_len --check_point_load $check_point_load
            
    done

