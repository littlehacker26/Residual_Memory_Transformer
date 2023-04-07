copy_vocab_path="../dataset/new_copy_vocab.txt"
train_path="../../data/commongen.train.jsonl"
dev_path="../../data/commongen.dev.jsonl"
test_path="../../data/commongen.test.jsonl"
pretrain_path="/home2/zhanghanqing/formatted_wikipedia"
pretrain_path_val="/home2/zhanghanqing/formatted_wiki_val"
long_test_path="../../data/c2gen.json"

model_name_or_path="/home2/zhanghanqing/pretrained_model/gpt2/large"
check_point_load="../check_point/pretrain/prompt_model/layer_36_epoch_2000064_metric_2.23.ckpt"
#check_point_load="../check_point/general/prompt_model/template_(8,4)_epoch_7500096_metric_1.86.ckpt"

pretrain_plm="gpt"
train_stage="fine_tuning"
tuning_mode='fp'
model_type="Prompt_Residual_Tuning"


top_p=0.5
training_sample_num=4
step_size=7000000
temperature=0.1
num_layer=36
lr=1e-5
memory_p=0.5

out_dir="../check_point/general"
batch_size=60
max_epoch=5


for seed in 2
    do
        CUDA_VISIBLE_DEVICES=2  python ../train.py  --train --model_name_or_path $model_name_or_path --copy_vocab_path $copy_vocab_path --train_path $train_path  --dev_path $dev_path --test_path $test_path --batch_size $batch_size  --max_epoch $max_epoch --out_dir $out_dir --seed $seed  --pretrain_plm $pretrain_plm --top_p $top_p  --tuning_mode $tuning_mode --training_sample_num $training_sample_num --temperature $temperature --lr $lr --num_layer $num_layer --train_stage $train_stage --pretrain_path $pretrain_path --step_size $step_size --pretrain_path_val $pretrain_path_val  --long_test_path $long_test_path --model_type $model_type --memory_p $memory_p  --check_point_load $check_point_load
            
    done



# echo  $seed

