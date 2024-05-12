export CUDA_VISIBLE_DEVICES=$1

seed=$2
task=$3
task=${task}NoFrameskip-v4

noise_dim=4
noise_per_sample=20
target_noise_per_sample=1

action_sample_num=1
action_select_scheme=Greedy

epoch=100
step_per_epoch=20000

for i in $(seq 5)
do
    tag=$(date "+%Y%m%d%H%M%S")
    python -m hyperagent.scripts.hyperagent.run_atari --seed=${seed} --task=${task} \
    --noise-dim=${noise_dim} \
    --noise-per-sample=${noise_per_sample} \
    --target-noise-per-sample=${target_noise_per_sample} \
    --action-sample-num=${action_sample_num} \
    --action-select-scheme=${action_select_scheme} \
    --epoch=${epoch} \
    --step-per-epoch=${step_per_epoch} \
    > ~/logs/${task}_${tag}.out 2> ~/logs/${task}_${tag}.err &
    echo "run $seed $tag"
    let seed=$seed+1
    sleep 2
done

