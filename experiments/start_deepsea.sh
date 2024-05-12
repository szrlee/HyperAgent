cuda_id=0
size=$1

alg_type=HyperAgent
for seed in 2020 2025
do
    sh experiments/deepsea/run_${alg_type}.sh $cuda_id 2020 $size
    sleep 0.5
    echo "run $cuda_id $task"

    let cuda_id=$cuda_id+1
done

