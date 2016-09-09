#!/bin/sh
echo FM policy for the Movielens-100k dataset
echo Model uses content-based item representation
cd ..
rm -Rf ddpg-results/experiment1/
nohup python src/run.py --outdir ddpg-results/experiement1 --env CollaborativeFiltering-v3 --train 100 --test 100 --tmax 2314 --total 2163000 --fmpolicy --create_index --wp_total_actions 1682 --wp_action_set_file "data/ml-100k/embeddings-movielens100k.csv" --knn 84 --skip_space_norm >> run.log &
echo $! > pid

sleep 1

echo Process started. Logs are written in run.log