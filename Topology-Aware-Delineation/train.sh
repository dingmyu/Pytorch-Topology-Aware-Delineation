#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 GLOG_logtostderr=1 \
srun --mpi=pmi2 -p bj11part -n1  --job-name=seg --gres=gpu:4 --ntasks-per-node=1 python -u main.py  2>&1 | tee log.txt &

