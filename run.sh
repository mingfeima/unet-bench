#!/bin/sh

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING"
echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME\n"

# if intended to run on CPU, need to specify --no-cuda on machine with CUDA support
# the code runs on GPU by default if CUDA is available
python -u benchmark.py --batch-size 1

# collet nvprof summary log
nvprof=/usr/local/cuda/bin/nvprof
#$nvprof --profile-child-processes --csv --log-file nvprof_sum_%p.csv python -u benchmark.py --batch-size 1

# collet nvprof metric log
metrics="ipc,flop_sp_efficiency,achieved_occupancy,branch_efficiency,shared_load_transactions_per_request,gld_transactions_per_request,dram_read_throughput,dram_write_throughput,dram_utilization,shared_load_throughput,shared_store_throughput,shared_efficiency,shared_utilization,l2_read_throughput,l2_write_throughput,l2_utilization,l2_tex_read_hit_rate,l2_tex_write_hit_rate,tex_cache_throughput,tex_cache_hit_rate,tex_utilization"
#$nvprof --metrics $metrics --csv --log-file nvprof_metric_%p.csv python -u benchmark.py --batch-size 1
