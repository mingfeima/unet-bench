# unet-bench
This repo contains performance benchmark on UNet and UNet3D, simply use `./run.sh` to launch the script. 
On a CUDA available machine,`benchmark.py` will run on GPU by default, need to specify `--no-cuda`.
The script runs on CPU in case CUDA is not available, with the following setting
```bash
# set OMP_NUM_THREADS to be the number of physical cores
export KMP_AFFINITY="granularity=fine,compact,1,0"
export OMP_NUM_THREADS=56
export KMP_BLOCKTIM=1
```
