#!/bin/bash -l
# DWPose front-end FPS/latency benchmark (gap G1 — the real-time bottleneck).
#SBATCH --job-name=dwpose-fps
#SBATCH --partition=ondemand-gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
export LD_LIBRARY_PATH=$(python -c "import os,glob,nvidia;print(':'.join(glob.glob(os.path.dirname(nvidia.__file__)+'/*/lib')))"):$LD_LIBRARY_PATH
DE=/scratch/pmc099/pprasad/dwpose
echo "### GPU ###"
python /group/pmc099/pprasad/code/results/dwpose_experiment/dwpose_fps.py --rgb_dir "$DE/rgb_clips" --device cuda --clips 8
echo "### CPU (deployment realism) ###"
python /group/pmc099/pprasad/code/results/dwpose_experiment/dwpose_fps.py --rgb_dir "$DE/rgb_clips" --device cpu --clips 4
echo "DWPOSE_FPS_DONE"
