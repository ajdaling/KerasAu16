#PBS -N run_viz
#PBS -l walltime=0:30:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l mem=32GB
#PBS -j oe

# uncomment if using qsub
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

source setup.csh
module load cuda
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python visualize_tt.py >& testViz.log

