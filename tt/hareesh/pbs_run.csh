#PBS -N chan5
#PBS -l walltime=2:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=32GB
#PBS -j oe

# uncomment if using qsub
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

module load python/2.7.8
python test2.py  >& chan5_35k.log
