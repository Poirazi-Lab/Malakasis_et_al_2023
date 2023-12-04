#Maximum walltime for this job
#$ -l h_rt=9999:00:00
#Maximum cpu time for this job
#$ -l h_cpu=9999:00:00

# Specify the shell to use when running the job script
#$ -S /bin/sh

# Directory to perform the job
#$ -cwd

# Name of the Job
#$ -N lamodelSNN

#$ -o log_files
#$ -e log_files

./lamodel-nikos $LAPARAMS
