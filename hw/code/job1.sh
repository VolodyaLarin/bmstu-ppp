#!/bin/bash
# Название расчитываемой задачи. Может быть любым.
#SBATCH --job-name="homework"
#
# Множество вычислительных узлов для расчета задачи. Определяет характеристику
# вычислительных узлов.
#SBATCH --partition=debug
#
# Запускать кажды расчет на одном узле.
#SBATCH --nodes=10
#
# Расчетное время, после истечения которого задача будет принудительно
# остановлена. В данном случае --- 7 дней.
#SBATCH --time=0-00:10:00
#
# Количество потоков одного процессора (20 для intelv3-batch, 24 для
# intelv4-batch, 256 для knl-batch).
#SBATCH --ntasks-per-node=1
#
#SBATCH --mem=1992 

# Чтобы srun заработал с impi
###  export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
echo "HOSTNAME = $HOSTNAME"


mpirun -np $1 a.out 10 100 200 400 600 800 1000 1200 1400 1600 1800 2000 

