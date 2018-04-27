## Instructions to use FB's detectron on Prince HPC

Ask for a **GPU** instance using SBATCH / run an interactive one (will not work on cpu)

Load the pre-installed detectron image that NYU's HPC people kindly installed for us

```bash
module load singularity/2.4.4
singularity shell --nv /beegfs/work/public/singularity/detectron.img
# or something other than shell, see singularity doc
```

Add detectron to your python path 
```bash
export PYTHONPATH="${PYTHONPATH}:/detectron/lib/" 
```

Run a test
```
python2 /detectron/tests/test_spatial_narrow_as_op.py 
```
