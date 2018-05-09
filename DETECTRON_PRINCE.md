## Instructions to use FB's detectron on Prince HPC

Ask for a **GPU** instance using SBATCH / run an interactive one (will not work on cpu)

Load the pre-installed detectron image that NYU's HPC people kindly installed for us

```bash
module load singularity/2.5.1
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

## How to blackout specific objects in Images

1. Download the custom built version  [infer_tools](https://raw.githubusercontent.com/Iwontbecreative/Detectron/master/tools/infer_simple.py) . 
2. Put your image in a directory
3. On a **singularity shell with detectron image loaded and the PYTHONPATH setup**, run:
```bash
python2 /your/path/to/tools/infer_simple.py \
    --cfg /detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir /where/you/want/to/save/blackened/image \
    --image-ext jpg \
    --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    --pixel "(y_location, x_location)"
    /your/path/to/your/image
```
The "wts" will download the model file from AWS. Alternatively, you can use */scratch/tjf324/DL/model_final.pkl* for this specific model (you should have the rights). Also note, `x_location` and `y_location` are measured from the top left corner of the image.

If the pixel you mention has a mask in it, it will return the path to the blackened out image, the mask, the removed class and the score of that class. If no mask, it raises an Exception.

Example commands:
```bash
python2 tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir demo/custom_demo/output/ \
    --image-ext png \
    --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    --pixel "(163,50)" demo/custom_demo/football.jpg
```
```bash
python2 tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir demo/custom_demo/output/ \
    --image-ext png \
    --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    --pixel "(390,534)" demo/custom_demo/snow.jpg
```

```bash
python2 tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir demo/custom_demo/output/ \
    --image-ext png \
    --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    --pixel "(324,989)" demo/custom_demo/mihir.jpg
```
