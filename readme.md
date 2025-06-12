# MetaShape: Concept-Driven Shape Morphing
## env setup
- first create an env
```bash
conda create --name morphing python=3.8.18 -y
conda activate morphing
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -y numpy scikit-image
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite svgpathtools cssutils numba torch-tools scikit-fmm easydict visdom freetype-py shapely
pip install opencv-python==4.5.5.64
pip install kornia==0.7.1
pip install wandb
pip install diffusers
pip install transformers scipy ftfy accelerate
```
- second clone diffvg
```bash
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
python setup.py install
```

## experiment
```bash
git clone https://github.com/Prince-Polo/DLproject
cd DLproject
## defalt
#bash Run_for_png_pig.sh 
```

