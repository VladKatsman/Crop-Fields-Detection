import os
from google_maps_python.data_utils import create_grid

if __name__ == '__main__':
    center ='31.060685,74.087856'
    output_path='/data/experiments/grid'
    arch='densenet_unet.DensenetUnet(seg_classes=3)'
    grid='/data/dev/grids/06-21_09:17.jpg'
    center='31.060685,74.087856'
    create_grid(exp_dir=exp_dir, output_dir=output_dir, arch=arch, grid=grid, center=center)
    os.system('sudo shutdown')

    --center
    31.060685,74.087856
    --output_path
    /media/noteme/3981506f-63a3-41b7-9142-6439704159e2/data/ricult/grids
    --grid_side_len
    20