import os
from field_boundaries.ricult_boundary_engine import main

if __name__ == '__main__':
    exp_dir = '/data/dev/models/denseunet'
    output_dir='/data/experiments/result'
    arch='densenet_unet.DensenetUnet(seg_classes=3)'
    grid='/data/dev/grids/06-21_09:17.jpg'
    center='31.060685,74.087856'
    main(exp_dir=exp_dir, output_dir=output_dir, arch=arch, grid=grid, center=center)
    os.system('sudo shutdown')
