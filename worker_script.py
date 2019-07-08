import os
from tqdm import tqdm
from field_boundaries.ricult_boundary_engine import main

if __name__ == '__main__':
    model_path = '/data/dev/models/denseunet'
    output_dir = '/data/experiments/result'
    json_path = '/data/dev/models/denseunet'
    arch = 'densenet_unet.DensenetUnet(seg_classes=3)'

    with open(json_path, 'w') as json_file:
        points_list = json_file['points']
    for i in tqdm(range(len(points_list))):
        points = points_list[i]
        main(model_path=model_path, output_dir=output_dir, arch=arch, grid_name=i, points=points)
    os.system('sudo shutdown')
