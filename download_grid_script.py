from google_maps_python.data_utils import create_grid

if __name__ == '__main__':
    center ='31.060685,74.087856'
    output_path='/data/experiments/grids'
    grid_side_len = 20
    create_grid(center=center, output_path=output_path, grid_side_len=grid_side_len)
