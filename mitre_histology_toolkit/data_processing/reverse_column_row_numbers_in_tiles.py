import pandas as pd
import os

slides = [i.replace('.vsi','') for i in os.listdir('../AMP_Slides_v2') if '.vsi' in i]

slide_args = pd.read_csv('slide_arguments_tile.txt', delimiter = ' ')
all_slides_spec = list(slide_args.filename)

slides_spec = []
for slide in slides:
    for slide_spec in all_slides_spec:
        if (slide in slide_spec) and (slide != '300-410'):
            slides_spec += [slide_spec]

slides_spec = list(set(slides_spec))

a = open('vsi_slides.txt', 'w')
a.writelines([i + ' 0\n' for i in slides_spec])
a.close()

os.mkdir('tiles/temp')
for ss in slides_spec:
    tile_dir = f'tiles/{ss}'
    tiles = os.listdir(tile_dir)
    new_tiles = []
    for tile in tiles:
        row_num = tile.split('__R')[1].split('_')[0]
        col_num = tile.split('__R')[1].split('_C')[1].split('_')[0]
        new_tiles += [tile.replace(f'_R{row_num}', f'_R{col_num}').replace(f'_C{col_num}', f'_C{row_num}')]
    
    new_tile_dir = f'tiles/temp/{ss}'
    os.mkdir(new_tile_dir)
    for i in range(len(tiles)):
        os.rename(f'{tile_dir}/{tiles[i]}', f'{new_tile_dir}/{new_tiles[i]}')
    