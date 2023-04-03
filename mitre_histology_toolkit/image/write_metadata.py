from . import image_loader
import yaml

def write_metadata(image_path, metadata_path, valid_magnification = 20):
    """
    Writes the image metadata to file for future reference. To accomodate
    slides with multiple scenes, each scene will be a nested key in the dict.

    Parameters
    ----------
    image_path : str
        The path to the image.
    metadata_path : str
        The path to the output metadata yaml file.
    valid_magnification : int, optional
        The value of image magnification that is required for a scene to be
        valid. The default is 20.

    Returns
    -------
    None.

    """
    slide = image_loader.open_image(image_path, valid_magnification)
    metadata_dict = {}
    for scene_id in slide.valid_scenes:
        slide.set_scene(scene_id)
        scene_info = slide.get_info()
        metadata_dict[scene_id] = scene_info
    
    with open(metadata_path, 'w') as outfile:
        yaml.dump(metadata_dict, outfile, default_flow_style=False)
    return
