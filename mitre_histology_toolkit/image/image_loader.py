from bs4 import BeautifulSoup
import bioformats as bf
from PIL import Image
import numpy as np
import javabridge
import openslide
import slideio
import os

def open_image(fpath, valid_magnification):
    """
    Opens an image using the appropriate backend.

    Parameters
    ----------
    fpath : str
        The path to the image.
    valid_magnification : int
        The level of magnification expected for valid scenes.

    Returns
    -------
    The image object and a list of the valid scenes associated with it.

    """
    image_type = os.path.splitext(fpath)[1]
    if image_type in ['.vsi']:
        #javabridge.start_vm(class_path=bf.JARS, max_heap_size='8G', run_headless=True)
        #slide = ImageLoaderBioformat(fpath, manage_javabridge = False)
        slide = ImageLoaderBioformat(fpath)
    elif image_type in ['.czi', '.scn', '.svs']:
        slide = ImageLoaderSlideio(fpath, valid_magnification = valid_magnification)
    
    return(slide)

class ImageLoaderSlideio(object):
    """Image loader class using OpenSlide."""
    def __init__(self, filename, valid_magnification = 20):
        """Init Image loader class using OpenSlide.

        Arguments
        ----------
        filename : string
            The path to the whole slide image file.
        valid_magnification : int
            The magnification that corresponds to valid scenes. The default is 20.
        """
        self.filetype = filename[-3:].upper()
        self.slide = slideio.open_slide(filename, self.filetype)
        self.valid_scenes = []
        for scene_id in range(self.slide.num_scenes):
            if int(self.slide.get_scene(scene_id).magnification) == int(valid_magnification):
                self.valid_scenes += [scene_id]
        
        self.set_scene(self.valid_scenes[0])
    def get_thumbnail(self, dimension = None, max_dimension = 256, scene_id = None):
        """Get thumbnail that fit within specified dimension.

        Arguments
        ----------
        dimension : tuple(2)
            Specify the dimension (cols, rows) within which the thumbnail will 
            fit. The default is None which invokes the max_dimension and maintains
            the original aspect ratio.
        max_dimension : int
            The max dimension of the thumbnail is used when dimension is None. 
            The aspect ratio is maintained and the largest dimension will be set
            to max_dimension.

        Returns
        --------
        numpy array of the thumbnail
        """
        if scene_id is None:
            scene_id = self.idx
        
        cols, rows = self.slide.get_scene(scene_id).size
        if dimension is None:
            if rows > cols:
                final_dimension = (round(cols / rows * max_dimension), max_dimension)
            else:
                final_dimension = (max_dimension, round(rows / cols * max_dimension))
        else:
            final_dimension = dimension
        
        if self.filetype == 'CZI':
            img_tn = self.slide.get_scene(scene_id).read_block((0, 0, cols, rows), size = final_dimension)
            img_tn = img_tn[:,:,::-1]
        else:
            img_tn = self.slide.get_scene(scene_id).read_block((0, 0, cols, rows), size = final_dimension)
        return(img_tn)
    
    def set_scene(self, scene_id):
        """
        Sets the scene for the image object.

        Parameters
        ----------
        scene_id : int
            The scene within the slide.

        Returns
        -------
        None.

        """
        self.idx = scene_id
        return
    
    def get_low_res_image(self, magnification, scene_id = None):
        """
        Returns the full image at the specified magnification

        Parameters
        ----------
        magnification : int
            The magnification for the output image.
        scene_id : int
            The scene for which to return the image.
        
        Returns
        -------
        numpy array of the low res image.

        """
        if scene_id is None:
            scene_id = self.idx
        
        scene_info = self.get_info(scene_id = scene_id)
        
        return(self.get_region(0, scene_info['sizeX'], 0, scene_info['sizeY'], 
                               magnification, scene_id = scene_id, 
                               filter_black = True))
        
    def get_info(self, scene_id = None):
        """Get metadata about the image.

        Returns
        --------
        Dictionary
            Contains:
                - magnification: magnification of raw image
                - resolution: the microns per pixel in x and y (width and height)
                - mm_x: physical pixel size (in mm) in x axis
                - mm_y: physical pixel size (in mm) in y axis
                - sizeX: x axis resolution
                - sizeY: y axis resolution

        """
        if scene_id is None:
            scene_id = self.idx
        
        temp_scene = self.slide.get_scene(scene_id)
        return({
            "magnification": int(temp_scene.magnification),
            "resolution": (temp_scene.resolution[0] * 1e6, temp_scene.resolution[0] * 1e6),
            "mm_x": temp_scene.resolution[0] * temp_scene.size[0] * 1000,
            "mm_y": temp_scene.resolution[1] * temp_scene.size[1] * 1000,
            "sizeX": int(temp_scene.size[0]),
            "sizeY": int(temp_scene.size[1])
            })

    def get_scene(self, scene_id):
        """Get a scene object from the slide
        
        Arguments
        ---------
        scene_id : int
            the scene id to query
        
        Returns
        -------
        Slideio scene object
        """
        return(self.slide.get_scene(scene_id))
        
    def get_region(self, xmin, xmax, ymin, ymax, mag, scene_id = None, filter_black = True):
        """Get image within bounding box, scaled to magnification

        Arguments
        ----------
        xmin : int
        xmax : int
        ymin : int
        ymax : int
        mag : int
        scene_id : int
            The scene from which to extract the region.
        filter_black : boolean
            If true, sets all (0,0,0) to (255,255,255). This is useful because
            the background is generally gray or white and pitch black pixels
            are likely to be background pixels.

        Returns
        --------
        numpy array of the image
        """
        if scene_id is None:
            scene_id = self.idx
        
        temp_scene = self.slide.get_scene(scene_id)
        orig_mag = temp_scene.magnification
        xdim = xmax - xmin
        ydim = ymax - ymin
        if mag != orig_mag:
            new_size = (int(xdim * mag / orig_mag), int(ydim * mag / orig_mag))
            region = temp_scene.read_block((xmin, ymin, xdim, ydim), size = new_size)
        else:
            region = temp_scene.read_block((xmin, ymin, xdim, ydim))
        
        if filter_black:
            region[np.where(region.max(axis = 2) == 0)] = (255, 255, 255)
        
        if self.filetype == 'CZI':
            region = region[:,:,::-1]
        
        return(region)

class ImageLoaderBioformat():
    """Image loader class using Bioformats.
       Bioformats is a Java library, linked to python via javabridge.
       Thus, environmental variabe `JAVA_HOME` needs to be defined before 
       initiating this class. By default, `JAVA_HOME` points to 
       "/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home".

       TODO: deal with large image with openBytesXYWH: https://forum.image.sc/t/bio-formats-read-region-equivalent/39150/10
    """
    idx = None
    def __init__(self, filepath, valid_magnification = 20, manage_javabridge=True):
        """Init Image loader class using Bioformats.

        Arguments
        ----------
        filename : string
            path to the whole slide image file
        """
        self.filepath = filepath
        self.filetype = filepath[-3:].upper()
        if manage_javabridge:
            # start java VM
            javabridge.start_vm(class_path=bf.JARS, max_heap_size='8G', run_headless=True)
            logger = self._init_logger()
        
        self.metadata = bf.get_omexml_metadata(self.filepath)
        self.md_rdr = bf.OMEXML(self.metadata)
        self.rdr = bf.ImageReader(self.filepath)
        soup = BeautifulSoup(self.metadata, 'lxml')
        objectives = {}
        for o in soup.select("instrument > objective"):
            objectives[o['id']] = float(o['nominalmagnification'])
        images = []
        resolution_tags = ['size' + c for c in ['x','y','z','c','t']]
        pixel_size_tags = ['physicalsize' + c for c in ['x', 'y']]
        physical_position_tags = ['position' + c for c in ['x','y']]
        for idx, image in enumerate(soup.select("image")):
            try:
                objective = image.find("objectivesettings")
                magnification = objectives[objective['id']]
                pixels = image.find('pixels')
                plane = image.find('plane')
                pixel_size = tuple([float(pixels[t]) for t in pixel_size_tags])
                resolution = tuple([int(pixels[t]) for t in resolution_tags])
                physical_position = tuple([float(plane[t]) for t in physical_position_tags])
                images.append({
                    "pixel_size": pixel_size,
                    "resolution": (resolution[0], resolution[1]),
                    "zct": (resolution[2], resolution[3], resolution[4]),
                    "magnification": magnification,
                    "name": image['name'],
                    "physical_position": physical_position,
                    "idx": idx
                })
            except Exception as e:
                # print(e)
                pass

        self.images = sorted(
                sorted([list(filter(lambda x: x['physical_position'] == pp, images) )
                    for pp in 
                        set(map(lambda x:  x['physical_position'], images))
                ], key=lambda x: x[0]['resolution'][0], reverse=True)
            , key=lambda x: x[0]['idx'])

        for i in self.images:
            raw_resolution = i[0]['resolution']
            raw_magnification = i[0]['magnification']

            for ii in i:
                ii['magnification'] = raw_magnification * ii['resolution'][0] / i[0]['resolution'][0]
        
        self.get_valid_index(valid_magnification)
        self.set_scene(self.valid_scenes[0])
        self.slide = self
    
    def _init_logger(self):
        """This is so that Javabridge doesn't spill out a lot of DEBUG messages
        during runtime.
        From CellProfiler/python-bioformats.
        """
        rootLoggerName = javabridge.get_static_field("org/slf4j/Logger",
                                                     "ROOT_LOGGER_NAME",
                                                     "Ljava/lang/String;")

        rootLogger = javabridge.static_call("org/slf4j/LoggerFactory",
                                            "getLogger",
                                            "(Ljava/lang/String;)Lorg/slf4j/Logger;",
                                            rootLoggerName)

        logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level",
                                               "WARN",
                                               "Lch/qos/logback/classic/Level;")

        javabridge.call(rootLogger,
                        "setLevel",
                        "(Lch/qos/logback/classic/Level;)V",
                        logLevel)
    
    def validate_index(self):
        """Validate the scene index

        Throw ValueError if self.idx is not defined or not valid, 
        otherwise return nothing.
        """
        if self.idx == None:
            raise ValueError("Image idx not defined")
        elif self.idx not in set(map(lambda x: x[0]['idx'], self.images)):
            raise ValueError(f"Image not found with idx {self.idx}")
    
    def get_all_scene_indices(self):
        """Get all the valid scene indices.

        Returns
        -------
        set of integer
            Set of valid scene indices
        """
        
        return set(map(lambda x: x[0]['idx'], self.images))
    
    def get_valid_index(self, valid_magnification):
        """Get all the valid scene indices.
        
        Arguments
        ---------
        valid_magnification : int
            The expected magnification for valid scenes.
        
        Returns
        -------
        list of integers
            Set of valid scene indices
        """
        self.valid_scenes = []
        for scene_id in self.get_all_scene_indices():
            self.idx = scene_id
            if int(self.get_info()['magnification']) == int(valid_magnification):
                self.valid_scenes += [scene_id]
        
        self.valid_scenes = sorted(self.valid_scenes)
        return
    
    def set_scene(self, scene_id):
        """
        Sets the scene for the image object.

        Parameters
        ----------
        scene_id : int
            The scene within the slide.

        Returns
        -------
        None.

        """
        self.idx = scene_id
        return
    
    def get_image(self, max_resolution=None, magnification=None, region_xywh=None, rescale=True, verbose = False):
        """Return image based on specified size and position parameters

        Arguments
        ----------
        max_resolution : tuple(2)
            Whole slide image resolution for the output region. If max_resolution has 
            a different aspect ratio compared to the original WSI, the whole slide image 
            will be scaled down to the smaller dimension.
        magnification: int
            if max_resolution is not specified, scale image based on this. If both
            max_resolution and magnification is specified, max_resolution takes precedence.
        region_xywh: tuple(4)
            Specify the region of image based on the mininum x and y position, 
            width and height, based on the original raw resolution
        rescale : bool
            If set to False, return the smallest image from the stack larger than
            the specified max_resolution or magnification

        Returns
        -------
        numpy array of the image
        """
        self.validate_index()
        self.rdr.rdr.setSeries(self.idx)
        stack = [i for i in self.images if i[0]['idx'] == self.idx][0]
        
        if (max_resolution != None and all(i > j for i,j in zip(max_resolution, stack[0]['resolution']))) or (magnification != None and magnification > stack[0]['magnification']):
            raise ValueError("Specified max resolution or magnification is greater than that of original WSI")

        # max_resolution of target whole slide image
        if max_resolution == None and magnification != None:
            max_resolution = tuple(map(lambda x: x * magnification / stack[0]['magnification'], stack[0]['resolution']))
        elif max_resolution == None and magnification == None: 
            max_resolution = stack[0]['resolution']

        if not region_xywh:
            region_xywh = (0,0) + stack[0]['resolution']

        target_region_resolution = tuple(map(lambda x: int(round(x * min(max_resolution[0] / stack[0]['resolution'][0], max_resolution[1] / stack[0]['resolution'][1]))), region_xywh[2:]))
        if verbose:
            print("max resolution (for select pyramid level):", max_resolution)
            print("resolution of output region:", target_region_resolution)
        image_meta = sorted(
            [i for i in stack if i['resolution'][0] >= max_resolution[0] and i['resolution'][1] >= max_resolution[1]],
            key=lambda x: x['resolution'][0])[0]

        # if we found a lower resolution pyramid level image that we can get region for, rescale xywh down to the pyramid level
        region_xywh = tuple(map(lambda x: int(round(x * image_meta['resolution'][0] / stack[0]['resolution'][0])), region_xywh))


        if region_xywh[2] * region_xywh[3] * 3 > 2**31 - 1:
            height_segments = list(range(0, region_xywh[3], int((2**31 - 1) / region_xywh[2] / 3))) + [region_xywh[3]]
            parts = [self.rdr.rdr.openBytesXYWH(0,0,h,region_xywh[2],height_segments[i + 1] - h) for i,h in enumerate(height_segments[:-1])]
            region = np.concatenate(parts)
            region.shape = (region_xywh[3], region_xywh[2], 3)
        else:
            region = self.rdr.read(series=image_meta['idx'],rescale=False,XYWH=region_xywh)
        if verbose:
            print(region.shape)
        if (not rescale):
            return region
        else:
            region_pil = Image.fromarray(region)
            # scale to max_resolution of target image region
            region_pil.thumbnail( tuple(map(lambda x: int(round(x * min(max_resolution[0] / image_meta['resolution'][0], max_resolution[1] / image_meta['resolution'][1]))), region_xywh[2:])) )
            if verbose:
                print(region_pil.size)
            return np.array(region_pil)
    
    def get_thumbnail(self, dimension = None, max_dimension = 256):
        """Get thumbnail that fit within specified dimension.

        Arguments
        ----------
        dimension : tuple(2)
            Specify the dimension within which the thumbnail will fit, default to (256,256)

        Returns
        --------
        numpy array of the thumbnail
        """
        self.validate_index()
        if dimension is None:
            dimension = (max_dimension, max_dimension)
        return self.get_image(max_resolution = dimension)
    
    def get_low_res_image(self, magnification):
        """
        Returns the full image at the specified magnification

        Parameters
        ----------
        magnification : int
            The magnification for the output image.
        scene_id : int
            The scene for which to return the image.
        
        Returns
        -------
        numpy array of the low res image.

        """
        scene_info = self.get_info()
        return(self.get_region(0, scene_info['sizeX'], 0, scene_info['sizeY'], 
                               magnification, filter_black = True))
    
    def get_region(self, xmin, xmax, ymin, ymax, mag, filter_black = True, verbose = False):
        """Get image within bounding box, scaled to magnification

        Arguments
        ----------
        xmin : int
        xmax : int
        ymin : int
        ymax : int
        mag : int
        filter_black : boolean
            If true, sets all (0,0,0) to (255,255,255). This is useful because
            the background is generally gray or white and pitch black pixels
            are likely to be background pixels.
        
        Returns
        --------
        numpy array of the image
        """
        self.validate_index()
        
        region = self.get_image(magnification=mag, region_xywh=(xmin, ymin, (xmax - xmin), (ymax - ymin)), verbose = verbose)
        if filter_black:
            region[np.where(region.max(axis = 2) == 0)] = (255, 255, 255)
        
        return(region)

    def get_info(self):
        """Get metadata about the image.

        Returns
        --------
        Dictionary
            Contains:
                - levels: magnification levels included
                - magnification: magnification of raw image
                - resolution: the resolution of the width and height in mcm
                - mm_x: image width in mm
                - mm_y: image height in mm
                - sizeX: image width in pixels
                - sizeY: image height in pixels

        """
        self.validate_index()
        stack = [i for i in self.images if i[0]['idx'] == self.idx][0]
        return({
                "levels": len(stack),
                "magnification": int(stack[0]['magnification']),
                "resolution": (stack[0]['pixel_size'][0], stack[0]['pixel_size'][1]),
                "mm_x": stack[0]['resolution'][0] * stack[0]['pixel_size'][0] / 1000,
                "mm_y": stack[0]['resolution'][1] * stack[0]['pixel_size'][1] / 1000,
                "sizeX": stack[0]['resolution'][0],
                "sizeY": stack[0]['resolution'][1]
              })
    
    def close(self):
        """Close the Javabridge VM"""
        javabridge.kill_vm()

class ImageLoaderOpenSlide(openslide.OpenSlide):
    """Image loader class using OpenSlide."""
    def __init__(self, filename):
        """Init Image loader class using OpenSlide.

        Arguments
        ----------
        filename : string
            path to the whole slide image file
        """
        super().__init__(filename)

    def get_thumbnail(self, dimension=(256,256)):
        """Get thumbnail that fit within specified dimension.

        Arguments
        ----------
        dimension : tuple(2)
            Specify the dimension within which the thumbnail will fit, default to (256,256)

        Returns
        --------
        numpy array of the thumbnail
        """
        return np.array(super().get_thumbnail(dimension))

    def get_info(self):
        """Get metadata about the image.

        Returns
        --------
        Dictionary
            Contains:
                - levels: magnification levels included
                - magnification: magnification of raw image
                - mm_x: physical pixel size (in mm) in x axis
                - mm_y: physical pixel size (in mm) in y axis
                - sizeX: x axis resolution
                - sizeY: y axis resolution
                - tileHeight: x axis resolution of tile
                - tileWidth: y axis resolution of tile

        """
        return {
            "levels": int(self.properties['openslide.level-count']),
            "magnification": int(self.properties['openslide.objective-power']),
            "mm_x": float(self.properties['openslide.mpp-x']) / 1000,
            "mm_y": float(self.properties['openslide.mpp-y']) / 1000,
            "sizeX": self.dimensions[0],
            "sizeY": self.dimensions[1],
            "tileHeight": int(self.properties['openslide.level[0].tile-height']),
            "tileWidth": int(self.properties['openslide.level[0].tile-width'])
            }


    def get_region(self, xmin, xmax, ymin, ymax, mag):
        """Get image within bounding box, scaled to magnification

        Arguments
        ----------
        xmin : int
        xmax : int
        ymin : int
        ymax : int
        mag : int

        Returns
        --------
        numpy array of the image
        """
        orig_mag = self.get_info()['magnification']

        region = self.read_region((xmin, ymin), 0, (xmax - xmin, ymax - ymin))
        scaled_region = region.resize((int(region.size[0] * mag / orig_mag), int(region.size[1] * mag / orig_mag)))
        scaled_region_3_channel = Image.new("RGB", scaled_region.size, (255,255,255))
        scaled_region_3_channel.paste(scaled_region, mask=scaled_region.split()[3])
        scaled_region_3_channel_array = np.array(scaled_region_3_channel)
        return scaled_region_3_channel_array
