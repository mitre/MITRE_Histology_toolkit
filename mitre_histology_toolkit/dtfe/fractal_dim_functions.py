import numpy as np

def box_counting_regression(image, lower_binary_exp = 0.01, upper_binary_exp = 10, num_scales = 10):
    """
    

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    lower_binary_exp : TYPE, optional
        DESCRIPTION. The default is 0.01.
    upper_binary_exp : TYPE, optional
        DESCRIPTION. The default is 10.
    num_scales : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    None.

    """
    pixels = np.array(np.where(image > 0)).T
    Lx = image.shape[1]
    Ly = image.shape[0]
    scales = np.logspace(lower_binary_exp, upper_binary_exp, num = num_scales, endpoint = False, base = 2)
    Ns = []
    # looping over several scales
    for scale in scales:
        # computing the histogram
        H, edges = np.histogramdd(pixels, bins = (np.arange(0, Lx, scale), np.arange(0, Ly, scale)))
        if np.sum(H>0) > 0:
            Ns.append(np.sum(H>0))
        else:
            break
    
    if len(Ns) < len(scales):
        print(f'Setting # of Scales to {len(Ns)}')
        scales = scales[:len(Ns)]
    
    # linear fit, polynomial of degree 1
    coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
    return(-coeffs[0], scales, Ns)

def get_box_counting_dimension(image, lower_binary_exp = 0.01, upper_binary_exp = 10, num_scales = 10):
    """
    

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    lower_binary_exp : TYPE, optional
        DESCRIPTION. The default is 0.01.
    upper_binary_exp : TYPE, optional
        DESCRIPTION. The default is 10.
    num_scales : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    None.

    """
    fdim, scales, Ns = box_counting_regression(image, 
                                               lower_binary_exp = lower_binary_exp, 
                                               upper_binary_exp = upper_binary_exp, 
                                               num_scales = num_scales)
    return(fdim)
