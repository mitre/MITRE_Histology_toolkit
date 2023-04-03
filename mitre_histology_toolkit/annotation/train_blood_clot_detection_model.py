from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from skimage.draw import polygon
from PIL import Image
import pandas as pd
import scipy.sparse
import numpy as np
import slideio
import pickle
import json
import os

def load_image(image_path, magnification):
    ts0 = slideio.open_slide(image_path, image_path[-3:].upper())
    ts = ts0.get_scene(0)
    sx, sy = int(ts.size[0]), int(ts.size[1])
    ratio = magnification / ts.magnification
    new_size = (int(sx * ratio), int(sy * ratio))
    im_low_res = ts.read_block((0, 0, sx, sy), size = new_size)
    return(im_low_res)

def get_tissue_mask(im_low_res, image_name, project, annotation_path = None, nv = 200):
    
    tissue_mask = np.where((im_low_res[:,:,0] < nv) |
                           (im_low_res[:,:,1] < nv) |
                           (im_low_res[:,:,2] < nv), 1, 0)
    if project == 'HSS_OA':
        annotation_mask = scipy.sparse.load_npz(f'{annotation_path}/{image_name}'.replace('.svs', '.npz').replace('blood_clots', 'HSS_OA'))
        annotation_mask = annotation_mask.toarray()
        
        with open(f'{annotation_path}/{image_name}'.replace('.svs', '.json'), 'r') as anno_file:
            annotation_params = json.loads(anno_file.read())
            
        # Remove tissue components
        for ii in annotation_params['not_synovium']:
            annotation_mask[annotation_mask == ii] = 0
        
        anno_mask = np.where(annotation_mask > 0, 1, 0)
    
        tissue_mask = tissue_mask * anno_mask
    
    return(tissue_mask)

def assign_bc_vals(im_cut, coords):
    coords = np.array(coords)
    r, c = polygon(coords[:,1], coords[:,0])
    im_cut[r,c] = -1
    return(im_cut)

def get_blood_clot_mask(tissue_mask, blood_clot_model_path, image_name):
    blood_clot_mask = np.copy(tissue_mask)
    for ii0 in range(10):
        bci_path = f'{blood_clot_model_path}/{image_name}'.replace('.svs', f'_bc_{ii0}.json')
        if os.path.isfile(bci_path):
            with open(bci_path, 'r') as anno_file:
                blood_clot_params = json.loads(anno_file.read())
            
            coords = blood_clot_params['blood_clots']
            blood_clot_mask = assign_bc_vals(blood_clot_mask, coords)
        else:
            break
    
    return(blood_clot_mask)

def resample_imbalnce(xx, yy):
    labels, counts = np.unique(yy, return_counts=True)
    if counts[1] < counts[0]:
        labels = np.array([labels[1], labels[0]])
        counts = np.array([counts[1], counts[0]])
    diff = counts[1] - counts[0]
    yind = (yy == labels[0])
    ratio = int(diff // counts[0])
    xx2 = np.tile(xx[yind], (ratio, 1))
    yy2 = np.repeat(yy[yind], ratio)
    yind_alt = (yy == labels[1])
    xxf = np.concatenate((xx[yind_alt], xx2))
    yyf = np.concatenate((yy[yind_alt], yy2))
    return(xxf, yyf)

def gen_classifier_data(tissue_mask, blood_clot_mask, im_low_res):
    final_image = tissue_mask * blood_clot_mask
    bc_vals = im_low_res[final_image == -1]
    reg_vals = im_low_res[final_image == 1]
    
    image_obj = Image.fromarray(im_low_res.astype('uint8'))
    hsv_image = np.array(image_obj.convert('HSV'))
    bcv = hsv_image[final_image == -1]
    rgv = hsv_image[final_image == 1]
    
    X_l = np.concatenate((reg_vals, bc_vals))
    X_r = np.concatenate((rgv, bcv))
    X = np.concatenate((X_l, X_r), axis = 1)

    y = np.array(['Other']*len(reg_vals) + ['Blood']*len(bc_vals))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.5, 
                                                        random_state=44,
                                                        stratify=y)

    X_train, y_train = resample_imbalnce(X_train, y_train)
    return(X_train, X_test, y_train, y_test)

blood_clot_model_path = 'data/processed/annotations/models/blood_clots'
magnification = 4

img_names = set()
for fn in os.listdir(blood_clot_model_path):
    if '.json' in fn:
        img_names.add(fn.split('_bc_')[0] + '.svs')

xtr, xte, ytr, yte = [], [], [], []
for image_name in img_names:
    project = 'HSS_RA'
    if '_SYN_' in image_name:
        project = 'HSS_OA'
    
    slide_path = f'data/raw/slides/{project}'
    annotation_path = f'data/processed/annotations/{project}/menisci'
    image_path = f'{slide_path}/{image_name}'
    im_low_res = load_image(image_path, magnification)
    tissue_mask = get_tissue_mask(im_low_res, image_name, project, annotation_path = annotation_path, nv = 200)
    blood_clot_mask = get_blood_clot_mask(tissue_mask, blood_clot_model_path, image_name)
    X_train_temp, X_test_temp, y_train_temp, y_test_temp = gen_classifier_data(tissue_mask, blood_clot_mask, im_low_res)
    xtr += [X_train_temp]
    xte += [X_test_temp]
    ytr += [y_train_temp]
    yte += [y_test_temp]

X_train = np.concatenate(xtr)
X_test = np.concatenate(xte)
y_train = np.concatenate(ytr)
y_test = np.concatenate(yte)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

labels = ['Blood', 'Other']
cmc = confusion_matrix(y_test, y_pred, labels = labels)
cmp = np.round(cmc / cmc.sum(1).reshape((2,1)), 4)
cmc_df = pd.DataFrame(cmc)
cmc_df.columns = labels
cmc_df.index = labels
cmp_df = pd.DataFrame(cmp)
cmp_df.columns = labels
cmp_df.index = labels
print(cmc_df)
print(cmp_df)

with open(f'{blood_clot_model_path}/blood_clot_model_rgb_hsv_composite.pkl','wb') as f:
    pickle.dump(clf, f)






