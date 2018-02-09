from nibabel.processing import resample_from_to

import nibabel as nib
import numpy as np


data_dir = 'E:/brains/'

lr_tracts = ['Anterior_Segment', 'Arcuate', 'Cingulum', 'Cortico_Ponto_Cerebellum', 'Cortico_Spinal', 'Inferior_Cerebellar_Pedunculus', 'Inferior_Longitudinal_Fasciculus', 'Inferior_Occipito_Frontal_Fasciculus', 'Long_Segment', 'Optic_Radiations', 'Posterior_Segment', 'Superior_Cerebelar_Pedunculus', 'Uncinate']
other_tracts = ['Anterior_Commissure', 'Corpus_Callosum', 'Fornix', 'Internal_Capsule']

img = nib.load(data_dir + 'test.nii.gz')
# atlas = nib.load(data_dir + 'atlases/Catani/Arcuate/Arcuate_Left.nii')
# img2 = resample_from_to(atlas, img)


for tract_name in lr_tracts:
    left = nib.load(data_dir + 'atlases/Catani/all/' + tract_name + '_Left.nii')
    right = nib.load(data_dir + 'atlases/Catani/all/' + tract_name + '_Right.nii')

    output = nib.Nifti1Image(left.get_data() + right.get_data(), left.affine)

    atlas = resample_from_to(output, img)
    nib.save(atlas, data_dir + 'atlases/Catani/all/resampled/' + tract_name + '.nii')

for tract_name in other_tracts:
    atlas = nib.load(data_dir + 'atlases/Catani/all/' + tract_name + '.nii')
    output = resample_from_to(atlas, img)
    nib.save(output, data_dir + 'atlases/Catani/all/resampled/' + tract_name + '.nii')