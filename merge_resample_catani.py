from nibabel.processing import resample_from_to

import nibabel as nib
import numpy as np


data_dir = 'E:/brains/'

lr_tracts = ['Anterior_Segment', 'Arcuate', 'Cingulum', 'Cortico_Ponto_Cerebellum', 'Cortico_Spinal', 'Inferior_Cerebellar_Pedunculus', 'Inferior_Longitudinal_Fasciculus', 'Inferior_Occipito_Frontal_Fasciculus', 'Long_Segment', 'Optic_Radiations', 'Posterior_Segment', 'Superior_Cerebellar_Pedunculus', 'Uncinate']
other_tracts = ['Anterior_Commissure', 'Corpus_Callosum', 'Fornix', 'Internal_Capsule']

combined_tracts = ['Projection_Network', 'Cerebellar_Network', 'Inferior_Network', 'Perisylvian_Network']

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

for tract_name in combined_tracts:
    affine = nib.load(data_dir + 'atlases/Catani/all/resampled/Internal_Capsule.nii').get_affine()

    if 'Projection' in tract_name:
        atlas1 = nib.load(data_dir + 'atlases/Catani/all/resampled/Internal_Capsule.nii').get_data()
        atlas2 = nib.load(data_dir + 'atlases/Catani/all/resampled/Cortico_Spinal.nii').get_data()

        output = (atlas1 + atlas2) / 2

    if 'Cerebellar' in tract_name:
        atlas1 = nib.load(data_dir + 'atlases/Catani/all/resampled/Cortico_Ponto_Cerebellum.nii').get_data()
        atlas2 = nib.load(data_dir + 'atlases/Catani/all/resampled/Superior_Cerebellar_Pedunculus.nii').get_data()
        atlas3 = nib.load(data_dir + 'atlases/Catani/all/resampled/Inferior_Cerebellar_Pedunculus.nii').get_data()

        output = (atlas1 + atlas2 + atlas3) / 3

    if 'Inferior' in tract_name:
        atlas1 = nib.load(data_dir + 'atlases/Catani/all/resampled/Inferior_Longitudinal_Fasciculus.nii').get_data()
        atlas2 = nib.load(data_dir + 'atlases/Catani/all/resampled/Inferior_Occipito_Frontal_Fasciculus.nii').get_data()
        atlas3 = nib.load(data_dir + 'atlases/Catani/all/resampled/Uncinate.nii').get_data()

        output = (atlas1 + atlas2 + atlas3) / 3

    if 'Perisylvian' in tract_name:
        atlas1 = nib.load(data_dir + 'atlases/Catani/all/resampled/Anterior_Segment.nii').get_data()
        atlas2 = nib.load(data_dir + 'atlases/Catani/all/resampled/Long_Segment.nii').get_data()
        atlas3 = nib.load(data_dir + 'atlases/Catani/all/resampled/Posterior_Segment.nii').get_data()

        output = (atlas1 + atlas2 + atlas3) / 3

    to_save = nib.Nifti1Image(output, affine)
    nib.save(to_save, data_dir + 'atlases/Catani/all/combined/' + tract_name + '.nii')
