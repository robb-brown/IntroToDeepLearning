import nibabel as nib
import glob,os
import numpy

dataPath = '.'

images = glob.glob(os.path.join(dataPath,'?????.nii.gz'))
images = [(i,i.replace('.nii.gz','_cc.nii.gz')) for i in images]

for image in images:
	im = nib.load(image[0]); mask = nib.load(image[1])
	slices = numpy.unique(numpy.where(mask.get_data()>0)[0])
	im2 = im.get_data()[slices.min()-1:slices.max()+2]
	mask2 = mask.get_data()[slices.min()-1:slices.max()+2]
	im2 = nib.Nifti1Image(im2.astype(numpy.int16),im.affine)
	nib.save(im2,os.path.join(dataPath,'smaller',os.path.basename(image[0])))
	mask2 = nib.Nifti1Image(mask2.astype(numpy.uint8),mask.affine)
	nib.save(mask2,os.path.join(dataPath,'smaller',os.path.basename(image[1])))
