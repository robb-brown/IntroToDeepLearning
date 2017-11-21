from setuptools import setup

setup(name='tfs',
	version='0.1',
	description='Support Package for the Tensor Flow Deep Learning Seminar',
	url='http://github.com/robb-brown/DeepLearning',
	author='Robert A. Brown',
	author_email='robert.brown@mcgill.ca',
	license='MIT',
	packages=['tfs'],
	install_requires=[
		'numpy',
		'matplotlib',
		'nibabel',
	],
	zip_safe=False)

