# ESSL_GAN_HSI
##Implementation of an Extended Semi Supervised Learning Generative Adversarial Network (ESSL GAN) for hyper-spectral imagery 

The following code shows an implementation of a GAN trained using the Extended Semi-Supervised Learning methodology introduced in 

> A. Hahn, M. Tummala and J. Scrofani, "Extended Semi-Supervised Learning GAN for Hyperspectral Imagery Classification," *2019 13th International Conference on Signal Processing and Communication Systems (ICSPCS)*, 2019, pp. 1-6, doi: 10.1109/ICSPCS47537.2019.9008719. (https://ieeexplore.ieee.org/document/9008719)

## Extended Semi Supervised Learning
Extended Semi Supervised Learning provides an enhanced training scheme to Semi-Supervised learning by randomly generating conditional elements to the Generator input and updating the Generator loss function to include the conditional element. The reuslting network yeilds a highly accurate Generator network for controlable synthetic data generator and a dicriminator network, which yeilds a high accuracy classifier when transfered. Additionally, this scheme naturally lends itself to anomolay detection and is shown to be effective in this area in other work.

## Hyper Spectral Imagery
This implementation is validated using Hyper-spectral Imagery as the data means. The "Indian Pines" data set comes from (http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) and shows a scene with 16 classes. Each sample is a 204 band spectral response. This network is optimized for this dataset. 

## Running this model
The code used to implement ESSL GAN has four main components, GAN Run, GAN model, GAN utilities, and data set scrips. 

- Clone this repository
- Install requirements from requirements.txt. Python version should be 3.6 or 3.7.
- Run GAN_IP_RUN. This will automatically create data set and logging utilities.
- Upon completion of the model run, a folder containing completed test data will be in the working directory, labeled in the form Test\_YYYY\_MM\_DD\_HH\_MM. The folder contains the following:

1. *README.txt*: Text file containing the summary structure of the discriminator, generator, and GanModel() class parameters for that run. 
2. *d\_weight.h5*: H5 file containing the trained discriminator weights. 
3. *g\_weight.h5*: H5 file containing the trained generator weights.
4. *acc\_log\_YYYY\_MM\_DD\_HH\_MM.csv*: CSV file containing test data with respect to the performance metrics.
5. *hsi\_loss\_log\_YYYY\_MM\_DD\_HH\_MM.csv*: CSV file containing the generator and discriminator losses.     

- Once the GAN training run has been completed, the weights from the discriminator can be loaded into separate networks for classifier training and testing.  
