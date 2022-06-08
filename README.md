# HiCARN: Resolution Enhancement of Hi-C Data Using Cascading Residual Networks

___________________
#### OluwadareLab, University of Colorado, Colorado Springs
___________________

#### Developers:
Parker Hicks <br />
Department of Biology <br />
Concordia University Irvine <br />
Email: [parker.hicks@eagles.cui.edu](mailto:parker.hicks@eagles.cui.edu)

#### Contact:
Oluwatosin Oluwadare, PhD <br />
Department of Computer Science <br />
University of Colorado, Colorado Springs <br />
Email: [ooluwada@uccs.edu](mailto:ooluwada@uccs.edu)
            
___________________

## Build Instructions:
HiCARN runs in a Docker-containerized environment. Before cloning this repository and attempting to build, install the Docker engine. To install and build HiCARN follow these steps.

1. Clone this repository locally using the command `git clone https://github.com/OluwadareLab/HiCARN.git && cd HiCARN`.
2. Pull the HiCARN docker image from docker hub using the command `docker pull oluwadarelab/hicarn:latest`. This may take a few minutes. Once finished, check that the image was sucessfully pulled using `docker image ls`.
3. Run the HiCARN container and mount the present working directory to the container using `docker run --rm --gpus all -it --name hicarn -v ${PWD}:${PWD} oluwadarelab/hicarn`.
4. `cd` to your home directory.
___________________
## Dependencies
HiCARN is written in Python3 and uses the Pytorch module. All dependencies are included in the Docker environment. <br />
**_Note:_** GPU usage for training and testing is highly recommended.



The following versions are recommended when using HiCARN:
- Python 3.8
- Pytorch 1.10.0, CUDA 11.3
- Numpy 1.21.1
- Scipy 1.7.0
- Pandas 1.3.1
- Scikit-learn 0.15.2
- Matplotlib 3.4.2
- tqdm 4.61.2

___________________

## Data Preprocessing
Click [here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525) to view the GSE62525
GEO accession for Hi-C data from (Rao *et al.* 2014). We used [GM12878](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FCH12%2DLX%5Fintrachromosomal%5Fcontact%5Fmatrices%2Etar%2Egz)
primary intrachromosomal, [K562](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FK562%5Fintrachromosomal%5Fcontact%5Fmatrices%2Etar%2Egz)
intrachromasomal, and [CH12-LX](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FCH12%2DLX%5Fintrachromosomal%5Fcontact%5Fmatrices%2Etar%2Egz)
(mouse) intrachromosomal contact matrices.

* Set your root directory as a string in `Data/Arg_Parser.py`. For example, we set `root_dir = './Datasets_NPZ'`
* Make a new direcrory named `raw` to store your raw datasets. Command:  `mkdir $root_dir/raw`
* Download and Unzip your data into the `$root_dir/raw` directory.  For example for GM12878 data, a folder with
the cell line name will be created containing contact matrices for all chromosomes for all available resolutions. See
the [README](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FOVERALL%5FREADME%2Ertf)
for further details.

Follow the following steps to generate datasets in .npz format:
1. **Read the raw data.** 
   * This will create a new directory `$root_dir/mat/<cell_line_name>` where all chrN_[HR].npz files will be stored.

```bash
$ python Read_Data.py -c GM12878 
```
Required arguments:
* `-c`: Specify only the name of the directory holding the  Unziped Cell line data you downloaded in above `$root_dir/raw/<cell_line_name>`. In our case,  the directory <cell_line_name> = GM12878 

Optional arguments:
* `-hr`: Specified resolution. You can choose from 5kb, 10kb, 25kb, 50kb, 100kb, 250kb, 500kb, and 1mb. Default is 10kb.
* `-q`: Specified map quality. Options are MAPQGE30 and MAPQG0. Default is MAPQGE30.
* `-n`: Normalization. Options are KRnorm, SQRTVCnorm, and VCnorm. Default is KRnorm.

2. **Randomly downsample the data.** This adds downsampled HR data to `$root_dir/mat/<cell_line_name>` as chrN_[LR].npz.

```bash
$ python Downsample.py -hr 10kb -lr 40kb -r 16 -c GM12878
```
All arguments:
* `-hr`: Specified resolution from the previous step. Default is 10kb
* `lr`: Provides a resolution for [LR] in chrN_[LR].npz. Default is 40kb
* `-r`: Downsampling ratio. Default is 16
* `-c`: Cell line name.

3. **Generate train, validation, and test datasets.** 
   * You can set your desired chromosomes for each set in 
   `Data/Arg_Parser.py` within the `set_dict` dictionary. 
   * This specific example will create a file in `$root_dir/data` named 
   hicarn_10kb40kb_c40_s40_b201_nonpool_train.npz. 
   
```bash
$ python Generate.py -hr 10kb -lr 40kb -lrc 100 -s train -chunk 40 -stride 40 -bound 201 -scale 1 -c GM12878
```
All arguments:
* `-hr`: High resolution in chrN_[HR].npz used as a target for training. Default is 10kb.
* `-lr`: Low resolution in chrN_[LR].npz used as training inputs. Default is 40kb.
* `-lrc`: Set the lowest value in the LR matrix. Default is 100.
* `-s`: Dataset to be generated. Options are train, valid, GM12878_test, K562_test, and mESC_test. Default is train.
* `-chunk`: nxn size for each submatrix. Default is 40.
* `-stride`: Set equal to `-chunk`. Default is 40.
* `-bound`: The upper bound of genomic distance. Default is 201.
* `-scale`: Whether to pool input submatrices or not. Default is 1.
* `-c`: That cell line name again...

Congratulations! You now have your datasets. 

***Note***: For training, you must have both training and validation files present in `$root_dir/data`. Change the option `-s` to generate the validation and other datasets needed

___________________
## Using Our Processed Data

Processed data from our `Data/` directory should be placed in your `$root_dir/data` directory. There you can find training and validation files in `Data/Train_and_Validate/` and also test sets in `Data/Test/` where you may choose from a group file containing four chromosomes or a file containing only chromosome 4. 
            
___________________
## Training

We provide training files for both HiCARN-1 and HiCARN-2. 

To train:

```bash
$ python HiCARN_[1 or 2]_Train.py
```
This function will output .pytorch checkpoint files containing the trained weights. During validation, if the highest SSIM score is attained, then the weights of that epoch will be saved as `bestg`. There will be multiple `bestg` checkpoint files during a single training. Once training is complete after the full set of epochs, a `finalg` checkpoint file will be created. We used the `finalg` checkpoint files for our predictions.

**_Note:_** After training HiCARN-2, a `finald` checkpoint file will be generated. This contains the weights for the HiCARN-2 discriminator and is not used in predictions.
___________________
## Predicting

We provide pretrained weights for HiCARN and all other compared models. You can also use the weights generated by 
your own trained model. For quick predictions use the following commands below:

1. If predicting with HiCARN-1, HiCARN-2, or DeepHiC:
```bash
$ python 40x40_Predict.py -m HiCARN_1 -lr 40kb -ckpt root_dir/checkpoints/weights_file.pytorch -f hicarn_10kb40kb_c40_s40_b201_nonpool_human_GM12878_test.npz -c GM12878_HiCARN_1
```

2. If predicting with HiCSR, HiCNN, or HiCPlus:
* These models output a 28x28 matrix from a 40x40 input, so the inputs need to be padded to 52x52 so that a 40x40
output is returned.
```bash
$ python 28x28_Predict.py -m HiCSR -lr 40kb -ckpt root_dir/checkpoints/weights_file.pytorch -f hicarn_10kb40kb_c40_s40_b201_nonpool_human_GM12878_test.npz -c GM12878_HiCSR
```
All arguments:
* `-m`: Model to predict with. Options are HiCARN_1, HiCARN_2, DeepHiC, HiCSR, HiCNN, or HiCPlus.
* `-lr`: Low resolution to be enhanced. Default is 40kb.
* `-ckpt`: Checkpoint file from either our `Pretrained_weights` or your `$root_dir/checkpoints` directory.
* `-f`: Low resolution file name to be enhanced. Must be located in the `$root_dir/data` directory.
  * Example: `hicarn_10kb40kb_c40_s40_b201_nonpool_GM12878_test.npz.`
* `-c`: The cell line just one more time.

___________________

If you would like to perform analysis metrics for your predictions use the following commands:

1. If predicting with HiCARN-1, HiCARN-2, or DeepHiC:
```bash
$ python 40x40_Predict_With_Metrics.py -m HiCARN_1 -lr 40kb -ckpt root_dir/checkpoints/weights_file.pytorch -f hicarn_10kb40kb_c40_s40_b201_nonpool_human_GM12878_test.npz -c GM12878_HiCARN_1
```

2. If predicting with HiCSR, HiCNN, or HiCPlus:
```bash
$ python 28x28_Predict_With_Metrics.py -m HiCSR -lr 40kb -ckpt root_dir/checkpoints/weights_file.pytorch -f hicarn_10kb40kb_c40_s40_b201_nonpool_human_GM12878_test.npz -c GM12878_HiCSR
```
___________________
            
## Accessing Your Predicted Data
            
The output predictions are stored in .npz files that store numpy arrays under keys. The keys for your predicted .npz files are `hicarn` and `compact`. The predicted HR contact map is stored under the `hicarn` key. The `compact` key contains the indices for where there are non-zero entries in the contact map.
            
To access the predicted HR matrix, use the following command in a python file: `hic_matrix = np.load("path/to/file.npz, allow_pickle=True)['hicarn']`.
