# zetadesign

## Environment
You can configure the environment using either `requirements.txt` or `conda+environment.yaml`. It is recommended to use the latter as some packages have specific requirements. 
For example, `pdbfixer` is recommended to be installed using `conda install -c conda-forge pdbfixer` as it requires the latest version of `pdbfixer` which uses `openmm8.0`. 
However, this may cause conflicts with other packages, so please be cautious during installation. If you encounter any issues, feel free to contact me via email.

## Data
We use the cath4.3-s40-no redundant dataset, which can be downloaded using the multi-threaded script `/data/download-chains.py`. Alternatively, 
you can directly download the complete files from [ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/non-redundant-data-sets/].

We filter the original files using `/data/cath/spilt_dataset.py`, for example by applying a resolution threshold of <3 Å, 
and then split the dataset into training and test sets following the 95% - 5% ratio described in the paper. This process generates a final text file with the dataset configuration.
