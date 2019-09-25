# BioCaffe

Please Read the instruction  to generate Caffe Model for Leaf Detection

1. Load the Leaf image Data

   1. mkdir ~/data
   2. cd ~/data
   3. git clone https://github.com/sanjeeviisl/LeafDetectData.git

2. Add the Environmental Varible for BioCaffe

   1. vi ~/.bashrc
   2. export CAFFE_ROOT=/home/sanjeev/biocaffe
   3. export PYTHONPATH=/home/sanjeev/biocaffe/python
   4. source  ~/.bashrc


3. go to $CAFFE_ROOT
   
   1. make all
   2. make test
   3. make runtest 
   4. make pycaffe

4. go to $CAFFE_ROOT/data and run the script to create LMDB files
    
   1. ./create_list.sh
   2. ./create_data.sh

