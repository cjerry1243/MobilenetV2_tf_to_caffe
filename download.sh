# $1: mobilenet_v2_1.0_224
echo 'downloading '$1
mkdir $1
cd $1
wget 'https://storage.googleapis.com/mobilenet_v2/checkpoints/'$1'.tgz'
tar -xvf $1'.tgz'
cd ..
echo 'Successfully download and untar ' $1