if [ -z $1 ] ; then
    GPU=all
else
    GPU=$1
fi

docker run -it \
  -u $(id -u):$(id -g) \
  --rm \
  --gpus '"device='$GPU'"' \
  --hostname $(hostname) \
  -e HOME \
  -v /mnt/roahm:/mnt/roahm \
  -w /home/PUT_NAME_HERE \
  --ipc "host"\
  -v $(pwd):/home/PUT_NAME_HERE \
   multispectral-obj-det:latest
