if [ -z $1 ] ; then
    GPU=all
else
    GPU=$1
fi

#in gerneral do only read mode but since cache created need to write 
docker run -it \
  -u $(id -u):$(id -g) \
  --rm \
  --gpus '"device='$GPU'"' \
  --hostname $(hostname) \
  -e HOME \
  -v /mnt/roahm:/mnt/roahm \
  -v /mnt/workspace/datasets:/mnt/workspace/datasets \
  -v /mnt/roahm/users/akanu/projects/fusion_therm_rgb/state-of-art-detectors/yolov5:/home/yolov5 \
  -v /home/akanu/akanu/projects/Projects_to_make_public/MambaST:/home/MambaST \
  -v /mnt/ws-frb/projects/thermal_rgb/njotwani/multispectral-video-object-detection/runs/:/home/nitin/ \
  -w /home/akanu \
  --ipc "host"\
  -v $(pwd):/home/akanu \
   multispectral-obj-det:latest

#   -v /mnt/workspace/users/vsrikar/DeformableDetr/data:/mnt/workspace/users/vsrikar/DeformableDetr/data \
  # -v /home/akanu/akanu/local_datasets:/mnt/workspace/datasets \