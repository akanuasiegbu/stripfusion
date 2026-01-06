Sanitized Training Annotations of KAIST

To study the effects of annotation noise, we create a sanitized version of KAIST training annotations. 

Since annotating the whole training data is time-consuming (~50,000 frames), we first filtered the training images using the original annotations. We sampled images every 2 frames from training videos, excluded heavily occluded, truncated and small (< 50 pixels) pedestrian instances, thus obtained 7601 training images that contain at least one valid pedestrian instances. We only annotated this subset of training images. 

The annotation format generally follows the orignal one. There are two minor changes.
1. Instances that do not spatially align between color and thermal images are labelled as `person?a'.
2. Both `person' and `cyclist' categories are lablled as `person', since even the human annotators found it difficult to distinguish between pedestrians and cyclists when the illumination condition is poor or the resolution is small.

If you use this data, please cite the paper:

@inproceedings{li2018multispectral,
  title={Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation},
  author={Li, Chengyang and Song, Dan and Tong, Ruofeng and Tang, Min},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2018}
}

