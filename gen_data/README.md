
## Data preprocess
### nuScenes 
1. Prepare the Waymo data.

   - Download the [nuScenes data](https://www.nuscenes.org/) and follow [MotionNet](https://www.merl.com/research/?research=license-request&sw=MotionNet) to process the data, we want the data to be saved like this:
   
        ```
        |-- input-data (the processed input data from MotionNet)
        |-- nuScenes-data (downloaded raw data)
        |   |-- maps
        |   |-- samples
        |   |-- sweeps
        |   |-- v1.0-trainval
        ```
2. Prepare the Foreground/Background data for weak supervision:
    - Download the [nuScenes-lidarseg data](https://www.nuscenes.org/nuscenes#download), and then extract the `lidarseg` and `v1.0-*` folders to `nuScenes-data`. Run
      ```
      python gen_data/gen_weak_data.py --root /path_to/nuScenes/nuScenes-data/ --split train --savepath /path_to/nuScenes/weak-data/
      ```
      The final directory should be like this:
      ```
        |-- input-data (the processed input data from MotionNet)
        |-- nuScenes-data (downloaded raw data)
        |   |-- lidarseg
        |   |-- maps
        |   |-- samples
        |   |-- sweeps
        |   |-- v1.0-trainval
        |-- weak-data (data for weak supervision)
      ```

### Waymo 
1. Prepare the Waymo data:
   - Download the [Waymo Open Dataset (v1.3.2)](https://waymo.com/open/download/) and install the library `pip install waymo-open-dataset-tf-2-6-0`. 
   
   - Run this command to convert the `.tfrecord` files to `.npy` files:
        ``` 
        python gen_data/step1_waymo_prepare.py --waymo_tfrecord_dir /path_to/Waymo/Waymo-tf-data/ --split training --waymo_save_dir /path_to/Waymo/Waymo-npy-data/ 
        ```
      The current directory should be like this:
      ```
        |-- Waymo-tf-data (downloaded raw data)
        |-- Waymo-npy-data (converted raw data)
      ```
2. Prepare the input data, motion ground truth, and Foreground/Background data:
   - Run the  command:
        ``` 
        python gen_data/step2_waymo_generate_weak.py --DATA_DIR /path_to/Waymo/Waymo-npy-data/ --SAVE_ROOT_DIR /path_to/Waymo/ 
        ```
     The final directory should be like this:
      ```
        |-- Waymo-tf-data (downloaded raw data)
        |-- Waymo-npy-data (converted raw data)
        |-- input-data (the processed input data)
        |-- weak-data (data for weak supervision)
      ```

## Acknowledgement

The data generation references the codes in the following repos.   
* [MotionNet](https://www.merl.com/research/?research=license-request&sw=MotionNet)
* [nuScenes](https://github.com/nutonomy/nuscenes-devkit/tree/master)
* [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset)
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/tree/aa753ec0e941ddb117654810b7e6c16f2efec2f9)


