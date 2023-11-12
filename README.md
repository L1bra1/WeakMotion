# WeakMotionNet
This is the PyTorch code for [Weakly Supervised Class-Agnostic Motion Prediction for Autonomous Driving (CVPR2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Weakly_Supervised_Class-Agnostic_Motion_Prediction_for_Autonomous_Driving_CVPR_2023_paper.pdf).
The code is created by Ruibo Li (ruibo001@e.ntu.edu.sg).


## Prerequisities
* Python 3.7
* NVIDIA GPU + CUDA CuDNN
* PyTorch (torch == 1.9.0)


Create a conda environment for WeanMotionNet:
```
conda create -n WeakMotion python=3.7
conda activate WeakMotion
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=10.2 -c pytorch
pip install numpy tqdm scikit-learn opencv-python matplotlib pyquaternion
```

## Data preprocess

### nuScenes
1. Prepare the input data and the motion ground truth:
   - Download the [nuScenes data](https://www.nuscenes.org/), and then follow [MotionNet](https://www.merl.com/research/?research=license-request&sw=MotionNet) to process the training, validation, and test data.
2. Prepare the Foreground/Background data for weak supervision:
   -  Download the [nuScenes-lidarseg data](https://www.nuscenes.org/nuscenes#download), and then run
      ```
      python gen_data/gen_weak_data.py --root /path_to/nuScenes/nuScenes-data/ --split train --savepath /path_to/nuScenes/weak-data/
      ```
      The Foreground/Background information for training point clouds will be saved in `/path_to/nuScenes/weak-data/`. Please references `gen_data/README.md` for more details.


​    
### Waymo
1. Prepare the Waymo data:
   - Download the [Waymo Open Dataset (v1.3.2)](https://waymo.com/open/download/).
   - install required library in the conda environment:
        ```
        pip install waymo-open-dataset-tf-2-6-0
        ```
   - run the below command to convert the `.tfrecord` files to `.npy` files:
        ```
        python gen_data/step1_waymo_prepare.py --waymo_tfrecord_dir /path_to/Waymo/Waymo-tf-data/ --split training --waymo_save_dir /path_to/Waymo/Waymo-npy-data/
        ```
     The `.npy` data will be saved in `/path_to/Waymo/Waymo-npy-data/`. After processing the training set, set `--split` to `validation` to prepare the validation data.

2. Prepare the input data, motion ground truth, and Foreground/Background data:
   - Run the command:
        ```
        python gen_data/step2_waymo_generate_weak.py --DATA_DIR /path_to/Waymo/Waymo-npy-data/ --SAVE_ROOT_DIR /path_to/Waymo/
        ```
     The input data and ground truth will be saved in `/path_to/Waymo/input-data/`, and the Foreground/Background data will be saved in `/path_to/Waymo/weak-data/`.

Please references `README.md` in `gen_data` for more details.






## Evaluation

### Trained models
The trained models can be downloaded from the following links.
1. nuScenes

    | Annotation ratio       | Trained PreSegNet (Stage1)  | Trained WeakMotionNet (Stage2) |
    | :------------: | :---------: | :---------: |
    | 0.1%  FG/BG masks    |  [nuscenes_seg_0-001](https://drive.google.com/file/d/1HdkkNe4POnoMAwVbPmPDBMNm_Pn0aaZX/view?usp=sharing)    | [nuscenes_motion_0-001](https://drive.google.com/file/d/17ZQqZO56TvuqKLDacN14IHPluYsHLICZ/view?usp=sharing)  |
    | 1.0%   FG/BG masks  |  [nuscenes_seg_0-01](https://drive.google.com/file/d/1jUvpqjwigIqLqqhA65CRS2Ju-hh-tRDs/view?usp=sharing)    | [nuscenes_motion_0-01](https://drive.google.com/file/d/10jD1qeSkOd1Zsc4UrqbSION0vZ9MHA29/view?usp=sharing)  |
    | 100%   FG/BG masks  |  -    | [nuscenes_motion_1-0](https://drive.google.com/file/d/1FUaekaGbtn_hJUl72TtWkY5UC-s_oTUA/view?usp=sharing)  |

2. Waymo

    | Annotation ratio       | Trained PreSegNet (Stage1)  | Trained WeakMotionNet (Stage2) |
    | :------------: | :---------: | :---------: |
    | 0.1%  FG/BG masks    |  [waymo_seg_0-001](https://drive.google.com/file/d/17jpk4Kl5-fts8WtSkjAsci0gg74fifhV/view?usp=sharing)    | [waymo_motion_0-001](https://drive.google.com/file/d/1pjwmFRFf_sN7qWpF6uV_vgHqIi24q0tb/view?usp=sharing)  |
    | 1.0%   FG/BG masks  |  [waymo_seg_0-01](https://drive.google.com/file/d/1a8Za6eYxcOGsVkUE95JVJyW9tKV69fyP/view?usp=sharing)    | [waymo_motion_0-01](https://drive.google.com/file/d/11wLWmTprA_7Z2IlWadq2pqt0ZiHceh-r/view?usp=sharing)  |
    | 100%   FG/BG masks  |  -    | [waymo_motion_1-0](https://drive.google.com/file/d/1I_lZtsrQ4LzauldIQJH4_L1K--EDp_EG/view?usp=sharing)  |


### Testing

  -  Taking the WeakMotionNet trained on nuScenes data as an example, download the pretrained WeakMotionNet,`nuscenes_motion_1-0`, from the above link, and save it in `pretrained` folder.
  -  Run the command:
        ```
      python evaluate_WeakMotionNet.py --evaldata /path_to/nuScenes/input-data/test/ --pretrained pretrained/nuscenes_motion_1-0.pth --datatype nuScenes
      ```
     set `evaldata` to the directory of the test data (e.g., `/path_to/nuScenes/input-data/test/`).

     set `pretrained` to the trained model (e.g., `pretrained/nuscenes_motion_1-0.pth`).

     set `datatype` to `nuScenes` .

 - When testing the WeakMotionNet trained on `Waymo`, run the command:
      ```
      python evaluate_WeakMotionNet.py --evaldata /path_to/Waymo/input-data/val/ --pretrained pretrained/waymo_motion_1-0.pth --datatype Waymo
      ```

     set  `evaldata` to `/path_to/Waymo/input-data/val/`.

     set `pretrained` to the trained model (e.g., `pretrained/waymo_motion_1-0.pth`).

     set `datatype` to `Waymo`.



## Training
In the following, we take the training on nuScenes data as an example.


### Training with partially annotated Foreground/Background masks
1. Training of PreSegNet (Stage1):

      - Run the command:

        ```
        python train_PreSegNet.py --data /path_to/nuScenes/weak-data/train/ --evaldata /path_to/nuScenes/input-data/val/ --datatype nuScenes --annotation_ratio 0.01
        ```
        set `data` to the directory of the weak training data (e.g., `/path_to/nuScenes/weak-data/train/`).

        set `evaldata` to the directory of the input validation data (e.g., `/path_to/nuScenes/input-data/val/`).

        set `datatype` to `nuScenes` or `Waymo`.

        set `annotation_ratio` to the wanted FG/BG annotation ratio (e.g., 0.01). Ratio can range from 0.001 to 1.

2. Generate FG/BG masks for the training data using the trained PreSegNet

     - Choose a PreSegNet model with higher  Foreground Accuracy  and Overall Accuracy to generate FG/BG masks.
     - generate FG/BG masks for the training data:
       ```
       python predict_FGBG_mask.py --data /path_to/nuScenes/weak-data/train/ --save_FB /path_to/nuScenes/FGBG-data/ --datatype nuScenes --pretrained pretrained/nuscenes_seg_0-01.pth
       ```
       set `data` to the directory of the weak training data (e.g., `/path_to/nuScenes/weak-data/train/`).

       set `save_FB` to the directory to save the predicted FB/BG masks (e.g., `/path_to/nuScenes/FGBG-data/`).

       set `datatype` to `nuScenes` or `Waymo`.

       set `pretrained` to the trained PreSegNet model from Satge1(e.g., `pretrained/nuscenes_seg_0-01.pth`). In addition to training PreSegNet models yourself, we also provide a PreSegNet model trained with 0.01 annotations on nuScenes, [nuscenes_seg-0-01](https://drive.google.com/file/d/1jUvpqjwigIqLqqhA65CRS2Ju-hh-tRDs/view?usp=sharing). Other pretrained PreSegNet models can be found in above.

          Accordingly, the FG/BG masks predicted by `nuscenes_seg_0-01` will be saved in `/path_to/nuScenes/FGBG-data/nuscenes_seg_0-01`.

3. Training of WeakMotionNet (Stage2)
    - Run the command:

        ```
        python train_WeakMotionNet.py --motiondata /path_to/nuScenes/input-data/train/ --weakdata /path_to/nuScenes/weak-data/train/ --FBdata /path_to/nuScenes/FGBG-data/nuscenes_seg_0-01/ --evaldata /path_to/nuScenes/input-data/val/ --datatype nuScenes --annotation_ratio 0.01
        ```

        set `motiondata` to the directory of the input training data (e.g., `/path_to/nuScenes/input-data/train/`).

        set `weakdata` to the directory of the weak training data (e.g., `/path_to/nuScenes/weak-data/train/`).

        set `FBdata` to the directory to the predicted FB/BG masks from the last step (e.g., `/path_to/nuScenes/FGBG-data/nuscenes_seg_0-01/`).

        set `evaldata` to the directory of the input validation or test data (e.g., `/path_to/nuScenes/input-data/val/`).

        set `datatype` to `nuScenes` or `Waymo`.

        set `annotation_ratio` to the FG/BG annotation ratio (e.g., 0.01). Note that the annotation ratio should be the same in the two stages.



### Training with fully annotated Foreground/Background masks
When training WeakMotionNet with fully annotated Foreground/Background masks, the Stage1 and the FB/BG prediction are not required.

   - Run the command:
        ```
        python train_WeakMotionNet.py --motiondata /path_to/nuScenes/input-data/train/ --weakdata /path_to/nuScenes/weak-data/train/ --evaldata /path_to/nuScenes/input-data/val/ --datatype nuScenes --annotation_ratio 1.0
        ```
     The settings are similar to those in Stage2.

     Note that the `annotation_ratio` should be `1.0` and we do not need to set `FBdata`.



## Citation

If you find this code useful, please cite our paper:
```
@inproceedings{li2023weakly,
  title={Weakly Supervised Class-Agnostic Motion Prediction for Autonomous Driving},
  author={Li, Ruibo and Shi, Hanyu and Fu, Ziang and Wang, Zhe and Lin, Guosheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17599--17608},
  year={2023}
}
```

## Acknowledgement

Our project references the codes in the following repos.

* [MotionNet](https://www.merl.com/research/?research=license-request&sw=MotionNet)
* [nuScenes](https://github.com/nutonomy/nuscenes-devkit/tree/master)
* [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset)
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/tree/aa753ec0e941ddb117654810b7e6c16f2efec2f9)
