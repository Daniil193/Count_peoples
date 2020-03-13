## Count peoples on video in certain frame area

### _Settings_
0) - Clone or download this repository;
1) - Choose and download weights and config files for NN from [this](https://pjreddie.com/darknet/yolo) website;
2) - Put these files in Temp folder;
3) - pip install -r requirements.txt.

**Warning**

Best practices: Use ffmpeg instead cv2 for reading frames and determining time for certain frame

### _Run processing video_
In Base.ipynb set following variables:
  - path_to_video: full path to folder with videos;
  - temp_folder: full path to Temp folder with (coco.names, labelImg.py, predefined_classes.txt);
  - Set names for downloaded weights and config files:
  
  net, outputlayers = load_pretrained_model(temp_folder, '**model**.weights', '**model**.cfg').
  - After starting all cells, in the labelImg program select and save the necessary area and then close program window.

In Jupyter Notebook -> Base.ipynb -> Cell -> Run All
