# Count peoples on video on certain frame area

## Settings
0) - Clone or download <h3>Count_peoples</h3> repository
1) - Choose and download weights and config files for NN from [this](https://pjreddie.com/media/files/yolov3-spp.weights/, pjreddie.com) website
2) - Put these files in Temp folder 
3) - pip install -r requirements.txt

### Run processing video
In Base.ipynb set following variables:
  - path_to_video: full path to folder with videos;
  - temp_folder: full path to Temp folder with (coco.names, labelImg.py, predefined_classes.txt)
  - Set names for downloaded weights and config files:
    net, outputlayers = load_pretrained_model(temp_folder, '<h3>model</h3>.weights', '<h3>model</h3>.cfg') 
                                
                                          


 
