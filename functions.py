import cv2
import numpy as np
import os
import re

def search_persons(OUTS, WIDTH, HEIGHT, TRESH):
    """
    Get boxes for drawing, where score of person detection more or equal TRESHOLD
    Input: 
          - (array) outs from  net.forward(outputlayers)
          - width: (integer) width frame from video
          - height: (integer) height frame from video
          - tresh: (float) treshold by score of network prediction
    Output:
          - boxes: (lists in list) [[x,y,w,h],...] rectangles of object position
          - confidence: (list) score for each boxes
          - class_ids: (list) list of 1s, because we search only person
    """
    class_ids = []
    boxes = []
    confidences = []
    
    for out in OUTS:
        
        for detection in out:
            
            scores = detection[5:]
            person_score = scores[0]
            
            if person_score >= TRESH:
                
                center_x = int(detection[0] * WIDTH)
                center_y = int(detection[1] * HEIGHT)
                w = int(detection[2] * WIDTH)
                h = int(detection[3] * HEIGHT)
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(person_score))
                class_ids.append(1) ## searching only person
                #print(person_score)
                
    return boxes, confidences, class_ids


def draw_rectangles_on_img(boxes, confidences, img):
    """
    Add rectangles for predicted objects on image
    Input: 
          - boxes: (lists in list) [[x,y,w,h],...] values of rectangle position
          - confidence: (list) [0.5, 0.75, 0.9, 0.61] score for each box
          - img: (array) original image for preparing
    Output: (changed image array)add rectabgles for image
    """
    for i in range(len(boxes)):
        font = cv2.FONT_HERSHEY_COMPLEX
        label = 'Person'
        confidence = confidences[i]
        x, y, w, h = boxes[i]
        color = (255, 255, 0) ## yellow rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2);
        cv2.putText(img, label+" "+str(round(confidence, 2)), (x, y-10), font, 0.8, (255,255,0), 1)


def load_pretrained_model(path_to_model, weights, config):
    """
    Load pretrained model with help cv2
    Input: 
          - path_to_model: (string) path to folder with model.weights, model.config, class_names
          - weights: (string) for example 'yolov3-spp.weights'
          - config: (string) for example 'yolov3-spp.cfg'
    Output:
          - net: loaded model in cv2.dnn.readNet
          - outputlayers: last layers of the net
    """
#     if path_to_model[-1] != '/':
#         path_to_model = path_to_model + '/'
        
    net = cv2.dnn.readNet(path_to_model + weights, path_to_model + config)
    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    
#     with open(path_to_model+class_names, "r") as f:
#         classes = [line.strip() for line in f.readlines()]
        
    return net, outputlayers


def secondsToText(secs):
    """
    Conversion seconds to format (H:M:S)
    Input: (integer) 321
    Output: (string) 5min:21sec:
    """
#     secs = int(np.median(list_time))
    if secs > 1:
        days = round(secs//86400)
        hours = round((secs - days*86400)//3600)
        minutes = round((secs - days*86400 - hours*3600)//60)
        seconds = round(secs - days*86400 - hours*3600 - minutes*60)
        result = ("{}d:".format(days) if days else "") + \
        ("{}hrs:".format(hours) if hours else "") + \
        ("{}min:".format(minutes) if minutes else "") + \
        ("{}sec:".format(seconds))
    else:
        result = str(round(secs, 4)) +' '+ 'sec'
    
    return result



def write_to_txt(filename, fps, count_frame, boxes):
    """
    Write info to txt file
    Input:
          - filename: (string) name of video file
          - fps: (integer) frame per seconds of video
          - count_frame: (integer) current frame number
          - boxes: (list) of rectangles coordinates [[x,y,w,h], ....]
    Output:
          - txt file with columns ['FILENAME', 'TIME', 'COUNT_PEOPLE']
    """
    if os.path.exists('counts_people.txt'):
        with open('counts_people.txt', 'a') as f:
            f.write(f'{filename}'+'<--->'+f'{secondsToText(count_frame/fps)}'+'<--->'+f'{len(boxes)}'+'\n')
    else:
        with open('counts_people.txt', 'w') as f:
            f.write('FILENAME'+'<--->'+'TIME'+'<--->'+'COUNT_PEOPLE'+'\n')
        with open('counts_people.txt', 'a') as f:
            f.write(f'{filename}'+'<--->'+f'{secondsToText(count_frame/fps)}'+'<--->'+f'{len(boxes)}'+'\n')



def processing_image(image, net, outputlayers, treshold):
    """
    Count persons on frame
    Input: 
          - image: (array) from cv2
          - net: from load_pretrained_model function
          - outputlayers: from load_pretrained_model function
          - treshold: (float) treshold by score of network prediction
    Output:
          - boxes: (lists in list) [[x,y,w,h],...] rectangles coordinates of object position
          - confidence: (list) score for each boxes 
    """
    
    height, width, channels = image.shape
    
    blob = cv2.dnn.blobFromImage(image, 0.00392, (320, 320), (0,0,0), True, crop=False) ## SET IMAGE SIZE, quickly ~ 320*320
    net.setInput(blob)
    outs = net.forward(outputlayers)
    
    boxes, confidences, class_ids = search_persons(outs, width, height, treshold)
    
    return boxes, confidences


def processing_video(path_to_video, file_name, interval, net, outputlayers, coordinates, path_to_temp_folder, threshold):
    """
    Processing videos for count person in frame
    Input:
          - path_to_video: (string)
          - path_to_temp_folder: (string)
          - file_name: (string) - name video file for processing
          - interval: (integer) - frame rate for processing or how many seconds between video fragment processing
          - net: loaded model in cv2.dnn.readNet
          - outputlayers: last layers of the net
          - coordinates: (tuple of integers) - xmin, ymin, xmax, ymax
          - threshold: assessment of relation to a person
    """
    vid = cv2.VideoCapture(path_to_video + file_name)
    fps = vid.get(cv2.CAP_PROP_FPS) ##not always work incorrect     SET FPS FOR PROCESSING
    frame_per_time = fps * interval
    
    xmin, ymin, xmax, ymax = coordinates
    count_frame = 1
    success, frame = vid.read()
    while success:
        
        if (count_frame % frame_per_time == 0) or (count_frame == int(vid.get(cv2.CAP_PROP_FRAME_COUNT))):
            
            crop_frame = frame[ymin:ymax, xmin:xmax]
            boxes, confidences = processing_image(crop_frame, net, outputlayers, threshold) #### SET TRESHOLD FOR PROCESSING ~ 0.37
            boxes, confidences = drop_duplicated_box(boxes, confidences)
            write_to_txt(file_name, fps, count_frame, boxes)
            
            ##################################### save images #########################################            
            if len(boxes) > 0:  ## if count people more then 0
                
                if not os.path.exists(path_to_temp_folder+'images'):
                    os.makedirs(path_to_temp_folder+'images')
                
                draw_rectangles_on_img(boxes, confidences, crop_frame)
                cv2.imwrite(path_to_temp_folder + f'images/frame_{str(count_frame/fps)}+_sec.jpg', crop_frame)
            ##################################### save images #########################################
        
        success, frame = vid.read()
        count_frame += 1
    
    vid.release()
    cv2.destroyAllWindows()



def get_img_for_coord(path_to_video, file_name, temp_folder):
    """
    Get single frame from video for crop image coordinates
    Input: 
          - path_to_video: (string) path to folder with video from single camera
          - file_name: (string) any name video file
    Output:
          - .jpg file in temp folder
    """
    vid = cv2.VideoCapture(path_to_video + file_name)
    success, frame = vid.read()
    cv2.imwrite(temp_folder + 'Coordinates.jpg', frame)
    vid.release()
    cv2.destroyAllWindows()


def get_x_y_from_xml(path_to_file):
    """
    Parse x-y values from xml file
    Input: (string) path to file
    Output: (integers): xmin, ymin, xmax, ymax
    """
    name_file = [i for i in os.listdir(path_to_file) if '.xml' in i][0]
    
    with open(path_to_file + name_file, 'r') as xml:
        lines = [line.strip() for line in xml.readlines()]
    x_y = [i for i in lines if ('xmin' in i) or  ('xmax' in i) or  ('ymin' in i) or  ('ymax' in i)]
    
    xmin = int(re.sub("[<>/xmin]", '', x_y[0]))
    ymin = int(re.sub("[<>/ymin]", '', x_y[1]))
    xmax = int(re.sub("[<>/xmax]", '', x_y[2]))
    ymax = int(re.sub("[<>/ymax]", '', x_y[3]))
    
    return xmin, ymin, xmax, ymax


def drop_duplicated_box(boxes, confidences):
    
    new_boxes = []
    new_confidences = []
    if len(boxes) > 1:
        for i in range(len(boxes)-1):
            if np.mean( np.subtract(boxes[i], boxes[i+1]), axis = 0) > 20: ### treshold by mean of subtract arrays
                if boxes[i] not in new_boxes:
                    new_boxes.append(boxes[i])
                    new_confidences.append(confidences[i])
                if boxes[i+1] not in new_boxes:
                    new_boxes.append(boxes[i+1])
                    new_confidences.append(confidences[i+1])
            else:
                if boxes[i] not in new_boxes:
                    new_boxes.append(boxes[i+1])
                    new_confidences.append(confidences[i+1])
    else:
        new_boxes = boxes
        new_confidences = confidences
        
    return new_boxes, new_confidences








