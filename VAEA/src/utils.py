import cv2
import base64
import VAEA.model.TransNetV2.inference.transnetv2 as transnetv2 #镜头边界检测算法

# input: video path 
# output: a list of shot bonaries [[0 192] [192 465] ... ]
def get_shot_boundaries(path):
  model = transnetv2.TransNetV2()
  video_frames, single_frame_predictions, all_frame_predictions = \
    model.predict_video(path)
  scenes = model.predictions_to_scenes(single_frame_predictions)
  #print(scenes)
  return scenes

def print_features(features):
  for index, feature in enumerate(features):
    print(f"shot {index} : {feature}")


# input: frame read by opencv
# output: resized frame, max(height,width) = 512
def resize_frame(frame, limited_size = 512):
  try:
    original_height, original_width = frame.shape[:2]
    if (original_height <= limited_size and original_width <= limited_size):
      return frame
    if (original_width > original_height):
      new_width = limited_size
      new_height = int((original_height / original_width) * new_width)
    else:
      new_height = limited_size
      new_width = int((original_width / original_height) * new_height)
    resized_image = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image
  except:
    print("resize frame failed, check the format of incoming frame")
    return None 

def resize_video(video):
  frames = []
  success, frame = video.read()
  while success:
    frame = resize_frame(frame)
    frames.append(frame)
    success, frame = video.read()
  return frames


def output_shots(shots, folder, fps):
    for index,shot in enumerate(shots):
      output_video = cv2.VideoWriter(f"{folder}/{index}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (shot[0].shape[1], shot[0].shape[0]))
      for frame in shot:
        output_video.write(frame)
      output_video.release()

'''out_base64可以指定输出shots用base64编码还是cv原始数据格式'''
def get_shots(video, scenes, out_base64 = False, limited_size = 256):
  shots = []
  shot = []
  scene_cnt = 0

  while video.isOpened():
  # print(f"current frame:{int(video.get(cv2.CAP_PROP_POS_FRAMES))}")
    success, frame = video.read()
    if not success:
        break
    frame = resize_frame(frame, limited_size = limited_size)
    if out_base64:
      shot.append(base64_encode_frame(frame))
    else:
      shot.append(frame)

    if int(video.get(cv2.CAP_PROP_POS_FRAMES)) > scenes[scene_cnt, 1]  :
      # print(f"current frame cnt:{int(video.get(cv2.CAP_PROP_POS_FRAMES))}")
      shots.append(shot)
      scene_cnt += 1
      shot = []

  if len(shot) > 0:
    shots.append(shot)
    
  video.release()
  
  return shots  

'''将get_shots()中的原始格式转换成base64'''
def shot_to_base64(shot):
    base64_shot = []
    for frame in shot:
      base64_shot.append(base64_encode_frame(frame))
    return base64_shot

def print_video_info(video):
    fps = video.get(cv2.CAP_PROP_FPS)
    frameCnt = video.get(cv2.CAP_PROP_FRAME_COUNT)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("----video info----")
    print(f"{fps} fps, {frameCnt} frameCnt, {width} width, {height} height")
    print("------------------")

def base64_encode_frame(frame):
    _ , buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')