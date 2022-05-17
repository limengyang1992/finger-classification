import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

import cv2
import time
import glob
import random
import streamlit as st

model_path = "best.onnx"
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trans = transforms.Compose([transforms.Resize(
    224), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
ort_session = onnxruntime.InferenceSession(model_path)


def predict(path):
    img = Image.open(path)
    img_y = trans(img)
    img_y.unsqueeze_(0)
    ort_inputs = {ort_session.get_inputs()[0].name: img_y.cpu().numpy()}
    ort_outs = list(ort_session.run(None, ort_inputs)[0][0])
    return ort_outs.index(max(ort_outs))+1


st.title(f'手势识别模型测试~')

st.write(f" ")
st.sidebar.write(f"仅支持手指识别，仅支持jpg、png图片")
label = st.sidebar.radio("选择测试类型:", ["随机测试", "上传测试"])

if label == "随机测试":

    c00, c10 = st.columns([6, 2])
    
    path = random.choice(glob.glob("hand/val/*/*.jpg"))

    with c00:
      img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
      st.image(img, use_column_width=True)
      input_id = st.number_input(
        label="", help="点击加减刷新", min_value=1, step=1, max_value=10)
      

    with c10:
      st.write(f" ")
      st.write(f" ")
      st.write(f" ")
      st.write(f" ")
      st.write(f" ")
      st.write(f" ")
      st.write(f" ")
      st.write(f" ")
      st.write(f" ")
      st.write(f" ")
      st.write(f" ")
      st.write(f" ")
      st.write(f" ")
      st.write(f" ")
      num = predict(path)
      st.write(f"识别结果：这是{num}根手指")

else:

    f = st.file_uploader('', type=["jpg", "png"])
    if f is not None:
      file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
      opencv_image = cv2.imdecode(file_bytes, 1)
      path = "hand/"+str(time.time())+'.jpg'
      cv2.imwrite(path, opencv_image)

      img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
      st.image(img, use_column_width=True)
        
      num = predict(path)
      st.write(f"识别结果：这是{num}根手指")
