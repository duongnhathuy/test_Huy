import streamlit as st
import base64
def intro():
    import streamlit as st
    from PIL import Image
    image = Image.open('1. LOGO HCMUTE.png')
    st.image(image, caption=None, width=600, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.write("# Welcome to my Final-Project of Artificial Intelligence")
    st.header(":red[Research: Estimating human skeletons from body motion recognition using image processing and pattern recognition]")
    st.subheader(":orange[Student: Duong Nhat Huy]")
    st.subheader(":orange[ID: 20146125]")
    st.subheader(':orange[Lecturer: Assoc. Prof. Nguyen Truong Thinh]')
    
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('background.png') 

def nhanDienThongQuaVideo():
    import streamlit as st
    import tempfile
    import cv2
    from ctypes import sizeof
    import numpy as np
    import mediapipe as mp
    import os
    import h5py
    import tensorflow as tf
    import threading
    import concurrent.futures
    import time
    import datetime
    import pandas as pd
    label = "Warmup...."
    n_time_steps = 5
    lm_list = []

    action = {}
    pathSave = 'DATA'
    threadsold = 0.8

    for dirpath, dirnames, filenames in os.walk(pathSave):
        count=0
        for dirname in dirnames:
            print(dirname)
            action[f"{count}"] = dirname
            count+=1
            

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    with h5py.File("AImodel.h5", "r") as f:
        model = tf.keras.models.load_model(f, compile=False)

    # cap = cv2.VideoCapture(0)

    def make_landmark_timestep(results):
        c_lm = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            c_lm.append(lm.visibility)
        return c_lm


    def draw_landmark_on_image(mpDraw, results, img):
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            #print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return img


    def draw_class_on_image(label, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = (0, 255, 0)
        thickness = 2
        lineType = 2
        cv2.putText(img, label,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        return img


    def detect(model, lm_list):
        global label
        lm_list = np.array(lm_list)
        lm_list = np.expand_dims(lm_list, axis=0)
        #print(lm_list.shape)
        results = model.predict(lm_list)
        # print(results[0][0])
        
        if max(results[0]) > threadsold:
            max_index = results[0].argmax()
            label = action[f"{max_index}"]
            print(label)
            print(max_index)
        else:
            label = "Warmup...."
        return label


    i = 0
    warmup_frames = 60

    back = st.file_uploader("Choose video", type=["mp4", "mov"])
    font = cv2.FONT_HERSHEY_PLAIN
    color = (0, 128, 0)
    if back:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(back.read())
        # st.video(tfile.name)
        print(tfile.name)
        cap = cv2.VideoCapture("./video/" + back.name)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0

        now = datetime.datetime.now()
        current_time = now.strftime("%H%M%S")
        newFilePath = "./NewData/"+ current_time +"/"

        while True:
            success, img = cap.read()
            
            if success == False:
                print("Error: could not load image")
                st.write("End video")
                return
            height, width, channels = img.shape
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            i = i + 1
            if i > warmup_frames:
                #print("Start detect....")

                if results.pose_landmarks:
                    c_lm = make_landmark_timestep(results)

                    lm_list.append(c_lm)
                    if len(lm_list) == n_time_steps:
                        # predict
                        label = detect(model, lm_list)
                        lm_list = []

                    img = draw_landmark_on_image(mpDraw, results, img)
            img = draw_class_on_image(label, img)
            output.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")
def nhanDienThongQuaWebcam():
    import streamlit as st
    import tempfile
    import cv2
    from ctypes import sizeof
    import numpy as np
    import mediapipe as mp
    import os
    import h5py
    import tensorflow as tf
    import threading
    import concurrent.futures
    import time
    import datetime
    import pandas as pd
    label = "Warmup...."
    n_time_steps = 5
    lm_list = []

    action = {}
    pathSave = 'DATA'
    threadsold = 0.8

    for dirpath, dirnames, filenames in os.walk(pathSave):
        count=0
        for dirname in dirnames:
            print(dirname)
            action[f"{count}"] = dirname
            count+=1
            

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    with h5py.File("AImodel.h5", "r") as f:
        model = tf.keras.models.load_model(f, compile=False)

    cap = cv2.VideoCapture(0)

    def make_landmark_timestep(results):
        c_lm = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            c_lm.append(lm.visibility)
        return c_lm


    def draw_landmark_on_image(mpDraw, results, img):
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            #print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return img


    def draw_class_on_image(label, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = (0, 255, 0)
        thickness = 2
        lineType = 2
        cv2.putText(img, label,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        return img


    def detect(model, lm_list):
        global label
        lm_list = np.array(lm_list)
        lm_list = np.expand_dims(lm_list, axis=0)
        #print(lm_list.shape)
        results = model.predict(lm_list)
        # print(results[0][0])
        
        if max(results[0]) > threadsold:
            max_index = results[0].argmax()
            label = action[f"{max_index}"]
            print(label)
            print(max_index)
        else:
            label = "Warmup...."
        return label


    i = 0
    warmup_frames = 60

    # back = st.file_uploader("Choose background video", type=["mp4", "mov"])
    font = cv2.FONT_HERSHEY_PLAIN
    color = (0, 128, 0)
    # if back:
        # tfile = tempfile.NamedTemporaryFile(delete=False)
        # tfile.write(back.read())
        # st.video(tfile.name)
        # print(tfile.name)
        # cap = cv2.VideoCapture("./video/" + back.name)
    custom_size = st.sidebar.checkbox("Custom frame size")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if custom_size:
        width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
        height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

    fps = 0
    st1, st2, st3 = st.columns(3)
    with st1:
        st.markdown("## Height")
        st1_text = st.markdown(f"{height}")
    with st2:
        st.markdown("## Width")
        st2_text = st.markdown(f"{width}")
    with st3:
        st.markdown("## FPS")
        st3_text = st.markdown(f"{fps}")

    st.markdown("---")
    output = st.empty()
    prev_time = 0
    curr_time = 0

    now = datetime.datetime.now()
    current_time = now.strftime("%H%M%S")
    newFilePath = "./NewData/"+ current_time +"/"

    while True:
        success, img = cap.read()
        
        if success == False:
            print("Error: could not load image")
            st.write("Hết video")
            return
        height, width, channels = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        i = i + 1
        if i > warmup_frames:
            #print("Start detect....")

            if results.pose_landmarks:
                c_lm = make_landmark_timestep(results)

                lm_list.append(c_lm)
                if len(lm_list) == n_time_steps:
                    # predict
                    label = detect(model, lm_list)
                    lm_list = []

                img = draw_landmark_on_image(mpDraw, results, img)
        img = draw_class_on_image(label, img)
        output.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        st1_text.markdown(f"**{height}**")
        st2_text.markdown(f"**{width}**")
        st3_text.markdown(f"**{fps:.2f}**")
page_names_to_funcs = {
    "INTRODUCE": intro,
    "PROCESSING BY WEBCAM": nhanDienThongQuaWebcam,
    "PROCESSING BY VIDEO":nhanDienThongQuaVideo
}

demo_name = st.sidebar.selectbox("CHOOSE OPTIONS", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()