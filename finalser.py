import streamlit as st
import pickle
import librosa
import numpy as np
import pandas as pd
import moviepy.editor as mp
import os

model = pickle.load(open('final.pkl','rb'))

def save_audio(file):
    if file.size > 800000000000:
        return 1
    # if not os.path.exists("audio"):
    #     os.makedirs("audio")
    folder = "video"
    # clear the folder to avoid storage overload
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0

def main():
    st.title("Welcome to  Video Emotion Recognizer")
    st.markdown("#### Send the file and I will try my best to predict the emotion")
    video_file = st.file_uploader("Upload video", type=['mp4'])
    if video_file is not None:
        if not os.path.exists("video"):
            os.makedirs("video")
        path = os.path.join("video", video_file.name)
        if_save_audio = save_audio(video_file)
        if if_save_audio == 1:
            st.warning("file size is too large try anoother file")
        elif if_save_audio == 0:
            st.video(video_file,format='video/mp4')
            video1 = mp.VideoFileClip(path)
            video1.audio.write_audiofile('result.wav')
            data="result.wav"
            X, sample_rate = librosa.load(data)
            mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
            feature=mfccs.reshape(1,-1)
            pred = model.predict(feature)
            if pred == ['calm']:
                st.write("#### I predicted the emotion as CALM")
            elif pred == ['sad']:
                st.write("#### I predicted the emotion as SAD")
            elif pred == ['happy']:
                st.write("#### I predicted the emotion as HAPPY")
            elif pred == ['angry']:
                st.write("#### I predicted the emotion as ANGRY")
            elif pred == ['disgust']:
                st.write("#### I predicted the emotion as DISGUST")
            else:
                st.write("#### I predicted the emotion as SURPRISE")
        else:
            st.error("Unknown error")

if __name__ == '__main__':
    main()
