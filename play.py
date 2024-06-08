import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 플롯 함수 정의 (이미지 생성 및 저장 부분 제거)
def plot_latent_images(image_path, step=1, figsize=(30, 15)):  # 크기 조정
    image = Image.open(image_path)
    width, height = image.size
    image1 = image.crop((0, 0, width, height // 10))
    image2 = image.crop((0, height - (step + 1) * (height // 10), width, height - step * (height // 10)))

    figure, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(image1)
    ax[0].axis('off')
    ax[0].set_title('First Image')
    ax[1].imshow(image2)
    ax[1].axis('off')
    ax[1].set_title(f'Second Image')

    plt.tight_layout()
    st.pyplot(figure)

st.divider()
st.title('RNA')
st.write("Reminding Neighbors for Alzheimer's")
st.divider()
st.write("")

progress = st.progress(0)

step = st.session_state.get('step', 0)
image_path = './latent_space_images.png'  # 저장된 이미지 경로

if step <= 8:
    progress.progress((step + 1) / 9)
    
    plot_latent_images(image_path, step=step)
    st.write("")
    st.write("")
    chosen_image = st.radio("Choose your neighbor", ('First Image', 'Second Image'), index=0)

    # 답 확인 및 점수 측정
    correct_answer = 0  # 맞은 답의 수
    if chosen_image == 'First Image':
        correct_answer = 1

    # 이전 단계의 점수 불러오기
    prev_score = st.session_state.get('score', 0)

    # 현재 점수 계산
    current_score = prev_score + correct_answer

    # 단계별 점수 갱신
    st.session_state['score'] = current_score
    if step == 8:
        st.success(f"Final Score: {current_score-1}/9")

if st.button('Next Step'):
    step += 1
    st.session_state['step'] = step
    

st.divider()
