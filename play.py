import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 클래스 정의
class VAE(nn.Module):
    def __init__(self, input_dim=256*256*3, hidden_dim=400, latent_dim=1024, device=device):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )

    def encode(self, x):
        x = self.encoder(x)
        mean = self.mean_layer(x) 
        logvar = self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var

# 모델 인스턴스 생성 및 파라미터 로드
model = VAE()
model.load_state_dict(torch.load('/Users/mac/Downloads/data_CeleA.pth', map_location=torch.device('cpu')))
model.to(device)
model.eval()

# 플롯 함수 정의
def plot_latent_images(model, scale=15.0, digit_size=256, figsize=15, step=1):
    # display a single row of 2D manifold of digits
    figure, ax = plt.subplots(1, 2, figsize=(figsize, figsize // 2))

    # construct a grid
    grid_y = np.linspace(-scale, scale, 10)

    image_indexes = [0, 9 - step]
    for i, idx in enumerate(image_indexes):
        z_sample = torch.tensor([[-scale, grid_y[idx]]], dtype=torch.float).to(device)
        x_decoded = model.decode(z_sample)
        digit = x_decoded[0].detach().cpu().numpy().reshape(3, digit_size, digit_size)
        digit = digit.transpose(1, 2, 0)
        ax[i].imshow(digit)
        ax[i].axis('off')
    ax[0].set_title('First Image')
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

if step <= 8:
    progress.progress((step + 1) / 9)
    
    plot_latent_images(model, step=step)
    
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
   

if st.button(f'Next Step'):
    step += 1
    st.session_state['step'] = step
