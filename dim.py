import torch

# 파일 경로
file_path = '/data/jhlee39/workspace/repos/Diff-Foley/dataset/features/frame_last_mask_trainset/DJMKm6KA9H0_000494_feature.pt'

# 텐서 로드
data = torch.load(file_path)

# 텐서인지 확인하고 shape 출력
if isinstance(data, torch.Tensor):
    print("Loaded tensor shape:", data.shape)
else:
    print("Loaded data is not a tensor. Type:", type(data)) 

# dict 형태일 경우
if isinstance(data, dict):
    for k, v in data.items():
        print(f"{k}: {type(v)}, shape: {v.shape if isinstance(v, torch.Tensor) else 'N/A'}")