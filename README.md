# Assignment 1: Adversarial Attacks (FGSM & PGD)

본 과제에서는 CIFAR-10과 MNIST 데이터셋을 대상으로 FGSM 및 PGD 공격을 수행하고, targeted 및 untargeted 방식으로 실험한 결과를 비교합니다.

## 💡 사용 방법

MNIST에 대한 실험을 위해서는 `VGGNet_mnist` 모델을 사용합니다. 이 모델은 학습된 가중치를 필요로 하므로 다음과 같은 순서로 진행해야 합니다:

1. `VGGNet_train.py` 실행하여 모델 학습  
2. `test.py` 실행하여 FGSM / PGD 공격 수행 및 결과 확인

---

## 📊 결과 (Results)

### ✅ CIFAR-10

#### 🔹 FGSM - Untargeted
![FGSM_untarget_CIFAR10](https://github.com/user-attachments/assets/3c5954b2-3ed2-446b-bbea-d14634a77c24)

#### 🔹 FGSM - Targeted
![FGSM_target_CIFAR10](https://github.com/user-attachments/assets/949fc846-10e5-4e0e-9054-21318df1c85e)

#### 🔹 PGD - Untargeted
![PGD_untarget_CIFAR10](https://github.com/user-attachments/assets/79108efd-3b85-4b1c-9a5f-9dc2d161f7b5)

#### 🔹 PGD - Targeted
![PGD_target_CIFAR10](https://github.com/user-attachments/assets/1e1ceae0-e2c0-46eb-860f-3841c01651bb)

---

### ✅ MNIST

#### 🔹 FGSM - Untargeted
![FGSM_untarget_MNIST](https://github.com/user-attachments/assets/feea9aeb-99ae-4593-8640-17289f1075f1)

#### 🔹 FGSM - Targeted
![FGSM_target_MNIST](https://github.com/user-attachments/assets/450224cc-6a79-4790-8122-872d44466926)

#### 🔹 PGD - Untargeted
![PGD_untarget_MNIST](https://github.com/user-attachments/assets/16865b88-e06c-43b8-b3eb-48ef5ea99e99)

#### 🔹 PGD - Targeted
![PGD_target_MNIST](https://github.com/user-attachments/assets/bf6820bc-7fb6-4f54-abde-d76b554324d5)

---
