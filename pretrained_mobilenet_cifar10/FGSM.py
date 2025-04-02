import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

class FGSM_attack:
    def __init__(self, model, x, target, eps):
        self.model = model
        self.eps = eps
        self.x = x
        self.target = target
        self.loss_CE = nn.CrossEntropyLoss()

        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.random_sample = []
        self.sample_num = 5

    def attack(self, x, y):
        x_adv = x.clone().detach().requires_grad_(True) # 원본 입력 복사

        outputs = self.model(x_adv) #기존 모델 예측
        if self.target is not None: # targeted
            loss = -self.loss_CE(outputs, y)  
        else:  # untargeted
            loss = self.loss_CE(outputs, y)  
       
        #이전 gradient를 초기화
        self.model.zero_grad()

        loss.backward()

        with torch.no_grad():
            if self.target is None: #untarget
                x_adv = x_adv + self.eps * x_adv.grad.sign()
            else: #target
                x_adv = x_adv - self.eps * x_adv.grad.sign()

            x_adv = torch.clamp(x_adv, min=0, max=1).detach() #0,1 범위로 클리핑

        return x_adv

    def run(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        all_attacks = [] # 공격 성공 리스트
        total_attacks_attempted = 0 # 공격 시도 횟수

        for data, label in self.x:
            data, label = data.to(device), label.to(device)
            output = self.model(data)
            init_pred = output.argmax(dim=1)

            if init_pred.item() != label.item(): #원본의 예측이 틀린경우 수행 x
                continue

            total_attacks_attempted += 1

            if self.target is not None:  # targeted
                adv_data = self.attack(data, torch.tensor([self.target], dtype=torch.long).to(device))
                adv_output = self.model(adv_data)
                adv_prob_after = F.softmax(adv_output, dim=1)[0, int(self.target)].item()
            else:  # untargeted
                adv_data = self.attack(data, label)
                adv_output = self.model(adv_data)
                adv_prob_after = F.softmax(adv_output, dim=1)[0, adv_output.argmax(dim=1).item()].item()
           
            adv_pred_label = adv_output.argmax(dim=1).item()
           

            if self.target is not None:
                if adv_pred_label != self.target:
                    continue  # targeted인데 target으로 안 바뀌었으면 실패
            else:
                if adv_pred_label == label.item():
                    continue  # untargeted인데 예측이 안 바뀌면 실패

            correct_prob_before = F.softmax(output, dim=1)[0, label.item()].item()

            all_attacks.append({
                "orig_label": label.item(),
                "adv_pred": adv_pred_label,
                "correct_prob_before": correct_prob_before,
                "adv_prob_after": adv_prob_after,
                "orig_image": data.squeeze().detach().cpu(),
                "adv_image": adv_data.squeeze().detach().cpu()
            })


        successful_attacks = len(all_attacks)

        attack_success_rate = (successful_attacks / total_attacks_attempted) * 100
        print(f"Success Attack rate: {attack_success_rate:.2f}% ({successful_attacks}/{total_attacks_attempted})")
        self.random_sample = random.sample(all_attacks, k=self.sample_num)


    def visualize(self):

        rows = self.sample_num
        cols = 2

        plt.figure(figsize=(cols * 4, rows * 3))
        cnt = 1

        for atk in self.random_sample:
            orig, adv = atk["orig_label"], atk["adv_pred"]
            orig_ex, adv_ex = atk["orig_image"], atk["adv_image"]
            prob_before, adv_after = atk["correct_prob_before"], atk["adv_prob_after"]

            for i, image in enumerate([orig_ex, adv_ex]):
                plt.subplot(rows, cols, cnt)
                cnt += 1
                plt.xticks([], [])
                plt.yticks([], [])

                if isinstance(image, torch.Tensor):
                    image = image.permute(1, 2, 0).numpy()

                orig_name = self.class_names[orig]
                adv_name = self.class_names[adv]

                if i == 0:
                    plt.title(f"Original: {orig_name}\nP={prob_before:.2f}", fontsize=10)
                else:
                    plt.title(f"Adv: {orig_name}→{adv_name}\nP={adv_after:.2f}", fontsize=10)

                plt.imshow(image)

        plt.tight_layout()
        if self.target is not None:
            plt.savefig("results/img/FGSM_t{}.jpg".format(self.target))
        else:
            plt.savefig("results/img/FGSM_ut.jpg")

        plt.show()