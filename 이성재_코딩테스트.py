#!/usr/bin/env python
# coding: utf-8

# ### evaluate 라이브러리 설치

# In[1]:


get_ipython().run_line_magic('pip', 'install evaluate')


# ### 라이브러리 임포트

# In[2]:


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import evaluate
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tqdm.auto import tqdm


# ### 기본 config 정의 및 SEED 설정

# In[3]:


# 기본적인 config info 선언 및 초기화, 시드 설정

device = "cuda" if torch.cuda.is_available() else "mps"
model_name = "Qwen/Qwen2.5-0.5B"
precision = "bf16"
optim = "AdamW"
lr = 5e-5
scheduler = None
seed = 42

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(seed)


# # 데이터셋 전처리
# ### **DSBA 랩실 코딩 테스트 조건**
# 
# ``` 1. Answer가 미응답이면 필터링 ```\
# ``` 2. 훈련, 검증, 시험 데이터셋 각각 20000, 300, 500개 샘플링 ```
# 

# In[4]:


# 훈련 데이터셋 불러오기
split = "train"
train_size = 20000
train = load_dataset("rajpurkar/squad_v2", split=split)

# 답변이 비어있는 데이터 제외
train = train.filter(lambda x:len(x["answers"]["text"]) > 0)

# trainset 샘플링
train_idxs = np.random.choice(len(train), train_size, replace=False)
train = train.select(train_idxs)


# 검증 데이터셋 불러오기
split = "validation"
validation_size = 800 # valid 300 + test 500
valid_size = 300

valid = load_dataset("rajpurkar/squad_v2", split=split)

# 답변이 비어있는 데이터 제외
valid = valid.filter(lambda x:len(x["answers"]["text"]) > 0)

valid_idxs = np.random.choice(len(valid), validation_size, replace=False)

# valid_idxs가 이미 random하게 선택됨.
# 따라서 valid_idxs를 [:300]과 [300:]로 나누기만 해도 random하게 뽑힘
test_idxs = valid_idxs[valid_size:]
valid_idxs = valid_idxs[:valid_size]

test = valid.select(test_idxs)
valid = valid.select(valid_idxs)


# ### 모델, 토크나이저 로드 및 optimizer 선언
# ### 뒤에 배치 사이즈를 정할 때 이번 셀에서 초기화한 인스턴스를 쓰게 됩니다.

# In[5]:


tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype = torch.bfloat16
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


# ### tokenizer와 모델의 입출력을 확인하기 위한 셀
# ```출력에 공백이 있는 것을 확인```

# In[6]:


# out 형식 체크
title, question, context = ("What's Eating Gilbert Grape",
    "Who played Arnie Grape in the film?",
    """What's Eating Gilbert Grape is a 1993 American drama film directed by Lasse Hallström.
Based on the novel by Peter Hedges, it is set in the small town of Endora, Iowa.
Johnny Depp stars as Gilbert, a young man caring for his younger brother Arnie,
played by Leonardo DiCaprio, and their mother.
DiCaprio received an Academy Award nomination for Best Supporting Actor for the role.""")

inst_prompt = f"Please give me an answer to the question based on the following context: Title: {title} Question: {question} Context: {context} Answer:"
enc = tokenizer(inst_prompt, return_tensors="pt")

with torch.no_grad():
    out_idxs = model.generate(**enc, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)

print(tokenizer.decode(out_idxs, skip_special_tokens=True)[0].split(":")[-1])


# ### get_prompt: title, question, context를 입력으로 받고 포맷에 맞게 return해주는 함수
# ### 입력 텍스트의 최대 길이로 자를 때 Answer가 잘리지 않도록 프롬프트를 나뉜 채로 return해줍니다.
# ### tokenizer 후의 길이를 기준으로 max length를 설정해줬습니다.
# ### 다른 통계적인 방법들보다도 256, 512와 같은 2의 거듭제곱 중에서 어느정도 범위를 커버리지하는지를 기준으로 판단하여 정했습니다.

# In[ ]:


def get_prompt(title, question, context):

    return f"Please give me an answer to the question based on the following context: Title: {title} Question: {question} Context: {context}", " Answer:"
            # max_length를 넘을 때 context를 슬라이싱하기 위해 " Answer:"를 분리해서 return

prompts = [get_prompt(train[i]["title"], train[i]["question"], train[i]["context"]) for i in range(len(train))]

enc_prompts = [tokenizer(prompt1 + prompt2) for prompt1, prompt2 in prompts]

lengths = []
for i in range(len(enc_prompts)):
    lengths.append(len(enc_prompts[i]["input_ids"]))

inst_l = sorted(lengths)

print(f"최소 길이: {min(lengths)}\n평균 길이: {sum(lengths) / len(lengths)}\n최대 길이: {max(lengths)}")
print(f"중앙값: {sum(inst_l[int(len(lengths)/2)-1:int(len(lengths)/2)+1])/2}")
print()

print(f"길이가 256을 넘는 프롬프트 수: {sum(l > 256 for l in lengths)} (비율:{sum(l > 256 for l in lengths)/len(lengths)*100})") # 3542 << 너무 많음
print(f"길이가 512를 넘는 프롬프트 수: {sum(l > 512 for l in lengths)} (비율:{sum(l > 512 for l in lengths)/len(lengths)*100})") # 57 << 512면 대부분을 처리할 수 있음

# max length는 512로 결정
max_len = 512


# ### 나눠서 context만 자르고 붙이는 것이 데이터 무결성을 해치는지 확인하였습니다.
# ```prompt1(title, question, context) + " Answer:" --- (1)```\
# \
# ``` prompt --- (2)```
# ### (1)과 (2)를 tokenize했을 때 다른 값이 나오면 문제가 되기 때문에 이를 비교했습니다.

# In[8]:


# 나눠서 context만 자르는 게 데이터 무결성을 해치지 않는지 확인 (Dataset에서 context를 자를 계획)
mismatch = 0
for ex in train:
    a, b = get_prompt(ex["title"], ex["question"], ex["context"])
    if tokenizer(a + b)["input_ids"] != tokenizer(a)["input_ids"] + tokenizer(b)["input_ids"]:
        mismatch += 1
print(f"데이터 무결성: {mismatch==0}")


# ### Dataset 클래스
# Dataset에서 ids를 모델 입장에서 input과 output을 return하도록 구현했습니다.\
# self.input에 데이터를 저장할 때 context만 슬라이싱해주는 max_len_slice 함수를 사용합니다.

# In[9]:


def max_len_slice(prompt1, prompt2, max_len):
    if len(prompt1) + len(prompt2) <= max_len:
        return prompt1 + prompt2
    return prompt1[:max_len - len(prompt2)] + prompt2

class idxset(Dataset):
    def __init__(self, data, max_len):
        super().__init__()
        self.length = len(data)

        prompts = [ get_prompt(data[i]["title"],
                                data[i]["question"],
                                data[i]["context"]) for i in range(self.length) ]

        self.input = [max_len_slice(tokenizer(prompt1)['input_ids'], tokenizer(prompt2)['input_ids'], max_len) for prompt1, prompt2 in prompts]



        self.output = [tokenizer(" " + data[i]["answers"]["text"][0])['input_ids'] + [tokenizer.eos_token_id] for i in range(len(data))]

    def __len__(self):
        return self.length

    def __getitem__(self, id):
        return (self.input[id], self.output[id])


# ### collate 함수
# ```
# DataLoader에서 collate_fn을 정의하지 않으면 기본 collate_fn 사용 (길이를 맞추는 부분이 없음)
# 모델은 배치 내에서는 데이터의 길이가 같아야 오류가 안 남
# 그러나 자연어 모델의 데이터는 데이터가 다르게 주어짐
# 따라서 데이터 길이 조정 + attention 마스크와 train과 eval/test에 따라 다른 방식으로 작동할 필요
# ```

# In[10]:


def collate_train(batch):
    input_ids = [inp + output for inp, output in batch]
    labels = [[-100]*len(inp) + output for inp, output in batch] # 학습시 -100이면 loss 함수에서 제외, output 부분은 그대로
    attn_mask = [[1]*len(inp) + [1]*len(output) for inp, output in batch] # 0이면 마스킹
    max_len = max([len(i) for i in input_ids])

    padded_input, padded_mask, padded_label = [], [], []
    for i, l, m in zip(input_ids, labels, attn_mask):
        rest_len = max_len - len(i)
        padded_input.append(i + [tokenizer.pad_token_id]*rest_len)
        padded_label.append(l + [-100]*rest_len)
        padded_mask.append(m + [0]*rest_len)

    return {"input_ids": torch.tensor(padded_input),
            "attention_mask": torch.tensor(padded_mask),
            "labels": torch.tensor(padded_label)}

def collate_eval(batch):
    input_ids = [inp for inp, _ in batch]
    labels = [tokenizer.decode(output, skip_special_tokens=True).strip() for _, output in batch] # 목표 토큰
    attn_mask = [[1]*len(inp) for inp, _ in batch]
    max_len = max([len(i) for i, _ in batch])

    padded_input, padded_mask = [], []
    for i, m in zip(input_ids, attn_mask):
        rest_len = max_len - len(i)
        padded_input.append([tokenizer.pad_token_id]*rest_len + i)
        padded_mask.append([0]*rest_len + m)

    return {"input_ids": torch.tensor(padded_input),
            "attention_mask": torch.tensor(padded_mask)}, labels


# In[11]:


trainset = idxset(train, max_len)
validset = idxset(valid, max_len)
testset = idxset(test, max_len)


# ### 배치 사이즈 확인 셀
# ```
# 본 학습 이전에만 실행하고, 본 학습시에는 건너뛰었습니다.
# ```

# In[12]:


# 배치 사이즈를 보기 위한 셀: 본 학습시 제외하고 실햄 (OOM 위험)
batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

model.train()
for batch_size in batch_sizes:
    print(f"batch size: {batch_size}")
    train_loader = DataLoader(trainset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_train)
    step = 0
    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        if step == 20: # 적절한 batch_size를 보기 위해 전체 데이터셋을 돌지 않고 20배치만 확인.
            break


# ### 모델 저장을 위한 드라이브 마운트

# In[12]:


from google.colab import drive
drive.mount('/content/drive')


# ### 본 학습 코드

# In[13]:


from datetime import datetime
import wandb

batch_size = 4

# 본 학습 진행 이전 wandb
wandb.login()
wandb.init(project="dsba-nlp-coding-test", name="train",   # 실험 이름
           config={"batch_size": batch_size,
                    "max_len": max_len})

set_seed(seed)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

traintime = datetime.now().strftime("%Y%m%d%H")
out_path = f"/content/drive/MyDrive/dsba_coding_test/weights/{traintime}.pth"
model.train()

generator = torch.Generator().manual_seed(seed)

train_loader = DataLoader(trainset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=collate_train,
                        generator=generator)
step = 0



step_losses = []

while step < 10000:

    for batch in tqdm(train_loader):
        batch = {k:v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            out = model(**batch)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step_losses.append(out.loss.item())
        step += 1
        if step % 100 ==0:
            train_loss = sum(step_losses[-100:]) / 100
            print(f"average loss of {step-99}~{step}: {train_loss}")
            wandb.log({ "train_loss": sum(step_losses[-100:]) / 100 },
                        step=step)
        if step >= 10000:
            break

wandb.finish()


# ### 모델 저장 및 로드

# In[14]:


# 모델 저장
torch.save(model.state_dict(), out_path)


# In[15]:


# 모델 불러오기
checkpoint = torch.load(out_path)

model.load_state_dict(checkpoint)


# ### max_new tokens 결정
# ```
# eval을 할 때 max_new_tokens를 정하지 않으면 200토큰까지 생성
# 따라서 적절한 max_new_tokens를 정하기 위해 Answers의 최대 길이를 확인했습니다.
# ```

# In[16]:


# max_new_tokens 인자를 정하기 위한 측정 (train set 기준으로 측정)
train_lengths = [len(train_data) for train_data in trainset.output]
print(f"학습셋 Answer 최대길이: {max(train_lengths)}")


# ### EM을 계산하는 함수
# ```
# 모델 평가가 수 차례 진행되기 때문에
# 더 간략한 코드를 위해 함수로 관리했습니다.
# ```

# In[17]:


import evaluate

def compute_em(model, metric_fct, dataset, k, generation_method, batch_size):
    set_seed(seed)
    model.eval()
    eval_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_eval)

    predictions, references = [], []
    with torch.no_grad():
        for inputs, refs in eval_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if generation_method == "beam_search":
                preds = model.generate(**inputs, max_new_tokens=96, # train answer max token이 76이라서 그보다 큰 96
                                        num_beams=k, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id)
            else:
                preds = model.generate(**inputs, max_new_tokens=96, # train answer max token이 76이라서 그보다 큰 96
                                        top_k=k, do_sample=True, top_p=1.0, temperature=1.0,
                                        pad_token_id=tokenizer.pad_token_id)
            predictions = predictions + [pred.strip() for pred in tokenizer.batch_decode(preds[:,inputs["input_ids"].shape[1]:], skip_special_tokens=True)]
            references = references + refs

    result = metric_fct.compute(predictions = predictions,
                                            references = references,
                                            ignore_case=True,
                                            ignore_punctuation=True)

    return result


# ### beam search vs top k 성능 비교 코드

# In[18]:


import wandb
wandb.init(project="dsba-nlp-coding-test", name="beam_search vs top_k",
           config={"max_new_tokens": 96, "val_size": len(validset)})

metric_fct = evaluate.load("exact_match")

em_beam_search = compute_em(model, metric_fct, validset, 4, "beam_search", 16)
em_top_k = compute_em(model, metric_fct, validset, 10, "top_k", 16)

print(f"beam search 결과: {em_beam_search['exact_match']}")
print(f"top k 결과: {em_top_k['exact_match']}")

wandb.log({"val/beam_search_4": em_beam_search['exact_match']})
wandb.log({"val/top_k_10": em_top_k['exact_match']})
wandb.finish()


# ### beam search, top-K 샘플링 성능 비교표
# |Exact Match|검증용|
# |------|---|
# |Beam Search|0.51333|
# |Top-K sampling|0.49333|

# ### beam search 하이퍼 파라미터 튜닝
# ```
# beam search가 top-k 샘플링 방식보다 더 좋은 성능을 보였기에 beam search의 하이퍼파라미터 튜닝
# 
# num_beams가 커지면 OOM이 발생하기 때문에 num_beams에 따라 batch size를 조절하도록 했습니다.
# ```

# In[19]:


from collections import defaultdict
wandb.init(project="dsba-nlp-coding-test", name="hyperparameter_tuning",
           config={"max_new_tokens": 96, "val_size": len(validset)})

num_beams_list = [1, 2, 10, 16]

results = defaultdict()
results[4] = em_beam_search['exact_match']

best_num_beams = 4
m = em_beam_search['exact_match']
for num_beams in num_beams_list:
    result = compute_em(model, metric_fct, validset, num_beams, "beam_search", 4 if num_beams < 10 else 2)['exact_match']
    results[num_beams] = result
    if result > m:
       best_num_beams = num_beams
       m = result
print("==== eval results ====")

for k, v in sorted(results.items()):
    print(f"{k}: {v}")
    wandb.log({f"num_beams_{k}": v})
print(f"best num beams: {best_num_beams} {m}")
wandb.finish()


# ### 빔 서치 eval 성능 비교 결과표
# |Exact Match|검증용|
# |------|---|
# |num_beams < 10(=1)|0.53|
# |num_beams < 10(=2)|0.53|
# |num_beams < 10(=4)|0.51333|
# |num_beams = 10|0.5|
# |num_beams > 10(=16)|0.49667|

# ### 시험용 데이터셋 최종 결과
# ```
# 시험용 데이터셋에서 best_num_beams로 평가하도록 구현했습니다.
# ```

# In[20]:


wandb.init(project="dsba-nlp-coding-test", name="final_test",
           config={"max_new_tokens": 96, "test_size": len(testset),
           "best_num_beams": best_num_beams})
em_final = compute_em(model, metric_fct, testset, best_num_beams, "beam_search", 4 if best_num_beams < 10 else 2)
print(f"시험용 데이터셋 결과: {em_final['exact_match']}")

wandb.log({"em_test": em_final['exact_match'], "test_num_beams": best_num_beams})
wandb.finish()


# ### 최종 시험 데이터 성능
# |Exact Match|시험용|
# |------|---|
# |num_beams = 1|0.574|
# 

# 
