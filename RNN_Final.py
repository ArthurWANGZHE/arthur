import torch
import torch.nn as nn
import torch.optim as optim
import jieba
import random
import re
import zhon
from zhon import hanzi
import time
import pickle
from tqdm import tqdm

# 设置设备

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据
def load_file(x):
    a=time.time()
    with open("summary1.txt", "r") as f:  # 打开文本
        data = f.read()  # 读取文本
        # print(data)
        f.close()
    sentences = re.findall(zhon.hanzi.sentence, data)
    len_=len(sentences)
# print(len_)
    index=random.sample(range(len_),x)
    data=[]
    for i in index:
        data.append(sentences[i])
    b=time.time()
    print(len(data))
    print("读取数据时间：",b-a)
    return data

# 随机掩码和保存掩码部分的函数
def random_mask_sentence(sentence):
    words = jieba.lcut(sentence)
    mask_idx = random.randint(0, len(words) - 1)
    masked_word = words[mask_idx]
    words[mask_idx] = "[MASK]"
    masked_sentence = " ".join(words)
    return masked_sentence, masked_word


# 构建词汇表
def build_vocab(sentences):
    a=time.time()
    vocab = set()
    with tqdm(total=len(sentences)) as pbar:
        for sentence in sentences:
            words = jieba.lcut(sentence)
            for word in words:
                vocab.add(word)
            vocab.add("[MASK]")
            pbar.update(1)

    vocab = list(vocab)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    b=time.time()
    print("构建词汇表时间：",b-a)
    return vocab,word2idx,idx2word

# 保存词汇表
def save_vocab(vocab,x):
    with open("rnn_vocabs/vocab.pkl{}".format(x), "wb") as f:
        pickle.dump(vocab, f)
    print("Vocabulary saved.")


# 加载词汇表
def load_vocab(x):
    with open("rnn_vocabs/vocab.pkl{}".format(x), "rb") as f:
        loaded_vocab = pickle.load(f)
    print("Vocabulary loaded.")
    return loaded_vocab

# 定义模型
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output[:, -1, :])
        return output


# 将数据转换为模型所需的张量形式
def sentence_to_tensor(sentence, word2idx):
    words = jieba.lcut(sentence)
    tensor = torch.tensor([word2idx.get(word, word2idx["[MASK]"]) for word in words], dtype=torch.long)
    return tensor.unsqueeze(0).to(device)


# 加载模型并进行预测
def predict_masked_word(model, sentence, word2idx, idx2word):
    input_tensor = sentence_to_tensor(sentence, word2idx)
    output = model(input_tensor)
    _, predicted_idx = torch.max(output, 1)
    predicted_word = idx2word[predicted_idx.item()]
    return predicted_word


# 训练模型
def training(sentences, word2idx, idx2word,x):
    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            for sentence in sentences:
                model.zero_grad()
                input_tensor = sentence_to_tensor(sentence, word2idx)
                masked_sentence, masked_word = random_mask_sentence(sentence)
                target_tensor = torch.tensor([word2idx[masked_word]], dtype=torch.long).to(device)
                output = model(input_tensor)
                loss = criterion(output, target_tensor)
                loss.backward()
                optimizer.step()
                loss_=[]
                loss_.append(loss.item())
            pbar.set_description('这是第{}次训练,loss:{}'.format(epoch+1,loss.item()))
            pbar.update(1)


    # 保存模型
    torch.save(model.state_dict(), "rnn_models/rnn_model{}.pt".format(x))


    print("Model{} saved.".format(x))


# 超参数
embedding_dim = 150
hidden_dim = 100
num_layers = 3
learning_rate = 0.0002
num_epochs = 100

# 加载数据
sentences = load_file(5000)
# 构建词汇表
vocab,word2idx,idx2word = build_vocab(sentences)
# 保存词汇表
save_vocab(vocab,18)
# 加载词汇表
# loaded_vocab = load_vocab()
# 重新构建词语到索引的映射
# word2idx = {word: idx for idx, word in enumerate(loaded_vocab )}
# idx2word = {idx: word for idx, word in enumerate(loaded_vocab )}

# 初始化模型、损失函数和优化器
model = RNNModel(len(vocab), embedding_dim, hidden_dim, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#训练模型
training(sentences, word2idx, idx2word,18)

# 加载模型
model = RNNModel(len(vocab), embedding_dim, hidden_dim, num_layers).to(device)
model.load_state_dict(torch.load("rnn_models/rnn_model18.pt"))
model.train()  # 设置模型为训练模式
print("Model loaded.")

"""
model = RNNModel(len(vocab), embedding_dim, hidden_dim, num_layers).to(device)
model.load_state_dict(torch.load("rnn_model.pt"))
model.train()  # 设置模型为训练模式
print("Model loaded.")
"""

# 使用模型进行预测
sentence = "我想要放假，但是我还有很多作业要做"
masked_sentence = "我想要放假，但是我还有很多[MASK]要做"
masked_word = "作业"
predicted_word = predict_masked_word(model, masked_sentence, word2idx, idx2word)
print(f"Original sentence: {sentence}")
print(f"Masked sentence: {masked_sentence}")
print(f"Masked word: {masked_word}")
print(f"Predicted word: {predicted_word}\n")

sentence = "我们学校居然七月份军训，真是太酷了"
masked_sentence = "我们[MASK]居然七月份军训，真是太酷了"
masked_word = "学校"
predicted_word = predict_masked_word(model, masked_sentence, word2idx, idx2word)
print(f"Original sentence: {sentence}")
print(f"Masked sentence: {masked_sentence}")
print(f"Masked word: {masked_word}")
print(f"Predicted word: {predicted_word}\n")




