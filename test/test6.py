import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import jieba
import random

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 中文句子示例
sentences = ["我喜欢吃苹果。", "这是一个好的例子。"]


# 随机掩码和保存掩码部分的函数
def random_mask_sentence(sentence):
    words = list(jieba.cut(sentence))
    mask_idx = random.randint(0, len(words) - 1)
    masked_word = words[mask_idx]
    words[mask_idx] = "[MASK]"
    masked_sentence = " ".join(words)
    return masked_sentence, masked_word


# 构建词汇表
vocab = set()
for sentence in sentences:
    words = list(jieba.cut(sentence))
    vocab.update(words)
vocab.add("[MASK]")
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}

# 超参数
embedding_dim = 50
hidden_dim = 100
num_layers = 2
learning_rate = 0.001
batch_size = 2
num_epochs = 100


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
        logits = self.fc(output)
        return logits


# 将中文句子转换为模型可以使用的训练样本
def sentence_to_tensor(sentence, word2idx):
    words = jieba.lcut(sentence)
    tensor = torch.tensor([word2idx.get(word, word2idx["[MASK]"]) for word in words], dtype=torch.long)
    return tensor.unsqueeze(0).to(device)



# 随机掩码和保存掩码部分的函数
def random_mask_sentence(sentence):
    words = list(jieba.cut(sentence))
    mask_idx = random.randint(0, len(words) - 1)
    masked_word = words[mask_idx]
    words[mask_idx] = "[MASK]"
    masked_sentence = " ".join(words)
    return masked_sentence, masked_word


# 加载模型并进行训练
model = RNNModel(len(vocab), embedding_dim, hidden_dim, num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_loss = 0.0
    for sentence in sentences:
        model.train()
        masked_sentence, masked_word = random_mask_sentence(sentence)
        input_tensor = sentence_to_tensor(masked_sentence,word2idx).unsqueeze(0).to(device)
        target_tensor = torch.tensor([word2idx[masked_word]], dtype=torch.long).unsqueeze(0).to(device)

        optimizer.zero_grad()
        logits = model(input_tensor)

        loss = nn.CrossEntropyLoss()(logits.view(-1, len(vocab)), target_tensor.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(sentences)}")

    # 保存模型
    torch.save(model.state_dict(), "rnn_model.pt")
    print("Model saved.")

    # 加载模型
    model = RNNModel(len(vocab), embedding_dim, hidden_dim, num_layers).to(device)
    model.load_state_dict(torch.load("rnn_model.pt"))
    model.eval()
    print("Model loaded.")

    # 使用模型进行预测
    for sentence in sentences:
        model.eval()
        masked_sentence, masked_word = random_mask_sentence(sentence)
        input_tensor = sentence_to_tensor(masked_sentence).unsqueeze(0).to(device)
        logits = model(input_tensor)
        predicted_word_id = torch.argmax(logits, dim=-1).item()
        predicted_word = idx2word[predicted_word_id]

        print(f"Original sentence: {sentence}")
        print(f"Masked sentence: {masked_sentence}")
        print(f"Masked word: {masked_word}")
        print(f"Predicted word: {predicted_word}\n")