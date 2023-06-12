import torch
import torch.nn as nn
import torch.optim as optim
import jieba
import random

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 中文句子示例
sentences = ["我喜欢吃苹果。", "这是一个好的例子。"]


# 随机掩码和保存掩码部分的函数
def random_mask_sentence(sentence):
    words = jieba.lcut(sentence)
    mask_idx = random.randint(0, len(words) - 1)
    masked_word = words[mask_idx]
    words[mask_idx] = "[MASK]"
    masked_sentence = " ".join(words)
    return masked_sentence, masked_word


# 构建词汇表
vocab = set()
for sentence in sentences:
    words = jieba.lcut(sentence)
    for word in words:
        vocab.add(word)
    vocab.add("[MASK]")

vocab = list(vocab)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}

# 超参数
embedding_dim = 50
hidden_dim = 100
num_layers = 2
learning_rate = 0.001
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


# 初始化模型、损失函数和优化器
model = RNNModel(len(vocab), embedding_dim, hidden_dim, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
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

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    # 保存模型
torch.save(model.state_dict(), "rnn_model.pt")
print("Model saved.")

    # 加载模型
model = RNNModel(len(vocab), embedding_dim, hidden_dim, num_layers).to(device)
model.load_state_dict(torch.load("rnn_model.pt"))
model.train()  # 设置模型为训练模式
print("Model loaded.")

"""
model = RNNModel(len(vocab), embedding_dim, hidden_dim, num_layers).to(device)
model.load_state_dict(torch.load("rnn_model.pt"))
model.train()  # 设置模型为训练模式
print("Model loaded.")
"""

# 使用模型进行预测
sentence = "我喜欢吃[MASK]。"
masked_sentence, masked_word = random_mask_sentence(sentence)
predicted_word = predict_masked_word(model, masked_sentence, word2idx, idx2word)
print(f"Original sentence: {sentence}")
print(f"Masked sentence: {masked_sentence}")
print(f"Masked word: {masked_word}")
print(f"Predicted word: {predicted_word}\n")


