import torch
import torch.nn as nn
import torch.optim as optim

# 准备数据
sentences = ["I love [MASK] !", "This is a [MASK] example."]
masks = ["dog", "great"]

# 构建词汇表
vocab = set()
for sentence, mask in zip(sentences, masks):
    words = sentence.split()
    for word in words:
        vocab.add(word)
    vocab.add(mask)
vocab = list(vocab)
word2idx = {word: idx for idx, word in enumerate(vocab)}

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


# 初始化模型、损失函数和优化器
model = RNNModel(len(vocab), embedding_dim, hidden_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 将数据转换为模型所需的张量形式
def sentence_to_tensor(sentence, word2idx):
    words = sentence.split()
    tensor = torch.tensor([word2idx[word] for word in words], dtype=torch.long)
    return tensor.unsqueeze(0)


# 训练模型
for epoch in range(num_epochs):
    for sentence, mask in zip(sentences, masks):
        model.zero_grad()
        input_tensor = sentence_to_tensor(sentence, word2idx)
        target_tensor = torch.tensor([word2idx[mask]], dtype=torch.long)

        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

# 保存模型
torch.save(model.state_dict(), "rnn_model.pt")
print("Model saved.")


# 使用模型进行预测
def predict_masked_word(model, sentence, word2idx, idx2word):
    input_tensor = sentence_to_tensor(sentence, word2idx)
    output = model(input_tensor)
    _, predicted_idx = torch.max(output, 1)
    predicted_word = idx2word[predicted_idx.item()]
    return predicted_word


# 加载模型
model = RNNModel(len(vocab), embedding_dim, hidden_dim, num_layers)
model.load_state_dict(torch.load("rnn_model.pt"))
model.eval()
print("Model loaded.")

# 将索引到单词的映射
idx2word = {idx: word for word, idx in word2idx.items()}

# 示例预测
test_sentence = "This is a [MASK] example."
predicted_word = predict_masked_word(model, test_sentence, word2idx, idx2word)
print(f"Predicted word: {predicted_word}")