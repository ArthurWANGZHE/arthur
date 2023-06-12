import torch
import torch.nn as nn
import torch.optim as optim
from tokenizer import tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MaskedLanguageModelDataset(Dataset):
    def __init__(self, sentences, tokenizer):
        self.sentences = sentences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tokens = self.tokenizer.tokenize(sentence)
        masked_tokens, labels = self.mask_tokens(tokens)
        masked_indices = self.tokenizer.convert_tokens_to_ids(masked_tokens)
        label_indices = self.tokenizer.convert_tokens_to_ids(labels)
        return masked_indices, label_indices

    def mask_tokens(self, tokens):
        masked_tokens = []
        labels = []
        for token in tokens:
            if token == '[MASK]':
                masked_tokens.append(token)
                labels.append(token)
            else:
                masked_tokens.append('[MASK]')
                labels.append(token)
        return masked_tokens, labels

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_units):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.Transformer(d_model=embedding_dim, nhead=num_heads, num_encoder_layers=2,
                                          num_decoder_layers=2, dim_feedforward=hidden_units, dropout=0.1)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src):
        embedded = self.embedding(src)
        transformer_output = self.transformer(embedded)
        output = self.fc(transformer_output)
        return output

# 设置训练参数
vocab_size = tokenizer.vocab_size()
embedding_dim = 256
num_heads = 4
hidden_units = 512
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# 创建模型
model = TransformerModel(vocab_size, embedding_dim, num_heads, hidden_units)
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

sentences= ["ChatGPT是一个[MASK]的语言模型。",
            "ChatGPT是一个基于深度学习的对话机器人。",
            "ChatGPT是一个基于深度学习的对话机器人。"]
# 创建数据集和数据加载器
dataset = MaskedLanguageModelDataset(sentences, tokenizer)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练模型
for epoch in range(num_epochs):
    for batch in data_loader:
        batch = batch[0].to(device)

        # 前向传播
        output = model(batch[:, :-1])  # 预测输入序列的下一个词
        loss = criterion(output.view(-1, vocab_size), batch[:, 1:].contiguous().view(-1))  # 计算损失

        # 反向传播和梯度更新
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 加载模型
model = TransformerModel(vocab_size, embedding_dim, num_heads, hidden_units)

model.load_state_dict(torch.load('model.pth'))
model.to(device)
model.eval()

# 预测句子中的 [MASK] 部分
sentence = "ChatGPT是一个[MASK]的语言模型。"
tokens = tokenizer.tokenize(sentence)
masked_index = tokens.index('[MASK]')
token_ids = tokenizer.convert_tokens_to_ids(tokens)

input_tensor = torch.tensor([token_ids]).to(device)

with torch.no_grad():
    output = model(input_tensor)

predicted_token_id = torch.argmax(output[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_token_id])[0]

print(f"Predicted token: {predicted_token}")