import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import requests
import time
import torch
from tqdm import tqdm
from bs4 import BeautifulSoup
import random
import json

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
'''
# 定义第一个论坛版块的 ID
fid = 735
titles735 = []



# 遍历第一个论坛版块的所有页面
for pid in tqdm(range(9, 12)):
    try:
        # 发送 HTTP 请求获取页面内容
        r = requests.get('https://xintianya.net/forum-%d.htm'%pid)
        r.raise_for_status()  # 检查请求是否成功
        # 将页面内容保存到本地文件
        with open('raw_data/%d-%d.html' % (fid, pid), 'wb') as f:
            f.write(r.content)
        # 使用 BeautifulSoup 解析 HTML 内容
        b = BeautifulSoup(r.text, 'html.parser')
        # 找到包含帖子的表格
        table = b.find('table', id='forum_%d' % fid)
        if table:
            trs = table.find_all('tr')
            for tr in trs[1:]:
                title = tr.find_all('a')[1].text
                titles735.append(title)
        time.sleep(1)  # 等待 1 秒，避免过于频繁的请求
    except Exception as e:
        print(f"Error processing page {pid}: {e}")

# 将所有抓取到的标题保存到本地文本文件
with open('%d.txt' % 735, 'w', encoding='utf8') as f:
    for l in titles735:
        f.write(l + '\n')

# 定义第二个论坛版块的 ID
fid = 644
titles644 = []

# 遍历第二个论坛版块的所有页面
for pid in tqdm(range(1, 80)):
    try:
        # 发送 HTTP 请求获取页面内容
        r = requests.get('https://tieba.baidu.com/f?kw=%d&pn=%d' % (fid, pid))
        r.raise_for_status()  # 检查请求是否成功
        # 将页面内容保存到本地文件
        with open('raw_data/%d-%d.html' % (fid, pid), 'wb') as f:
            f.write(r.content)
        # 使用 BeautifulSoup 解析 HTML 内容
        b = BeautifulSoup(r.text, 'html.parser')
        # 找到包含帖子的表格
        table = b.find('table', id='forum_%d' % fid)
        if table:
            trs = table.find_all('tr')
            for tr in trs[1:]:
                title = tr.find_all('a')[1].text
                titles644.append(title)
        time.sleep(1)  # 等待 1 秒，避免过于频繁的请求
    except Exception as e:
        print(f"Error processing page {pid}: {e}")

# 将所有抓取到的标题保存到本地文本文件
with open('%d.txt' % fid, 'w', encoding='utf8') as f:
    for l in titles644:
        f.write(l + '\n')'
'''
academy_titles = []
job_titles = []
with open('academy_titles.txt',encoding='utf8') as f:
    for line in f:
        academy_titles.append(line.strip())
with open('job_titles.txt',encoding='utf8') as f:
    for line in f:
        job_titles.append(line.strip())

char_set=set()
for title in academy_titles:
    for c in title:
        char_set.add(c)
for title in job_titles:
    for c in title:
        char_set.add(c) 
print(len(char_set))

#这里只是介绍使用one-hot编码，实际上使用word2vec等方法

char_list = list(char_set)
n_chars=len(char_list)+1  #加一个unk 未知字符
'''
def title_to_tensor(title):
    tensor = torch.zeros(len(title), 1, n_chars)
    for li, letter in enumerate(title):
        tensor[li][0][char_list.index(letter)] = 1
    return tensor
'''
def title_to_tensor(title):
    tensor = torch.zeros(len(title),dtype=torch.long,device=device)
    for li, ch in enumerate(title):
        try:
            ind=char_list.index(ch)
        except:
            ind=n_chars-1
        tensor[li] = ind
    return tensor

#字符型RNN
class RNN(torch.nn.Module):
    #word_count词表大小 embedding_size词向量维度 hidden_size隐藏层维度 output_size输出维度
    def __init__(self, word_count,embedding_size, hidden_size, output_size):
        super(RNN, self).__init__()#调用父类的构造函数，初始化模型
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(word_count, embedding_size)
        self.i2h = torch.nn.Linear(embedding_size + hidden_size, hidden_size)
        self.i2o = torch.nn.Linear(embedding_size + hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden):
        word_vector = self.embedding(input_tensor)
        combined = torch.cat((word_vector, hidden),1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=device) # 1 表示 batch size 为 1
    
embedding_size=200
n_hidden=128
n_categories=2
rnn=RNN(n_chars,embedding_size,n_hidden,n_categories).to(device)

input_tensor = title_to_tensor(academy_titles[0]).to(device)
print('input_tensor:\n',input_tensor)
hidden = rnn.initHidden()
output, next_hidden = rnn(input_tensor[0].unsqueeze(dim=0), hidden)
print('output:\n',output)
print('hidden:\n',hidden)
print('size of hidden:\n',hidden.size())


def run_rnn(rnn,input_tensor):
    hidden = rnn.initHidden()
    for i in range(input_tensor.size()[0]):
        output, hidden = rnn(input_tensor[i].unsqueeze(dim=0), hidden)
    return output

all_data=[]
categories=["考研考博","招聘信息"]
for l in academy_titles:
    all_data.append((title_to_tensor(l),torch.tensor([0],dtype=torch.long, device=device)))
for l in job_titles:
    all_data.append((title_to_tensor(l),torch.tensor([1],dtype=torch.long, device=device)))
           
random.shuffle(all_data)
data_len=len(all_data)
split_radio=0.7
train_data=all_data[:int(data_len*split_radio)]
test_data=all_data[int(data_len*split_radio):]
print("train data size:",len(train_data))
print("test data size:",len(test_data))

def train(rnn,criterion,input_tensor,category_tensor):
    rnn.zero_grad()
    output= run_rnn(rnn,input_tensor)
    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data,alpha=-learning_rate)
    return output, loss.item()

def evalute(rnn,input_tensor):
    with torch.no_grad():
        hidden=rnn.initHidden()
        output=run_rnn(rnn,input_tensor)
        return output
    
epoch=10
learning_rate=0.005
criterion = torch.nn.NLLLoss()
loss_sum=0
all_losses=[]
plot_every=100
for e in range(epoch):
    for ind,(title_tensor,label) in enumerate(tqdm(train_data)):
        output,loss=train(rnn,criterion,title_tensor,label)
        loss_sum+=loss
        if ind%plot_every==0:
            all_losses.append(loss_sum/plot_every)
            loss_sum=0
    c=0
    for title_tensor,category in tqdm(test_data):
        output=evalute(rnn,title_tensor)
        topn,topi=output.topk(1)
        if topi.item()==category[0].item():
            c+=1
    print('accuracy:',c/len(test_data))

plt.figure(figsize=(10,7))
plt.ylabel('Average Loss')
plt.plot(all_losses[1:])
plt.show()

torch.save(rnn,'rnn_model.pkl')

rnn = torch.load('rnn_model.pkl', weights_only=False).to(device)


with open('char_list.json','w') as f:
    json.dump(char_list,f)

with open('char_list.json','r') as f:
    char_list=json.load(f)

def get_category(title):
    title=title_to_tensor(title).to(device)
    output=evalute(rnn,title)
    topn,topi=output.topk(1)
    return categories[topi.item()]

def print_test(title):
    print('%s\t%s'%(title,get_category(title)))

print_test('2022考研数学真题及答案')
print_test('最新招聘信息')
print_test('北大实验室博士')
print_test('校招offer比较')