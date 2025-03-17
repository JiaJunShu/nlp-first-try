import json
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class RNN(nn.Module):
    def __init__(self, word_count,embedding_size, hidden_size, output_size):
        super(RNN, self).__init__()


        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(word_count, embedding_size)
        self.i2h = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(embedding_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden):
        word_vector = self.embedding(input_tensor)
        combined = torch.cat((word_vector, hidden),1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=device) # 1 表示 batch size 为 1
    
rnn=torch.load('rnn_model.pkl', weights_only=False).to(device)

categories=['考研考博','招聘信息']
with open('char_list.json','r') as f:
    char_list=json.load(f)
n_chars=len(char_list)+1  #加一个unk 未知字符

def title_to_tensor(title):
    tensor = torch.zeros(len(title),dtype=torch.long,device=device)
    for li, ch in enumerate(title):
        try:
            ind=char_list.index(ch)
        except:
            ind=n_chars-1
        tensor[li] = ind
    return tensor

def run_rnn(rnn,input_tensor):
    hidden = rnn.initHidden()
    for i in range(input_tensor.size()[0]):
        output, hidden = rnn(input_tensor[i].unsqueeze(dim=0), hidden)
    return output

def evaluate(rnn,input_tensor):
    with torch.no_grad():
        hidden=rnn.initHidden()
        output=run_rnn(rnn,input_tensor)
        return output
    
def get_category(title):
    title=title_to_tensor(title).to(device)
    output=evaluate(rnn,title)
    topn, topi = output.topk(1)
    return categories[topi.item()]

'''
if __name__=='__main__':
    while True:
        title=input('请输入标题：')
        if not title:
            break
        print(get_category(title))'
'''
if __name__=='__main__':
    import flask
    app=flask.Flask(__name__)
    @app.route('/')
    def index():
        title=flask.request.values.get('title')
        if title:
            return get_category(title)
        else:
            return "<form><input name='title' type='text'><input type='submit'></form>"
    app.run(host='0.0.0.0',port=12345)
        
