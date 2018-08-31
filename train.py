from generate_samples import *
from rnn import *
import torch

def train():
    gt, seq = rand_sequence()
    gt = torch.tensor(gt, dtype=torch.long)
    seq = torch.Tensor(seq)
    rnn = RNN(10, 10, 3)
    hidden = rnn.initHidden()
    rnn.zero_grad()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
    for e in range(20):
        for i in range(10):
            for j in range(3):
                output, hidden = rnn(seq[i][j], hidden)
                print(output, gt[i][j])
                loss = criterion(output, gt[i][j])
                loss.backward(retain_graph=True)
            optimizer.step()

    return output, loss.item()

def main():
    train()

if __name__ == "__main__":
    main()
