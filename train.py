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
def train_10():
    gt = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
    seq = np.array([[[0,1,0,0,0,0,0,0,0,0]],[[0,1,1,0,0,0,0,0,0,0]],[[1,1,1,0,0,0,0,0,0,0]],
    [[0,1,1,1,1,0,0,0,0,0]],[[1,1,1,1,1,0,0,0,0,0]],[[1,1,1,1,1,0,1,0,0,0]],
    [[1,1,1,1,1,0,1,1,0,0]],[[1,1,1,1,1,1,1,1,0,0]],[[1,1,1,1,1,0,1,1,1,1]],[[1,1,1,1,1,1,1,1,1,1]]])
    gt = torch.tensor(gt, dtype=torch.long)
    seq = torch.Tensor(seq)
    rnn = RNN(10, 10, 10)
    hidden = rnn.initHidden()
    rnn.zero_grad()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

    print (rand_sequence()[0].shape, gt.shape)
    print (rand_sequence()[1].shape, seq.shape)
    print (rand_sequence()[0][0], gt)
    solution_found=False
    for e in range(500):
        for j in range(10):
            if j==0:
                count=0
            output, hidden = rnn(seq[j], hidden)
            print(torch.argmax(output), gt[j])
            if torch.argmax(output)==gt[j][0]:
                count += 1
            if count==10:
                print ('solution found', e)
                solution_found=True
                # return
            if not solution_found:
                loss = criterion(output, gt[j])
                loss.backward(retain_graph=True)
                optimizer.step()
            else:
                pass
                # print(torch.argmax(output), gt[j])

    return output, loss.item()

def train_10_rev():
    gt = [[9],[8],[7],[6],[5],[4],[3],[2],[1],[0]]
    seq = np.array([[[1,1,1,1,1,1,1,1,1,1]],[[1,1,1,1,1,0,1,1,1,1]],[[1,1,1,1,1,1,1,1,0,0]],
    [[1,1,1,1,1,0,1,1,0,0]],[[1,1,1,1,1,0,1,0,0,0]],[[1,1,1,1,1,0,0,0,0,0]],
    [[0,1,1,1,1,0,0,0,0,0]],[[1,1,1,0,0,0,0,0,0,0]],[[0,1,1,0,0,0,0,0,0,0]],[[0,1,0,0,0,0,0,0,0,0]]])
    gt = torch.tensor(gt, dtype=torch.long)
    seq = torch.Tensor(seq)
    rnn = RNN(10, 10, 10)
    hidden = rnn.initHidden()
    rnn.zero_grad()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

    print (rand_sequence()[0].shape, gt.shape)
    print (rand_sequence()[1].shape, seq.shape)
    print (rand_sequence()[0][0], gt)
    solution_found=False
    for e in range(500):
        for j in range(10):
            if j==0:
                count=0
            output, hidden = rnn(seq[j], hidden)
            print(torch.argmax(output), gt[j])
            if torch.argmax(output)==gt[j][0]:
                count += 1
            if count==10:
                print ('solution found', e)
                solution_found=True
                return
            if not solution_found:
                loss = criterion(output, gt[j])
                loss.backward(retain_graph=True)
                optimizer.step()
            else:
                print(hidden, torch.argmax(output), gt[j])

    return output, loss.item()

def main():
    train_10()

if __name__ == "__main__":
    main()
