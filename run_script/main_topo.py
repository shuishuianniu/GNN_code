import sys
sys.path.append("..")
from dataprocess.data import prepare
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from config import params,setting,args
from utils.utils import gnn_model
from utils.utils import accuracy


if __name__ == "__main__":

    cuda = torch.cuda.is_available()
    NA,features,topofea,label,idx_train,idx_test = prepare(params,args)

    model,MODEL_NAME = gnn_model(args.model,params,setting)
    print("MODEL_NAME",MODEL_NAME)

    if cuda & setting['cuda']:
        model.cuda()
        topofea = topofea.cuda()
        features = features.cuda()
        NA = NA.cuda()
        label = label.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()

    optimizer = torch.optim.Adam(model.parameters(),setting['lr'],weight_decay=setting['weight_decay'])



    def train(MODEL_NAME,model,epochs,optimizer):
        model.train()
        optimizer.zero_grad()
        if MODEL_NAME=='gcn':
            output = model(features,NA)
        else:
            output = model(features)

        loss = F.nll_loss(output[idx_train],label[idx_train])

        acc = accuracy(output[idx_train],label[idx_train])
        loss.backward()
        optimizer.step()
        acc_test, macro_f1 = main_test(model,MODEL_NAME)
        print('e:{}'.format(epochs),
              'ltr: {:.4f}'.format(loss.item()),
              'atr: {:.4f}'.format(acc.item()),
              'ate: {:.4f}'.format(acc_test.item()),
              'f1te:{:.4f}'.format(macro_f1.item()))
        return loss.item(), acc_test.item(), macro_f1.item()



    def main_test(model,MODEL_NAME):
        model.eval()
        if MODEL_NAME=='gcn':
            output = model(features,NA)
        else:
            output = model(features)

        acc_test = accuracy(output[idx_test], label[idx_test])
        label_max = []
        for idx in idx_test:
            label_max.append(torch.argmax(output[idx]).item())
        labelcpu = label[idx_test].data.cpu()
        macro_f1 = f1_score(labelcpu, label_max, average='macro')
        return acc_test, macro_f1

    acc_max = 0
    f1_max = 0
    epoch_max = 0
    for epoch in range(setting["epoch"]):
        loss, acc_test, macro_f1 = train(MODEL_NAME,model,epoch,optimizer)
        if acc_test >= acc_max:
            acc_max = acc_test
            f1_max = macro_f1
            epoch_max = epoch

    print('epoch:{}'.format(epoch_max),
          'acc_max: {:.4f}'.format(acc_max),
          'f1_max: {:.4f}'.format(f1_max))

