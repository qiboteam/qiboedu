import os
import torch
from torch import nn, optim
import json, time
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_curve, roc_auc_score

def buildROC(labels, score, targetEff=[0.3,0.5]):
    r''' ROC curve is a plot of the true positive rate (Sensitivity) in the function of the false positive rate
    (100-Specificity) for different cut-off points of a parameter. Each point on the ROC curve represents a
    sensitivity/specificity pair corresponding to a particular decision threshold. The Area Under the ROC
    curve (AUC) is a measure of how well a parameter can distinguish between two diagnostic groups.
    '''
    if not isinstance(targetEff, list):
        targetEff = [targetEff]
    fpr, tpr, threshold = roc_curve(labels, score)
    idx = [np.argmin(np.abs(tpr - Eff)) for Eff in targetEff]
    eB, eS = fpr[idx], tpr[idx]
    return fpr, tpr, threshold, eB, eS


def run(model, epoch, loader, partition, loss_fn, optimizer=None, N_EPOCHS=None, device='cpu', dtype=torch.float64):
    if partition == 'train':
        model.train()
    else:
        model.eval()

    res = {'time':0, 'correct':0, 'loss': 0, 'counter': 0, 'acc': 0,
           'loss_arr':[], 'correct_arr':[],'label':[],'score':[]}

    tik = time.time()
    loader_length = len(loader)

    for i, (label, p4s, nodes, atom_mask, edge_mask, edges) in enumerate(loader):
        if partition == 'train':
            optimizer.zero_grad()
        
        batch_size, n_nodes, _ = p4s.size()
        atom_positions = p4s.view(batch_size * n_nodes, -1).to(device, dtype)
        atom_mask = atom_mask.view(batch_size * n_nodes, -1).to(device)
        edge_mask = edge_mask.reshape(batch_size * n_nodes * n_nodes, -1).to(device)
        nodes = nodes.view(batch_size * n_nodes, -1).to(device,dtype)
        edges = [a.to(device) for a in edges]
        label = label.to(device, dtype).long()

        pred = model(scalars=nodes, x=atom_positions, edges=edges, node_mask=atom_mask,
                         edge_mask=edge_mask, n_nodes=n_nodes)
        
        predict = pred.max(1).indices
        correct = torch.sum(predict == label).item()
        loss = loss_fn(pred, label)
        
        if partition == 'train':
            loss.backward()
            optimizer.step()
        elif partition == 'test':
            # save labels and probilities for ROC / AUC
            score = torch.nn.functional.softmax(pred, dim = -1)
            res['label'].append(label)
            res['score'].append(score)

        res['time'] = time.time() - tik
        res['correct'] += correct
        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())
        res['correct_arr'].append(correct)
        
    running_loss = sum(res['loss_arr'])/len(res['loss_arr'])
    running_acc = sum(res['correct_arr'])/(len(res['correct_arr'])*batch_size)
    avg_time = res['time']/res['counter'] * batch_size
    tmp_counter = res['counter']
    tmp_loss = res['loss'] / tmp_counter
    tmp_acc = res['correct'] / tmp_counter

    if N_EPOCHS:
        print(">> %s \t Epoch %d/%d \t Batch %d/%d \t Loss %.4f \t Running Acc %.3f \t Total Acc %.3f \t Avg Batch Time %.4f" %
             (partition, epoch + 1, N_EPOCHS, i, loader_length, running_loss, running_acc, tmp_acc, avg_time))
    else:
        print(">> %s \t Loss %.4f \t Running Acc %.3f \t Total Acc %.3f \t Avg Batch Time %.4f" %
             (partition, running_loss, running_acc, tmp_acc, avg_time))
        
    torch.cuda.empty_cache()
    # ---------- reduce -----------
    if partition == 'test':
        res['label'] = torch.cat(res['label']).unsqueeze(-1)
        res['score'] = torch.cat(res['score'])
        res['score'] = torch.cat((res['label'],res['score']),dim=-1)
    res['counter'] = res['counter']
    res['loss'] = res['loss'] / res['counter']
    res['acc'] = res['correct'] / res['counter']
    return res

def train(model, optimizer, lr_scheduler, dataloaders, res, N_EPOCHS, model_path, log_path, loss_fn,
          device='cpu', dtype=torch.float64):

    ### training and validation
    for epoch in range(N_EPOCHS):
        train_res = run(model, epoch, dataloaders['train'], partition='train', 
                        loss_fn=loss_fn, optimizer=optimizer, N_EPOCHS = N_EPOCHS, 
                        device=device, dtype=dtype)
        print("Time: train: %.2f \t Train loss %.4f \t Train acc: %.4f" % (train_res['time'],train_res['loss'],train_res['acc']))
        
        torch.save(model.state_dict(), os.path.join(model_path, "checkpoint-epoch-{}.pt".format(epoch)) )
        with torch.no_grad():
            val_res = run(model, epoch, dataloaders['val'], partition='val', loss_fn=loss_fn, device=device, dtype=dtype)
            
        # if (args.local_rank == 0): # only master process save
        res['lr'].append(optimizer.param_groups[0]['lr'])
        res['train_time'].append(train_res['time'])
        res['val_time'].append(val_res['time'])
        res['train_loss'].append(train_res['loss'])
        res['train_acc'].append(train_res['acc'])
        res['val_loss'].append(val_res['loss'])
        res['val_acc'].append(val_res['acc'])
        res['epochs'].append(epoch)

        ## save best model
        if val_res['acc'] > res['best_val']:
            print("New best validation model, saving...")
            torch.save(model.state_dict(), os.path.join(model_path,"best-val-model.pt"))
            res['best_val'] = val_res['acc']
            res['best_epoch'] = epoch

        print("Epoch %d/%d finished." % (epoch, N_EPOCHS))
        print("Train time: %.2f \t Val time %.2f" % (train_res['time'], val_res['time']))
        print("Train loss %.4f \t Train acc: %.4f" % (train_res['loss'], train_res['acc']))
        print("Val loss: %.4f \t Val acc: %.4f" % (val_res['loss'], val_res['acc']))
        print("Best val acc: %.4f at epoch %d." % (res['best_val'],  res['best_epoch']))

        json_object = json.dumps(res, indent=4)
        with open(os.path.join(log_path, "train-result-epoch{}.json".format(epoch)), "w") as outfile:
            outfile.write(json_object)

        ## adjust learning rate
        if (epoch < 31):
            lr_scheduler.step(metrics=val_res['acc'])
        else:
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*0.5


def test(model, dataloaders, res, model_path, log_path, loss_fn, 
         device='cpu', dtype=torch.float64):
    ### test on best model
    best_model = torch.load(os.path.join(model_path, "best-val-model.pt"), map_location=device)
    model.load_state_dict(best_model)
    with torch.no_grad():
        test_res = run(model, 0, dataloaders['test'], loss_fn=loss_fn, partition='test', device=device, dtype=dtype)

    print("Final ", test_res['score'])
    pred = test_res['score'].cpu()

    np.save(os.path.join(log_path, "score.npy"), pred)
    fpr, tpr, thres, eB, eS  = buildROC(pred[...,0], pred[...,2])
    auc = roc_auc_score(pred[...,0], pred[...,2])

    metric = {'test_loss': test_res['loss'], 'test_acc': test_res['acc'],
              'test_auc': auc, 'test_1/eB_0.3':1./eB[0],'test_1/eB_0.5':1./eB[1]}
    res.update(metric)
    print("Test: Loss %.4f \t Acc %.4f \t AUC: %.4f \t 1/eB 0.3: %.4f \t 1/eB 0.5: %.4f"\
           % (test_res['loss'], test_res['acc'], auc, 1./eB[0], 1./eB[1]))
    json_object = json.dumps(res, indent=4)
    with open(os.path.join(log_path, "test-result.json"), "w") as outfile:
        outfile.write(json_object)


# if __name__ == "__main__":
    
#     N_EPOCHS = 60

#     # put your model and log path here.
#     model_path = "../models/LieEQGNN/"
#     log_path = "../logs/LieEQGNN/"

#     ### set random seed
#     torch.manual_seed(42)
#     np.random.seed(42)

#     ### initialize cuda
#     device = 'cuda'
#     dtype = torch.float64

#     model = LieEQGNN(n_scalar = 1, n_hidden = 4, n_class = 2,\
#                        dropout = 0.2, n_layers = 1,\
#                        c_weight = 1e-3).to(device, dtype)

#     ### print model and dataset information
#     pytorch_total_params = sum(p.numel() for p in model.parameters())
#     print("Model Size:", pytorch_total_params)
#     for (split, dataloader) in dataloaders.items():
#         print(f" {split} samples: {len(dataloader.dataset)}")

#     ### optimizer
#     optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

#     ### lr scheduler
#     base_scheduler = CosineAnnealingWarmRestarts(optimizer, 4, 2)
#     lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1,\
#                                                 warmup_epoch=5,\
#                                                 after_scheduler=base_scheduler) ## warmup

#     ### loss function
#     loss_fn = nn.CrossEntropyLoss()

#     ### initialize logs
#     res = {'epochs': [], 'lr' : [],\
#            'train_time': [], 'val_time': [],  'train_loss': [], 'val_loss': [],\
#            'train_acc': [], 'val_acc': [], 'best_val': 0, 'best_epoch': 0}

#     ### training and testing
#     print("Training...")
#     train(model, res, N_EPOCHS, model_path, log_path)
#     test(model, res, model_path, log_path)