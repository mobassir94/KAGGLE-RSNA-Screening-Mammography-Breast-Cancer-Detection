import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import gc
from rsna_metrics import *
from rsna_model import save_model
from torch.utils.data import Dataset, DataLoader
from rsna_dataset import BreastCancerDataset

import pandas as pd

class CFG:
    epochs = 1
    clf_threshold = 0.1
    pos_weight = 0.0
    lr = 5e-4
    tta = False
    freeze = False
    fp16 = False
    SAVE_FOLDER = "../trained_folds/"

    
class FocalLoss(nn.Module):
    def __init__(self, alpha=(1,1), gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, preds, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(torch.flatten(preds), torch.flatten(targets).float())
        logits=nn.Sigmoid()(preds)
        F_loss = (self.alpha[0]*(targets)*((1-logits)**self.gamma)+ self.alpha[1]*(1-targets)*((logits)**self.gamma)) * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
def weighted_binary_cross_entropy(output, target, weights=None):
    output = torch.clamp(output, max=1-1e-7)  
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

def BCELoss_class_weighted(weights):
    """
    weights[0] is weight for class 0 (negative class)
    weights[1] is weight for class 1 (positive class)
    """
    def loss(y_pred, target):
        y_pred = torch.clamp(y_pred,min=1e-7,max=1-1e-7) # for numerical stability
        bce = - weights[1] * target * torch.log(y_pred) - (1 - target) * weights[0] * torch.log(1 - y_pred)
        return torch.mean(bce)

    return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

# criterion = FocalLoss(alpha=[50,1]).to(device)
# criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([CFG.pos_weight])).to(device)

# criterion = nn.BCEWithLogitsLoss().to(device)
def freeze_bn(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

    
def train_one_epoch(dataloader, model, scheduler, optimizer, scaler, epoch):
#     if scheduler is not None:
#         scheduler.step(epoch)
    torch.manual_seed(42)
    
    
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Train: Epoch {epoch + 1}", total=len(dataloader), mininterval=5)
    
    for img, target in pbar:
        img = img.to(device)
#         torch.set_grad_enabled(True)
        # Using mixed precision training
        if(CFG.fp16):
#             for m in model.modules():
#                 if isinstance(m, nn.BatchNorm2d):
#                     m.eval()
#                     m.weight.requires_grad = False
#                     m.bias.requires_grad = False

#             freeze_bn(model)
            optimizer.zero_grad()
            with autocast(enabled=True):
            
                outputs = model(img)
              
#                 target = target.unsqueeze(1)
            
                target = target.to(float).to(device)
                ones = (target == 1.).sum()
                zeroes = (target == 0.).sum()
                
               
                weight = target * 1
                weight[weight == 0] = zeroes.item()
#                 print(weight)
                CFG.pos_weight = (len(target) - ones)/(ones +1e-5)   
         
#                 criterion = BCELoss_class_weighted(weights = [w_neg, w_pos])
#                 criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([CFG.pos_weight])).to(device)

#                 print(outputs.shape,target.shape)
#                 loss = criterion(outputs, target)
                
#                 loss = weighted_binary_cross_entropy(outputs, target, weights=[w_pos,w_neg])

#                 substracted = zeroes - ones
#                 weight = substracted/zeroes

#                 CFG.pos_weight = 1.0-weight#1-(ones/24)
#                 print(CFG.pos_weight,weight)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        outputs,
                        target,
#                         weight = weight.to(device),
                        pos_weight=torch.tensor([CFG.pos_weight]).to(device)
                    )
            
                if np.isinf(loss.item()) or np.isnan(loss.item()):
                    print(f'Bad loss, skipping the batch ',loss)
                    del loss, outputs
                    gc.collect()
                    continue
#                 if(loss.item()>100.0):
#                     print(f'Bad loss, skipping the batch ',loss)
#                     del loss, outputs
#                     gc.collect()
#                     continue
            
            
            # scaler is needed to prevent "gradient underflow"

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, norm_type=2.0)
            
            scaler.step(optimizer)
            #https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step/164814/2
            old_scaler = scaler.get_scale()
            scaler.update()
            new_scaler = scaler.get_scale() 
            
            if new_scaler >= old_scaler:
                if scheduler is not None:
                    scheduler.step()
            else:
                print("old_scaler ",old_scaler,"new_scaler ",new_scaler)
                
            
        else:
            outputs = model(img)
            target = target.to(float).to(device)
            loss = criterion(outputs, target)
         
#             loss = torch.nn.functional.binary_cross_entropy_with_logits(
#                         outputs,
#                         target,
#                         pos_weight=torch.tensor([CFG.pos_weight]).to(device)
#                     )
            
            if np.isinf(loss.item()) or np.isnan(loss.item()):
                print(f'Bad loss, skipping the batch ')
                del loss, outputs
                gc.collect()
                continue
 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, norm_type=2.0)
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        lr = scheduler.get_last_lr()[0] if scheduler else CFG.lr
        loss = loss.item()
        
        pbar.set_postfix({"loss": loss, "lr": lr})
        total_loss += loss
        
    total_loss /= len(dataloader)
    gc.collect()
    torch.cuda.empty_cache()
    return total_loss

def valid_one_epoch(dataloader, model, epoch):
    torch.manual_seed(42)
    model.eval()
    is_autocast = False
    if(CFG.fp16):
        is_autocast = True
#     torch.set_grad_enabled(False)
    pred_cancer = []
    with torch.no_grad():
        total_loss = 0
        targets = []
        pbar = tqdm(dataloader, desc=f'Eval: {epoch + 1}', total=len(dataloader), mininterval=5)

        for img, target in pbar:
            if(is_autocast):
                with autocast(enabled=is_autocast):
                    img = img.to(device)
#                     target = target.unsqueeze(1)
                    target = target.to(float).to(device)

                    outputs = model(img)
                  
                    if CFG.tta:
                        outputs2 = model(torch.flip(img, dims=[-1])) # horizontal mirror
                        outputs = (outputs + outputs2) / 2
                        
                    ones = (target == 1.).sum()
                    zeroes = (target == 0.).sum()
#                     w_pos = ones 
#                     w_neg = zeroes 
#                     criterion = BCELoss_class_weighted(weights = [w_neg, w_pos])

#                     loss = criterion(outputs, target)

                    substracted = zeroes - ones
                    weight = substracted/zeroes
    
                    CFG.pos_weight = 0.0
        
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            outputs,
                            target,
                            pos_weight=torch.tensor([CFG.pos_weight]).to(device)
                        )

                    pbar.set_postfix({"loss": loss})

                    outputs = torch.sigmoid(outputs)
                    outputs = torch.nan_to_num(outputs)
                    pred_cancer.append(outputs)
                    total_loss += loss
                    targets.append(target.cpu().numpy())
                    
            else:
                img = img.to(device)
                
#                 target = target.unsqueeze(1)
                target = target.to(float).to(device)
                
                outputs = model(img)
                if CFG.tta:
                    outputs2 = model(torch.flip(img, dims=[-1])) # horizontal mirror
                    outputs = (outputs + outputs2) / 2
#                 loss = criterion(outputs, target)
            
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        outputs,
                        target
#                         pos_weight=torch.tensor([CFG.pos_weight]).to(device)
                    )
                
                pbar.set_postfix({"loss": loss})
                
                outputs = torch.sigmoid(outputs)
                outputs = torch.nan_to_num(outputs)
                pred_cancer.append(outputs)
                total_loss += loss
                targets.append(target.cpu().numpy())
            

    
    targets = np.concatenate(targets)
    pred = torch.concat(pred_cancer).cpu().numpy()
    tpf1, thres = optimal_f1(targets, pred)
    pf1 = pfbeta(targets, pred, 1.0)
    
    try:
        auc = metrics.roc_auc_score(targets, pred)
        
    except ValueError:
        auc = 0.0
        print("ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.")
    
    total_loss /= len(dataloader)
    
    all_outputs = (np.array(pred) > 0.2).astype(np.int8).tolist()
    try:
        bin_score = pfbeta(targets, all_outputs, 1.0)
    except:
        bin_score = 0.0
    print("pf1 -> ",pf1," *** bin_score for 0.2 threshold -> ",bin_score)    
    m = compute_metric(pred, targets)
    text = f'{"groupby mean()": <16}'
    text += f'\t auc {m["auc"]:0.5f}'
    text += f'\t threshold {m["threshold"]:0.5f}'
    text += f'\t f1score {m["f1score"]:0.5f} | '
    text += f'\t pf1score {m["pf1"]:0.5f}'
    text += f'\t pthr {m["pthr"]:0.5f} '
    text += f'\t precision {m["precision"]:0.5f}'
    text += f'\t recall {m["recall"]:0.5f} | '
    text += f'\t sensitivity {m["sensitivity"]:0.5f}'
    text += f'\t specificity {m["specificity"]:0.5f}'
    text += '\n'
    print(text)
    
    gc.collect()
    torch.cuda.empty_cache()
#     tpf1 = bin_score
    return total_loss.cpu().numpy(), tpf1,thres,pred,auc

def train_fnc(train_dataloader, valid_dataloader, model, fold, optimizer, scheduler):
    train_losses = []
    valid_losses = []
    valid_scores = []
    thresholds = []
    
    scaler = GradScaler(enabled = True)#init_scale=16384.0,
#     best_loss = 999
    best_score = -1
    for epoch in range(CFG.epochs):
        train_loss = train_one_epoch(train_dataloader, model, scheduler, optimizer, scaler, epoch)
        
        
#         tr_loss, tr_score, thr, auc = valid_one_epoch(train_dataloader, model, epoch)
        
#         print("TRAIN AUC = ",auc," train thresholded f1 = ",tr_score," for threshold = ",thr)
        
        valid_loss, valid_score, thres,pred,val_auc = valid_one_epoch(valid_dataloader, model, epoch)
  
        print("----------> Validation AUC = ",val_auc," valid thresholded probabilistic_f1 = ",valid_score," for threshold = ",thres)
    
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_scores.append(valid_score)
        thresholds.append(thres)

        
        if valid_score > best_score:
            best_score = valid_score
            save_model(f"{CFG.SAVE_FOLDER}fold{fold}_best_score.pth", model)
            print(">>>>>>>New Best Score")
        
#         if valid_loss < best_loss:
#             best_loss = valid_loss
#             save_model(f"{SAVE_FOLDER}fold{fold}_best_loss.pth", model)
#             print(">>>>>>>>>>>>>>>>New Best Loss")
        print()
        
        print(f"-------- Epoch {epoch + 1} --------")
        print("Train Loss: ", train_loss)
        print("Valid Loss: ", valid_loss)
        print("pF1: ", valid_score)
        print("Best Score: ", best_score)
#         print("Best Loss: ", best_loss)
        print()
        
    column_names = ['train_loss','valid_loss', 'pF1','threshold']
    df = pd.DataFrame(np.stack([train_losses, valid_losses, valid_scores,thresholds],
                               axis=1),columns=column_names)
    display(df)
#     plot_df(df)




def gen_predictions(models, train):
    train_predictions = []
    pbar = tqdm(enumerate(models), total=len(models), desc='Folds')
    for fold, model in pbar:
        if model is not None:
            eval_dataset = BreastCancerDataset(train.query('fold == @fold'), transforms="valid")
            eval_dataloader = DataLoader(eval_dataset, batch_size=CFG.batch_size, shuffle=False)
            
            eval_loss, pF1, thres, pred,val_auc = valid_one_epoch(eval_dataloader, model, -1)
           
            pbar.set_description(f'Eval fold:{fold} pF1:{pF1:.02f}')
            pred_df = pd.DataFrame(data=pred,
                                          columns=['cancer_pred_proba'])
            pred_df['cancer_pred'] = pred_df.cancer_pred_proba > thres

            df = pd.concat(
                [train.query('fold == @fold').reset_index(drop=True), pred_df],
                axis=1
            ).sort_values(['patient_id', 'image_id'])
            train_predictions.append(df)
    train_predictions = pd.concat(train_predictions)
    return train_predictions
