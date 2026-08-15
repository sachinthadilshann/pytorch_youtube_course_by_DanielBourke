
import torch

from tqdm.auto import tqdm
from typing import Dict,List,Tuple

def train_step(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               device:torch.device):

  model.train()
  train_loss,train_acc = 0,0

  for X,y in dataloader:
    X,y= X.to(device), y.to(device)

    y_pred = model(X)

    loss = loss_fn(y_pred,y)
    train_loss +=loss.item()

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #cal_accuracy

    y_pred_class = y_pred.argmax(dim=1)
    train_acc += (y_pred_class == y).float().mean().item()

  train_loss = train_loss / len(dataloader)
  train_acc =  train_acc / len(dataloader)

  return train_loss,train_acc

def test_step(model:torch.nn.Module,
              dataloader:torch.utils.data.DataLoader,
              loss_fn:torch.nn.Module,
              device:torch.device):

  model.eval()

  test_loss,test_acc = 0,0

  with torch.inference_mode():
    for X,y in dataloader:
      X,y = X.to(device), y.to(device)
      test_pred_logits = model(X)

      loss = loss_fn(test_pred_logits,y)
      test_loss += loss.item()

      test_pred_labels = test_pred_logits.argmax(dim=1)
      test_acc += (test_pred_labels == y).float().mean().item()

  test_loss /= len(dataloader)
  test_acc /= len(dataloader)

  return test_loss,test_acc


def train(
    model:torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn:torch.nn.Module,
    epochs:int,
    device:torch.device,
):
  results = {"train_loss":[],
             "train_acc":[],
             "test_loss":[],
             "test_acc":[]}


  for epoch in tqdm(range(epochs)):
    train_loss,train_acc = train_step(model=model,
                                      dataloader=train_dataloader,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      device=device,
                                      )

    test_loss,test_acc = test_step(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=device,
                                   )

    print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  return results
