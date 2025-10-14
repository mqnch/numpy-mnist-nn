# numpy-mnist-nn

# mnist neural network
a small neural network built from scratch using numpy. it trains on the classic mnist handwritten digit dataset and reaches solid accuracy; no tensorflow, no pytorch, just linear algebra

---

## features
- loads mnist `.idx` data files  
- normalizes pixel values (0–1)  
- 2-layer network with relu + softmax  
- mini-batch training  
- adam optimizer  
- prints loss + accuracy per epoch
- achieved 98.14% test accuracy and 99.83% training accuracy after 10 epochs of training

---

### running it
make sure your mnist files are in the same folder as `main.py`, then run:
```bash
python main.py
```

#### sample output:

```py
training...
[5 0 4 ... 5 6 8] [5 0 4 ... 5 6 8]
epoch 01 | loss = 0.1344 | train_acc = 95.96%
[5 0 4 ... 5 6 8] [5 0 4 ... 5 6 8]
epoch 02 | loss = 0.0750 | train_acc = 97.78%
[5 0 4 ... 5 6 8] [5 0 4 ... 5 6 8]
epoch 03 | loss = 0.0466 | train_acc = 98.69%
evaluating on test set...
[7 2 1 ... 4 5 6] [7 2 1 ... 4 5 6]
test accuracy: 0.9763
```


