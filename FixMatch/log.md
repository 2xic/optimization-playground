# CIFAR-10

The paper say they were able to train on just 4 labels for each class.

Let's try...

Epochs 400

Baseline
- We weight unlabeled with the parameter at value 7
- We use the mean loss reudction
- We have a static loss
- Single dropout
- No output activation, raw outputs
- Paper uses Nesterov, we use adam currently
- 

test_accuracy         0.16750000417232513
test_accuracy         0.17640000581741333

### SUM > MEAN for reduction
Looks like mean had the best results

test_accuracy         0.15320000052452087
test_accuracy         0.1387999951839447

### Results of switching to softmax 
Use softmax instead of raw output. Looks like loss got stuck

test_accuracy         0.10000000149011612
test_accuracy         0.10000000149011612

### Results of switching to sigmoid + dropout
Loss also get's stuck here, hm.

test_accuracy         0.10000000149011612


### Results of removing the unsupervised loss parameter
It also get's stuck here.

test_accuracy         0.10000000149011612

### Some adjustments to the augmentation -> increased the crop out square

test_accuracy         0.10000000149011612

### Do the activation outside the model
I.e working on the raw outputs again,instead of using what the model outputs.

test_accuracy         0.10000000149011612

### Do activation in model, and softmax outside
Better.

test_accuracy         0.18240000307559967
test_accuracy         0.19979999959468842

### Use power of two in the weights
62k to 139k paraemters

test_accuracy         0.21310000121593475

Loss is still going stuck

### Rescaling the supervised vs unsupervised input
Use the scaling factor of 7 again on the unsupervised input

Still stuck.

test_accuracy         0.10000000149011612


### Warming up the model
First train with the scaling factor of 0 for 100 epochs

Then reenable it 1.

test_accuracy         0.21150000393390656

Okay, what if instead of 0, we gradually increase it for each 100 step.

test_accuracy         0.16910000145435333

Not working well.

We keep it at 1.

### Rescale with focus on supervised.

We scale with sueprvised with higher weights.

We know this label is correct, so it should probably get a higher score also.


test_accuracy         0.20669999718666077


### We increase the batch size from 64 to 256 and use 0.5 of the supervised samples.

test_accuracy         0.2037999927997589

### We increase the batch size from 64 to 256 and use 0.25 of the supervised samples and remove the loss weights

test_accuracy         0.21809999644756317

### we give ourself some more labels each class, I think 

Since there are more than 3_000 labels for each class, we are allowed 300 (10%).

We should not play on hard mode before we are beating the easier level first.

test_accuracy         0.4449999928474426
test_accuracy         0.45179998874664307
test_accuracy         0.4307999908924103


### Apply mask without detach
test_accuracy          0.435699999332428

### Normalize the images


### Train the first layers for x epochs on supervised, and freeze the layers except the last layer, and train the rest supervised + fixmatch


### Normal model (no fixmatch)
Okay, but what will a normal model (without fix match loss) get here ? 

test_accuracy         0.4474000036716461
test_accuracy         0.424699991941452
test_accuracy         0.4129999876022339

