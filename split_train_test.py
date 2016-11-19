import sys
from random import shuffle

filename = sys.argv[1]
ptrain = 0.8

with open(filename, 'r') as f:
    train_test = f.readlines()

shuffle(train_test)
ntrain = int(ptrain * len(train_test))

train = train_test[:ntrain]
test = train_test[ntrain+1:]

with open(filename[:-4] + '_train.txt', 'w') as f:
    for i in xrange(len(train)):
	f.write(train[i])

with open(filename[:-4] + '_test.txt', 'w') as f:
    for i in xrange(len(test)):
	f.write(test[i])

