import utils
import run_knn
import plot_digits
import matplotlib.pyplot as plt

train_inputs, train_targets = utils.load_train()
valid_inputs, valid_targets = utils.load_valid()
test_inputs, test_targets = utils.load_test()

valid_rate=[]
test_rate=[]
values = [1, 3, 5, 7, 9]

for k in values:
  valid_labels = run_knn.run_knn(k, train_inputs, train_targets, valid_inputs)
  correct = 0
  for i in range(len(valid_inputs)):
    if valid_labels[i] == valid_targets[i]:
      correct += 1
  valid_rate.append(float(number_correct)/len(valid_inputs))

for k in values:
  test_labels = run_knn.run_knn(k, train_inputs, train_targets, test_inputs)
  correct = 0
  for i in range(len(test_inputs)):
    if test_labels[i] == test_targets[i]:
      correct += 1
  test_rate.append(float(number_correct)/len(test_inputs))

plt.plot(values,valid_rate, label="Validation set")
plt.plot(values,test_rate, label="Test set")
plt.legend()
plt.title("KNN")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()
