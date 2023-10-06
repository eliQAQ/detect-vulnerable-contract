import matplotlib.pyplot as plt
import pickle
with open('./draw_model/bilstm-attention', 'rb') as f:
    data_1 = pickle.load(f)
with open('./draw_model/bilstm', 'rb') as f:
    data_2 = pickle.load(f)
with open('./draw_model/lstm', 'rb') as f:
    data_3 = pickle.load(f)
with open('./draw_model/gru', 'rb') as f:
    data_4 = pickle.load(f)

print(data_1)
epochs = range(0, 30)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(epochs,data_1['acc'], 'gold', label='our_model')
plt.plot(epochs, data_2['acc'], 'darkturquoise', label='bilstm')
plt.plot(epochs, data_3['acc'], 'olive', label='lstm')
plt.plot(epochs, data_4['acc'], 'brown', label='gru')
plt.title('Training accuracy')
plt.legend(loc='lower right')
plt.figure()

plt.xlabel('epoch')
plt.ylabel('loss')
epochs = range(0, 30)
plt.plot(epochs,data_1['loss'], 'gold', label='our_model')
plt.plot(epochs, data_2['loss'], 'darkturquoise', label='bilstm')
plt.plot(epochs, data_3['loss'], 'olive', label='lstm')
plt.plot(epochs, data_4['loss'], 'brown', label='gru')
plt.title('Training loss')
plt.legend(loc='upper right')
plt.figure()


epochs = range(0, 30)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(epochs, data_1['acc'], 'darkgray', label='train')
plt.plot(epochs, data_1['val_acc'], 'mediumturquoise', label='val')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.figure()

epochs = range(0, 30)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(epochs, data_1['loss'], 'darkgray', label='train')
plt.plot(epochs, data_1['val_loss'], 'mediumturquoise', label='val')
plt.title('Training and validation loss')
plt.legend(loc='upper right')
plt.figure()
plt.show()
