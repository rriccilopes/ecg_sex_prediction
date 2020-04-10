import os
from twilio.rest import Client
import tensorflow as tf 

os.environ["TWILIO_AUTH_TOKEN"] = "ADDYOURS"
os.environ["TWILIO_ACCOUNT_SID"] = "ADDYOURS"

def send_message(message):
    # client credentials are read from TWILIO_ACCOUNT_SID and AUTH_TOKEN
    client = Client()
    
    # this is the Twilio sandbox testing number
    from_whatsapp_number='whatsapp:+ADDYOURS'
    # replace this number with your own WhatsApp Messaging number
    to_whatsapp_number='whatsapp:+ADDYOURS'
    
    client.messages.create(body=message,
                           from_=from_whatsapp_number,
                           to=to_whatsapp_number)
    
class NotifyWhatsAppCallback(tf.keras.callbacks.Callback):

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        send_message('Epoch: {:7d}\nloss: {:2.2f}, val_loss: {:2.2f}, acc: {:2.2f}, val_acc: {:2.2f}, auc: {:2.2f}, val_auc: {:2.2f}'.format(epoch, logs['loss'], logs['val_loss'], logs['acc'], logs['val_acc'], logs['auroc'], logs['val_auroc']))
