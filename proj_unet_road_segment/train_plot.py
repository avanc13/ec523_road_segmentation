from tensorboard.backend.event_processing import event_accumulator
import os
import matplotlib.pyplot as plt

log_dir = 'logs'  # or 'logs/run_date_time'
runs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]

# pick the most recent run
run_path = sorted(runs, key=os.path.getmtime)[-1]
ea = event_accumulator.EventAccumulator(run_path)
ea.Reload()

# Print available scalar tags (metric names)
print("Available metrics:", ea.Tags()['scalars'])

# Replace with your actual keys from the printed list
train_loss = [e.value for e in ea.Scalars('epoch_loss')]
train_f1 = [e.value for e in ea.Scalars('epoch_f1')]  # adapt key if needed
epochs = [e.step for e in ea.Scalars('epoch_loss')]

# Plot
plt.figure(figsize=(10, 4))
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, train_f1, label='Train F1')
plt.xlabel('Epoch')
plt.title('Metrics from TensorBoard Logs')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('loss_and_f1_try.png')
plt.show()
