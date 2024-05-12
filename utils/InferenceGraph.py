import matplotlib.pyplot as plt

################################################################################
# Model data from inference.xlsx
################################################################################
models = ['CNNBaseline', 'CNN Modified', 'Resnet18 modified', 'ViT']
avg_epoch_time_gpu = [4.26, 6.21, 11.05, 35.20]
test_eval_time_gpu = [0.39, 0.49, 0.72, 3.74]

################################################################################
# Plot epoch time and test eval time
################################################################################
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

axs[0].bar(models, avg_epoch_time_gpu, label='GPU', color='orange', alpha=0.6)
axs[0].set_title('Average Epoch Time (sec)')
axs[0].set_ylabel('Time (s)')
axs[0].legend()

axs[1].bar(models, test_eval_time_gpu, label='GPU', color='orange', alpha=0.6)
axs[1].set_title('Test Evaluation Time (sec)')
axs[1].set_xlabel('Model')
axs[1].set_ylabel('Time (s)')
axs[1].legend()
plt.show()
