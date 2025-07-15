import json
import matplotlib.pyplot as plt
import os
import seaborn as sns

if __name__ == "__main__":
    
    weight = 0
    # reward_type = 'arousal'
    reward_type = 'score'
    # reward_type = 'blended'
    
    # data_from = 'PPO'
    data_from = 'Explore'

    if weight == 0:
        label = 'Optimize'
    elif weight == 0.5:
        label = 'Blended'
    else:
        label = 'Arousal'
        
    dt_name = f"{data_from}_{label}_{reward_type}_SolidObs_DT_final"
    
    # Load the log_history.json file
    log_file_path = os.path.join("examples", "Agents", "DT", "Results", "preTrained", dt_name, "log_history.json")
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Results path {log_file_path} does not exist.") 
    with open(log_file_path, "r") as file:
        log_data = json.load(file)
        
    print(f"Generating plots for {dt_name}...")

    # Separate data into lists
    epochs = []
    loss = []
    grad_norm = []
    learning_rate = []
    eval_f1 = []
    eval_loss = []
    train_loss = 0

    for entry in log_data:
        if "epoch" in entry:
            epochs.append(entry["epoch"])
            if "loss" in entry:
                loss.append(entry["loss"])
            else:
                loss.append(None)
            if "grad_norm" in entry:
                grad_norm.append(entry["grad_norm"])
            else:
                grad_norm.append(None)
            if "learning_rate" in entry:
                learning_rate.append(entry["learning_rate"])
            else:
                learning_rate.append(None)
            if "eval_f1" in entry:
                eval_f1.append(entry["eval_f1"])
            else:
                eval_f1.append(None)
            if "eval_loss" in entry:
                eval_loss.append(entry["eval_loss"])
            else:
                eval_loss.append(None)
            if "train_loss" in entry:
                train_loss = entry["train_loss"]

    sns.set_palette("colorblind")
    palette = sns.color_palette("colorblind")

    img_name = f"{data_from}_{label}_DT"
    
    title_size = 25  # Font size for title
    axis_size = 25  # Font size for axes
    legend_size = 20  # Font size for legend
    tick_size = 25  # Font size for ticks
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, label="Loss", marker="o", markersize=4, linewidth=1, color=palette[0])  # Blue
    plt.plot(epochs, eval_f1, label="Eval F1", marker="o", markersize=4, linewidth=1, color=palette[1])  # Orange
    plt.xlabel("Epoch", fontsize=axis_size)
    plt.ylabel("Values", fontsize=axis_size)
    plt.title(f"Training loss and F1 evaluation", fontsize=title_size)
    plt.tick_params(axis='both', which='major', labelsize=tick_size)
    plt.legend(fontsize=legend_size)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"examples\\Agents\\DT\\Plots\\{img_name}_loss_F1_plot.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, grad_norm, label="Grad Norm", marker="o", markersize=4, linewidth=1, color=palette[2])  # Green
    plt.plot(epochs, eval_loss, label="Eval Loss", marker="o", markersize=4, linewidth=1, color=palette[3])  # Red
    plt.xlabel("Epoch", fontsize=axis_size) 
    plt.ylabel("Values", fontsize=axis_size)
    plt.title(f"Evaluation loss and Gradient Norm", fontsize=title_size)
    plt.tick_params(axis='both', which='major', labelsize=tick_size)
    plt.legend(fontsize=legend_size)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"examples\\Agents\\DT\\Plots\\{img_name}_GradNorm_EvalLoss.png", dpi=300)
    plt.show()
    
    print(f"Final Train Loss: {train_loss}")