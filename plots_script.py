import json
import math
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.pyplot as plt

json_path_teacher = "data/results/Feasibility_DeepSeek-R1-Distill-Qwen-32B-AWQ_Adult_Finetuned_False.json"
json_path_draft = "data/results/Feasibility_Qwen2.5-0.5B-Instruct_Adult_Finetuned_True.json"
json_path_refiner = "data/results/Feasibility_Worker_Qwen2.5-0.5B-Instruct_Refiner_Qwen2.5-3B-Instruct_adult_Finetuned_True.json"

with open(json_path_teacher, "r", encoding="utf-8") as f:
    data_teacher = json.load(f)

with open(json_path_draft, "r", encoding="utf-8") as f:
    data_draft = json.load(f)

with open(json_path_refiner, "r", encoding="utf-8") as f:
    data_refiner = json.load(f)


# Teacher
power_samples_teacher = data_teacher["power_samples"]
max_len_teacher = max(len(lst) for lst in power_samples_teacher)
for lst in power_samples_teacher:
    lst.extend([math.nan] * (max_len_teacher - len(lst)))
power_array_teacher = np.array(power_samples_teacher)
mean_vector_teacher = np.nanmean(power_array_teacher, axis=0)
std_vector_teacher = np.nanstd(power_array_teacher, axis=0)

# Draft
power_samples_draft = data_draft["power_samples"]
max_len_draft = max(len(lst) for lst in power_samples_draft)
for lst in power_samples_draft:
    lst.extend([math.nan] * (max_len_draft - len(lst))
)
power_array_draft = np.array(power_samples_draft)
mean_vector_draft = np.nanmean(power_array_draft, axis=0)
std_vector_draft = np.nanstd(power_array_draft, axis=0)

# Refiner
power_samples_refiner = data_refiner["power_samples"]
max_len_refiner = max(len(lst) for lst in power_samples_refiner)
for lst in power_samples_refiner:
    lst.extend([math.nan] * (max_len_refiner - len(lst)))
power_array_refiner = np.array(power_samples_refiner)
mean_vector_refiner = np.nanmean(power_array_refiner, axis=0)
std_vector_refiner = np.nanstd(power_array_refiner, axis=0)


# Time axes (in seconds, 200ms intervals)
time_teacher = np.arange(len(mean_vector_teacher)) * 0.2
time_draft = np.arange(len(mean_vector_draft)) * 0.2
time_refiner = np.arange(len(mean_vector_refiner)) * 0.2

plt.figure(figsize=(12, 6))

# Plot all functions on the same graph
plt.plot(time_teacher, mean_vector_teacher, label="DeepSeek-R1-D.-Qwen-32B", color="red")
plt.fill_between(
    time_teacher,
    mean_vector_teacher - std_vector_teacher,
    mean_vector_teacher + std_vector_teacher,
    color="red",
    alpha=0.2
)

plt.plot(time_draft, mean_vector_draft, label="Qwen2.5-0.5B-I.", color="green")
plt.fill_between(
    time_draft,
    mean_vector_draft - std_vector_draft,
    mean_vector_draft + std_vector_draft,
    color="green",
    alpha=0.2
)

plt.plot(time_refiner, mean_vector_refiner, label="MNR pipeline", color="blue")
plt.fill_between(
    time_refiner,
    mean_vector_refiner - std_vector_refiner,
    mean_vector_refiner + std_vector_refiner,
    color="blue",
    alpha=0.2
)

plt.xlabel("Time (s)", fontsize=24)
plt.ylabel("Power (W)", fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlim(0, max(len(mean_vector_teacher), len(mean_vector_draft), len(mean_vector_refiner)) * 0.2)
#plt.title("Power Consumption Over Time")
plt.legend(
    loc="center right",
    bbox_to_anchor=(1, 0.67),
    #ncol=3,
    fontsize=18,
    #frameon=False
)

plt.tight_layout()


# Add zoomed inset for the short draft sequence
# Define the zoom region (e.g., first 2 seconds)
zoom_start = 0
zoom_end = 3

ax = plt.gca()
axins = inset_axes(ax, width="35%", height="35%", loc="lower right", borderpad=4)
#axins.plot(time_teacher, mean_vector_teacher, color="blue")
#axins.fill_between(
#    time_teacher,
#    mean_vector_teacher - std_vector_teacher,
#    mean_vector_teacher + std_vector_teacher,
#    color="blue",
#    alpha=0.2
#)
axins.plot(time_draft, mean_vector_draft, color="green")
axins.fill_between(
    time_draft,
    mean_vector_draft - std_vector_draft,
    mean_vector_draft + std_vector_draft,
    color="green",
    alpha=0.2
)
axins.plot(time_refiner, mean_vector_refiner, color="blue")
axins.fill_between(
    time_refiner,
    mean_vector_refiner - std_vector_refiner,
    mean_vector_refiner + std_vector_refiner,
    color="blue",
    alpha=0.2
)

axins.set_xlim(zoom_start, zoom_end)
axins.set_ylim(160, 230)
axins.set_xticks([0, 1, 2, 3])
axins.set_yticks([160, 180, 200, 220])
#axins.set_xlabel("Time (s)", fontsize=12)
#axins.set_ylabel("Power (W)", fontsize=12)
axins.tick_params(axis='both', which='major', labelsize=14)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.savefig("power_consumption_comparison.pdf", dpi=300)
