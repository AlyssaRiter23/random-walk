import subprocess
import matplotlib.pyplot as plt

# Path to your CSV file
input_file = '2025_04_15_10_37.csv'

# Run the grep and awk commands from Python using subprocess
grep_command = f"grep -oE 'image_step_[0-9]+\\.png,[0-9]+\\.[0-9]+' {input_file}"
awk_command = "awk -F '[_. ,]' '{print $3, $NF}'"

# Combine and execute the shell command
command = f"{grep_command} | {awk_command}"
result = subprocess.check_output(command, shell=True, text=True)

# Build a dictionary of step → score
score_dict = {}
for line in result.splitlines():
    step_str, score_str = line.split()
    step = int(step_str)
    if not score_str.startswith('0.'):
        score_str = '0.' + score_str
    score = float(score_str)
    score_dict[step] = score

# Fill in all steps from 0 to 999, using 0 if missing
all_steps = list(range(500))
all_scores = [score_dict.get(step, 0) for step in all_steps]

# Assign colors based on score
colors = ['green' if score >= 0.7 else 'red' for score in all_scores]

# Plot the dot plot
plt.figure(figsize=(14, 6))
plt.scatter(all_steps, all_scores, c=colors, s=10)

# Add titles and labels
plt.title('Confidence Score Dot Plot by Step')
plt.xlabel('Step Number')
plt.ylabel('Confidence Score')
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.7)

# Save and show
plt.tight_layout()
plt.savefig('confidence_scores_dotplot_2025_04_15_10_37.png')
plt.show()

# Print debug info
print(f"Total steps processed: {len(score_dict)}")
print(f"Missing steps filled with 0: {500 - len(score_dict)}")

