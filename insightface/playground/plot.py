import subprocess
import matplotlib.pyplot as plt

# Path to your CSV file
input_file = '2025_04_15_10_37.csv'

# Run the grep and awk commands from Python using subprocess
grep_command = f"grep -oE 'image_step_[0-9]+\\.png,[0-9]+\\.[0-9]+' {input_file}"
awk_command = "awk -F '[_. ,]' '{print $3, $NF}'"

# Combine the commands using a pipe (|) to pass the output of grep to awk
command = f"{grep_command} | {awk_command}"

# Execute the command and capture the output
result = subprocess.check_output(command, shell=True, text=True)

# Initialize lists to store the extracted step numbers and confidence scores
steps, scores = [], []

# Process the output
for line in result.splitlines():
    step, score = line.split()
    if not score.startswith('0.'):
        score = '0.' + score  # Ensure score is properly formatted
    steps.append(int(step))
    scores.append(float(score))

# Count the number of confidence scores found
num_scores = len(scores)

# Calculate the number of missing scores (difference from 1000)
missing_scores = 500 - num_scores

# Append missing scores with a confidence score of 0
scores.extend([0] * missing_scores)

# Plotting the histogram
plt.figure(figsize=(10, 6))

# Create histogram bins
num_bins = 20  # Adjust this value to change bin granularity
plt.hist(scores, bins=num_bins, color='blue', alpha=0.7, edgecolor='black')

# Add titles and labels
plt.title('Histogram of Confidence Scores (Including Missing Scores)')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.xlim(0, 1)  # Set x-axis range from 0 to 1
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save and show the plot
plt.tight_layout()
plt.savefig('confidence_scores_histogram_2025_04_15_10_37.png')
plt.show()

# Print debug information
print(f"Number of confidence scores found: {num_scores}")
print(f"Number of missing scores (confidence 0): {missing_scores}")