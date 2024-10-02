import pandas as pd
import os

# Load the data
raw = pd.read_excel(r'raw/OCTA-DPN-第一、二部分.xlsx')

# Keep the first 14 columns
raw_cleaned = raw.iloc[:, :14]

# Proceed with the logical operations directly
raw_cleaned['upper'] = (raw_cleaned["正中神经1"] < 50) | (raw_cleaned["正中神经2"] < 50)
raw_cleaned['lower'] = (raw_cleaned["胫神经1"] < 40) | (raw_cleaned["胫神经2"] < 40)
raw_cleaned['symptom'] = raw_cleaned["症状"]
raw_cleaned['sign'] = (raw_cleaned["踝反射"] | raw_cleaned["振动觉"] | raw_cleaned["压力觉"] |
                      raw_cleaned["温度觉"] | raw_cleaned["针刺痛觉"])

# Create the label based on the logical conditions
raw_cleaned['label'] = (raw_cleaned['upper'] | raw_cleaned['lower']) & raw_cleaned['symptom'] & raw_cleaned['sign']

# Set '姓名' as index
raw_cleaned = raw_cleaned.set_index('姓名')

# Get the list of file names
names = os.listdir('clean/OCTA')

# Extract labels for each file
labels = [raw_cleaned.loc[name.split('_')[0], 'label'] for name in names]

# Create a DataFrame with names and labels
label = pd.DataFrame({
    'name': names,
    'label': labels,
})

# Convert label to integer
label['label'] = label['label'].astype(int)

# Print label counts
print(label['label'].value_counts())

# Save the labels to a CSV file
label.to_csv('./clean/label.csv', index=False, encoding='utf-8')
