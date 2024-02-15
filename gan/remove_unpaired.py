import shutil
import os
import glob
import torchaudio


clean_train_path = 'data-1/clean_processed'
noisy_train_path = 'data-1/noisy_processed'
clean_test_path = 'data-1/test_clean_processed'
noisy_test_path = 'data-1/test_noisy_processed'

clean_train = [file for file in os.listdir(clean_train_path)]
noisy_train = [file for file in os.listdir(noisy_train_path)]
clean_test = [file for file in os.listdir(clean_test_path)]
noisy_test = [file for file in os.listdir(noisy_test_path)]

print(f'Number of clean train files: {len(clean_train)}')
print(f'Number of noisy train files: {len(noisy_train)}')
print(f'Number of clean test files: {len(clean_test)}')
print(f'Number of noisy test files: {len(noisy_test)}')

# wrongs = 0
# for i in range(len(clean_train)):
#     if clean_train[i] != noisy_train[i]:
#         wrongs += 1

paired_clean_test = []
paired_noisy_test = []

wrongs = abs(len(clean_test) - len(noisy_test))
for i in range(len(clean_test)):
    if clean_test[i] not in noisy_test:
        wrongs += 1
    else:
        paired_clean_test.append(clean_test[i])
        # find the index of the clean file in the noisy files
        index = noisy_test.index(clean_test[i])
        paired_noisy_test.append(noisy_test[index])


print("len paired noisy test",len(paired_noisy_test))

#%%


# Define the source and target directories
source_dir = 'data-1/test_clean_processed'
target_dir = 'data-1/test_clean_paired'

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Move each file in the list to the target directory
for filename in paired_clean_test:
    source_file_path = os.path.join(source_dir, filename)
    target_file_path = os.path.join(target_dir, filename)
    
    # Check if the file exists before moving
    if os.path.exists(source_file_path):
        shutil.move(source_file_path, target_file_path)
        print(f'Moved {filename} to {target_dir}')
    else:
        print(f'File {filename} does not exist in the source directory.')

# Define the source and target directories
source_dir = 'data-1/test_noisy_processed'
target_dir = 'data-1/test_noisy_paired'

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Move each file in the list to the target directory
for filename in paired_noisy_test:
    source_file_path = os.path.join(source_dir, filename)
    target_file_path = os.path.join(target_dir, filename)
    
    # Check if the file exists before moving
    if os.path.exists(source_file_path):
        shutil.move(source_file_path, target_file_path)
        print(f'Moved {filename} to {target_dir}')
    else:
        print(f'File {filename} does not exist in the source directory.')


print('All specified files have been moved to the target directory.')



# save paired files
# for file in paired_clean_test:
#     noisy_waveform, _ = torchaudio.load(clean_test_path + file)
#     torchaudio.save(f'data-1/test_clean_paired/{file}.wav', noisy_waveform, 16000)


# with open('paired_clean_test.txt', 'w') as f:
#     for file in paired_clean_test:
#         f.write(f'data-1/test_clean_paired/{file}\n')

# print(f'Number of wrong files: {wrongs} out of {len(noisy_test)}')


# %%

clean_test_path = 'data-1/test_clean_paired'
noisy_test_path = 'data-1/test_noisy_paired'

clean_test = [file for file in os.listdir(clean_test_path)]
noisy_test = [file for file in os.listdir(noisy_test_path)]

wrongs = 0
for i in range(len(clean_test)):
    if clean_test[i] not in noisy_test:
        wrongs += 1

print(f'Number of wrong files: {wrongs} out of {len(noisy_test)}')
