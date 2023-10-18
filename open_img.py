from PIL import Image
import io
import os 

GA_folder = '/Volumes/fsmresfiles/Ophthalmology/Mirza_Images/AMD/dAMD_GA'

subfolder_names = os.listdir(GA_folder)
subfolders = []
subfolder_paths = []
for subfolder_name in subfolder_names:
    if os.path.isdir(os.path.join(GA_folder, subfolder_name)):
        subfolders.append(subfolder_name)
        subfolder_paths.append(os.path.join(GA_folder, subfolder_name))



# Assuming each image has a size of 1536x500 pixels and is in 8-bit format
image_height = 1536
image_width = 500
bits_per_pixel = 8

# Calculate image size in bytes
image_size = (image_height * image_width * bits_per_pixel) // 8

print('starting loop')

for i in range(len(subfolder_paths)):
    img_path = os.path.join(subfolder_paths[i], subfolders[i] + '.img')
    if not os.path.exists(img_path):
        print('NO IMG FILE FOUND IN ' + subfolder_paths[i])
    else:
        output_folder = os.path.join(subfolder_paths[i], '500_slices')
        os.makedirs(output_folder, exist_ok=True)
        with open(img_path, 'rb') as f:
            data = f.read()

        offset = 0
        for j in range(500):
            if j % 20 == 0:
                print(f'processing slice {j+1} of {subfolders[i]}')
            image_data = data[offset : offset + image_size]
            image = Image.frombytes('L', (image_width, image_height), image_data)
            out_image = os.path.join(output_folder, subfolders[i] + f'_{j+1}.jpg')
            image.save(out_image, 'JPEG')
            offset += image_size
        print('done with ' + subfolders[i])

