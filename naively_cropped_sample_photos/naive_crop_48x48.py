from PIL import Image
import os

directory_in_str = '.'

directory = os.fsencode(directory_in_str)


for file in os.listdir(directory):
    filename = os.fsdecode(file)
    basewidth = 300
    if not filename.startswith('.'):
        img = Image.open(filename)

        # 48px * 48px, starting in the center
        half_the_width = img.size[0] / 2
        half_the_height = img.size[1] / 2
        img_curr = img.crop(
            (
                half_the_width - 24,
                half_the_height - 24,
                half_the_width + 24,
                half_the_height + 24
            )
        )

        resized_name = 'CROPPED_' + filename  
        img_curr.save(resized_name)

        
