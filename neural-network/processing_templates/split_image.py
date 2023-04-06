from PIL import Image
import cv2
import  os
import uuid

class Split:
    def __init__(self, path):
        self.count = 1
        self.path = path
        unique_id = uuid.uuid4()
        dir_name = str(unique_id)
        self.output_folder = f"filled_in_templates/result/{dir_name}"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.new_folder()



    def new_folder(self):
        for filename in os.listdir(self.path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                filepath = os.path.join(self.path, filename)
                image = Image.open(filepath)
                self.split_page(image)










    def split_page(self, img):
        width, height = img.size
        square_size = min(width, height) // 9
        # Crop the image into 117 squares
        for j in range(13):
            if self.count >= 243:
                return
            for i in range(9):
                # Calculate the coordinates of the crop box for the current square
                print(self.count)
                left = i * square_size
                upper = j * square_size
                right = left + square_size
                lower = upper + square_size
                # Crop the image using the current crop box
                cropped_img = img.crop((left, upper, right, lower))
                # Save the cropped image with a filename that includes the row and column numbers
                filename = f'{self.count}.jpg'
                filepath = os.path.join(self.output_folder, filename)
                cropped_img.save(filepath)
                self.count+=1


Elram = Split('filled_in_templates/elram/after_preprocessing')
