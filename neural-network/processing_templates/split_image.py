from PIL import Image
import  os
import uuid
import shutil


class Crop_every_image:
    def __init__(self, path):
        self.path = path
        self.crop_image()












    def crop_image (self):
        for filename in os.listdir(self.path):
            img = Image.open((os.path.join(self.path, filename)))
            width, height = img.size

            # calculate the amount of pixels to crop from each side
            crop_amount = int(min(width, height) * 0.05)

            # crop the image
            img = img.crop((crop_amount, crop_amount, width - crop_amount, height - crop_amount))

            # save the cropped image with the same name
            img.save(os.path.join(self.path, filename))






class Split:
    def __init__(self, path):
        print(path)
        self.count = 1
        self.path = path
        self.parent_path = os.path.dirname(path)
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

       # self.delete_folder(self.parent_path)


    def split_page(self, img):
        width, height = img.size
        square_size = min(width, height) // 9
        # Crop the image into 117 squares
        for j in range(13):
            if self.count >= 243:
                return
            for i in range(9):
                # Calculate the coordinates of the crop box for the current square

                left = i * square_size
                upper = j * square_size
                right = left + square_size
                lower = upper + square_size
                # Crop the image using the current crop box
                cropped_img = img.crop((left , upper, right, lower))
                # Save the cropped image with a filename that includes the row and column numbers

                filename = f'{self.count}.jpg'
                filepath = os.path.join(self.output_folder, filename)
                cropped_img.save(filepath)
                self.count+=1


       # cropEvery = Crop_every_image(self.output_folder)
        
    def delete_folder (self, path):
        try:
            # remove the folder and all its contents
            shutil.rmtree(path)
            print("Folder deleted successfully")
        except OSError as error:
            print(f"Error: {path} : {error.strerror}")






