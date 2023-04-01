from PIL import Image
import cv2

template_path = 'neural-network/processing_templates/filled_in_templates/elram/elram-template-01.jpg'
# image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
img = Image.open(template_path)

width, height = img.size
chopsize = width // 9
print(chopsize)

# Save Chops of original image
for x0 in range(0, width, chopsize):
    for y0 in range(0, height, chopsize):
        box = (x0, y0,
               x0+chopsize if x0+chopsize < width else width - 1,
               y0+chopsize if y0+chopsize < height else height - 1)
        print('%s %s' % (template_path, box))
        img.crop(box).save('neural-network/processing_templates/results/letter.%s.x%03d.y%03d.jpg' %
                           ('elram-template-01', x0, y0))
