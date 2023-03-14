import PyPDF2
from PIL import Image


pdf_file = open('neural-network/processing pdfs/elka-template.pdf', 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)

# Iterate through each page of the PDF
for page_num in range(len(pdf_reader.pages)):
    page = pdf_reader.pages[page_num]

    page_width = int(page.mediabox.width)
    page_height = int(page.mediabox.height)

    # Calculate the width and height of each square
    square_width = page_width // 9
    square_height = page_height // 13

    for row in range(13):
        for col in range(9):

            # Calculate the coordinates of the top-left corner of the square
            x0 = col * square_width
            y0 = page_height - (row + 1) * square_height
            x1 = x0 + square_width
            y1 = y0 + square_height

            square = page.cropbox((x0, y0, x1, y1))

            # Save the square as an image
            image = square.convert('RGB')
            image.save(
                "neural-network/processing pdfs/output-images"f'page{page_num+1}_row{row+1}_col{col+1}.jpg')


pdf_file.close()
