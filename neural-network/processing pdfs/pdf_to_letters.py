import fitz  # PyMuPDF

# Open the PDF file
doc = fitz.open('neural-network\processing pdfs\elka-template.pdf')

# Iterate over each page in the PDF
for page in doc:
    # Extract the text from the page
    text = page.extract_text()
    print(text)
    # Split the text into individual letters
    letters = text.split()
    # Iterate over each letter and extract it as an image
    for letter in letters:
        # Get the bounding box of the letter
        bbox = page.search_for(letter)[0]
        # Extract the letter as an image
        letter_image = page.get_pixmap(alpha=False, clip=bbox)
        # Save the letter image to a file
        letter_image.save(
            "neural-network\processing pdfs/output-images"/f'{letter}.png')
