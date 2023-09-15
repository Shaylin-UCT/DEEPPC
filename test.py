import markdown2pdf
import argparse

def convert_md_to_pdf(input_file, output_file):
    markdown2pdf.convert(input_file, output_file)

if __name__ == "__main__":
    input_file = "README.md"
    output_file = "REAMME.pdf"

    convert_md_to_pdf(input_file, output_file)