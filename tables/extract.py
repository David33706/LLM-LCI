import pandas as pd
from docling.document_converter import DocumentConverter

class ConverterExtractor:
    def __init__(self):
        self.converter = DocumentConverter()

    def convert(self, input_path, output_path=None):
        if output_path is not None:
            self.converter.convert(input_path, output_path)
        else:
            result = self.converter.convert(input_path)
        doc = result.document

        return doc
    
    def fetch_table(self, doc, table_ttl):
        for table_ix, table in enumerate(doc.tables):
            table_df: pd.DataFrame = table.export_to_dataframe(doc=doc)
            print(f"## Table {table_ix}")
            print(table_df.to_markdown())
    
if __name__ == "__main__":
    print("[INFO]: init converter...")
    converter = ConverterExtractor()
    print("[INFO]: converting document...")
    doc = converter.convert("/dkucc/home/nt140/LLM-LCI/criteria_classification/data/papers/Vinardell 2023 Environmental and economic evaluation of implement.pdf")
    converter.fetch_table(doc, "Table 1")