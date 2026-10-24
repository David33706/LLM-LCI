# Built the v3 eval workbook from the original question list. Both workbooks are kept
# in the author's private archive (not published: they contain verbatim paper text or
# licensed data). Without the input this script exits with a message and writes nothing.
import os
import sys

import pandas as pd

SRC = "eval/RAG_Questions_v2.xlsx"
if not os.path.exists(SRC):
    sys.exit(f"{SRC} not found: this script builds the private eval workbook from the "
             "original question list, which is not published.")

src = pd.read_excel(SRC, sheet_name="Sheet1")

def s(col):                       # force to clean string; NaN/float -> "" or text
    return src[col].apply(lambda v: "" if pd.isna(v) else str(v).strip())

out = pd.DataFrame({
    "id":              [str(i) for i in range(1, len(src) + 1)],
    "question":        s("Question"),
    "expected_answer": s("Answer"),       # was float (e.g. 0.9) -> now "0.9"
    "source_doi":      s("DOI"),
    "source_location": s("Location"),
})
out["type"] = out["source_location"].str.match(r"\s*table", case=False).map(
    {True: "TASK1_LOCATE", False: "TASK2_ASSUMPTION"})
out["notes"] = ""

with pd.ExcelWriter("eval/LCA_RAG_eval_set_v3.xlsx", engine="openpyxl") as w:
    out.to_excel(w, sheet_name="Questions", index=False)

print("wrote", len(out), "questions")
print("type mix:", out["type"].value_counts().to_dict())
print("all-string dtypes:", all(str(d) == "object" for d in out.dtypes))
print("blank answers:", int((out["expected_answer"] == "").sum()),
      "| blank dois:", int((out["source_doi"] == "").sum()))
