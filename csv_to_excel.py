import pandas as pd

# ---- CONFIG ----
INPUT_CSV = "runs/run_20260406_152155/realtime_grid_2circles_3zones_results.csv"   # اسم ملفك
OUTPUT_XLSX = "cv_results.xlsx"
# -----------------

df = pd.read_csv(INPUT_CSV)

df.to_excel(OUTPUT_XLSX, index=False)

print("Done.")
print("Excel file created:", OUTPUT_XLSX)
