import pandas as pd, sys
sys.path.insert(0, '.')
from src.utils import PROJECT_ROOT, parse_level_str

gt = pd.read_csv(PROJECT_ROOT / 'data' / 'image_eval_human_gt.csv')
gt = gt[gt['image_generated'].astype(str).str.upper() == 'Y'].copy()
gt['level'] = gt['Lv.평가'].apply(parse_level_str)
gt['total'] = pd.to_numeric(gt['총점(만점 18점)'], errors='coerce')

for lv in ['L2', 'L3', 'L4', 'L5']:
    rows = gt[
        (gt['level'] == lv) &
        gt['비고'].notna() &
        (gt['비고'].astype(str).str.strip().str.len() > 10)
    ]
    print(f'=== {lv} ===')
    for _, r in rows.head(3).iterrows():
        print(f'총점={int(r["total"])} breed={r["breed"]} 비고={str(r["비고"]).strip()}')
    print()
